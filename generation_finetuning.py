"""LigandGPT RL fine-tuning, rebuilt in one pass.

Per step: sample 32 molecules with the structured KV-cache decoder -> cheap terms and gates (SA, QED, torsion, energy, steric)
-> dock ONLY the molecules that pass -> reward -> PPO-style update with the behaviour log-prob stored at sampling time, a KL
penalty to the frozen prior and replay. Every sampled molecule enters the loss (valid: its reward, gate failed: 0.0,
invalid: -1.0). One stage function (mo_mpo.resolve_stage) decides weights, gates and metrics.

Run:  .venv/bin/python generation_finetuning.py --ckpt_load_path final.pt --targets 0 --n_steps 100 --seed 0 --out runs/probe --probe
"""
import argparse
import collections
import csv
import json
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor

import lmdb
import numpy as np
import torch
from rdkit import Chem

import decoding as dec
import mo_mpo
import scoring_function as sf
from data_utils import receptor_paths
from model import LigandGPT
from tokenizer import ProtLigTokenizer
from vocabulary import read_vocabulary

OBJECTIVES = ["docking", "bioactivity", "qed", "sa", "steric", "torsion", "energy"]
FIELDS = ["step", "stage", "n_valid", "n_gate_pass", "reward_mean", "docking_mean", "bioactivity_mean", "qed_mean", "sa_mean",
          "steric_mean", "torsion_mean", "energy_mean", "kl_to_prior", "entropy", "log_ratio_mean", "percent_in_band", "grad_norm",
          "replay_share", "archive_size", "wall_seconds"]


def get_lmdb_data(lmdb_path):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
    txn = env.begin()
    return txn, list(txn.cursor().iternext(values=False))


def sequence_log_probs(model, encoded_sequences, numbers_sequences, allowed=None):
    """Per-token log-probabilities of encoded_sequences[:, 1:] under `model`, with dropout DISABLED (eval mode).
    Every log-probability in the loss (agent, frozen prior, stored behaviour policy) comes from this one function.
    allowed [B, T-1, V] (bool): the support the sampler drew from at each step (grammar mask AND top-k). The distribution is renormalised over
    it, so the loss sees the distribution that was actually sampled from; without it the policy can lower the likelihood of its own samples by
    moving probability onto tokens the sampler never draws (observed: kl_to_prior fell to -34 in 10 steps, -12 in 5 with only the grammar mask).
    The caller's train/eval mode is restored. Gradients flow if autograd is on. Returns (token log-probs [B, T-1], full log-probs)."""
    was_training = model.training
    model.eval()
    try:
        logits, _ = model(encoded_sequences[:, :-1], numbers_sequences[:, :-1])
    finally:
        model.train(was_training)
    if allowed is not None:
        logits = logits.masked_fill(~allowed, float("-inf"))
    log_probabilities = torch.log_softmax(logits, dim=-1)
    token_log_probabilities = torch.gather(log_probabilities, 2, encoded_sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    return token_log_probabilities, log_probabilities


def pack(prompt_ids, prompt_nums, gens, masks=None):
    """prompt [1, L] with its numbers (the pocket coordinates enter the model through them: they MUST be the real ones, as in sampling)
    + generated token lists (+ their per-step support masks [g, V]) -> ids [m, T], numbers (prompt: real, generated: 1),
    mask [m, T-1] that is 1 on generated tokens only (the same mask is used for the likelihood, the entropy and the KL),
    allowed [m, T-1, V] bool or None."""
    L = prompt_ids.shape[1]
    T = L + max(len(g) for g in gens)
    ids = torch.zeros(len(gens), T, dtype=torch.long)
    mask = torch.zeros(len(gens), T - 1)
    allowed = None if masks is None else torch.ones(len(gens), T - 1, masks[0].shape[-1], dtype=torch.bool)
    nums = torch.ones(len(gens), T)
    nums[:, :L] = prompt_nums.cpu()
    ids[:, :L] = prompt_ids.cpu()
    for i, g in enumerate(gens):
        ids[i, L:L + len(g)] = torch.tensor(g)
        mask[i, L - 1:L - 1 + len(g)] = 1.0              # logits at position t predict token t+1
        if allowed is not None:
            allowed[i, L - 1:L - 1 + len(g)] = masks[i][:len(g)]
    return ids.cuda(), nums.cuda(), mask.cuda(), None if allowed is None else allowed.cuda()


@torch.no_grad()
def behaviour_logp(agent, prompt_ids, prompt_nums, gens, masks=None, chunk=8):
    """Sequence log-prob (generated tokens only) under the current agent, stored at sampling time."""
    out = []
    for s in range(0, len(gens), chunk):
        ids, nums, mask, allowed = pack(prompt_ids, prompt_nums, gens[s:s + chunk], None if masks is None else masks[s:s + chunk])
        lp, _ = sequence_log_probs(agent, ids, nums, allowed)
        out.append((lp * mask).sum(1))
    return torch.cat(out)


def valid_smiles(tokens, terminated):
    """Strictly valid: ended with <LIG_END>, RDKit parses and sanitizes the raw SMILES, >= 4 heavy atoms."""
    if not terminated:
        return None
    smi = "".join(tokens)
    mol = Chem.MolFromSmiles(smi) if smi else None
    return smi if mol is not None and mol.GetNumAtoms() >= 4 else None


class Normalizer:
    """Per-objective running normalisation, created once per run. Bounds start at the objective's static range and follow the
    batch 5th/95th percentile of the scored molecules with momentum 0.99. Maps a score to [0, 1], 1 = better."""

    def __init__(self, momentum=0.99):
        self.obj = {o.name: o for o in mo_mpo.create_objectives()}
        self.lo = {n: o.vmin for n, o in self.obj.items()}
        self.hi = {n: o.vmax for n, o in self.obj.items()}
        self.m = momentum

    def update(self, values):
        for n, v in values.items():
            if len(v) >= 3:
                p5, p95 = np.percentile(v, [5, 95])
                self.lo[n] = self.m * self.lo[n] + (1 - self.m) * p5
                self.hi[n] = self.m * self.hi[n] + (1 - self.m) * p95

    def __call__(self, name, x):
        if x is None:
            return 0.0
        z = float(np.clip((x - self.lo[name]) / max(self.hi[name] - self.lo[name], 1e-6), 0.0, 1.0))
        return 1.0 - z if self.obj[name].minimize else z


def score_batch(smiles, valid, th, ctx):
    """Cheap terms and gates first; dock only the molecules that pass. -> list of (status, scores) with
    status 'invalid' | 'gate_fail' | 'scored'. Results are cached by SMILES (the receptor is fixed per run)."""
    cache, counters = ctx["cache"], ctx["counters"]
    uniq = list(dict.fromkeys(s for s, v in zip(smiles, valid) if v))
    for s in uniq:
        if s not in cache:
            try:
                md = sf.prepare_molecule_with_conformer(s)
            except Exception:                                # e.g. RDKit raises in UFF for atom types without parameters: a 3D-build failure
                md = None
            p = sf.evaluate_property_scores(md, ctx["oracles"]["qed"], ctx["oracles"]["sa"]) if md is not None else None
            cache[s] = dict(qed=p and p.qed, sa=p and p.synthetic_accessibility, steric=p and p.steric_clash,
                            torsion=p and p.torsion_strain, energy=p and p.internal_energy)
    passers = [s for s in uniq if not mo_mpo.violates_hard_constraints(cache[s], th) and "docking" not in cache[s]]
    with ThreadPoolExecutor(sf.cpu_quota()) as pool:         # QVina is a subprocess: threads parallelise it, one job per CPU
        def dock(s):
            try:
                return sf.perform_qvina_docking(s, ctx["rec"], ctx["center"], ctx["tmp"])
            except Exception:
                return None
        for s, d in zip(passers, pool.map(dock, passers)):
            cache[s]["docking"] = d
    if passers:
        for s, b in zip(passers, sf.predict_bioactivity_for_batch(passers, ctx["gnn"])):
            cache[s]["bioactivity"] = b
    out = []
    for s, v in zip(smiles, valid):
        if not v:
            counters["invalid"] += 1
            out.append(("invalid", None))
            continue
        sc = cache[s]
        if mo_mpo.is_unscored(sc):
            counters["valid_no_3d"] += 1
            out.append(("gate_fail", sc))
        elif mo_mpo.violates_hard_constraints(sc, th):
            counters["gate_cheap"] += 1
            out.append(("gate_fail", sc))
        elif sc.get("docking") is None:
            counters["valid_dock_failed"] += 1
            out.append(("gate_fail", sc))
        else:
            counters["scored"] += 1
            out.append(("scored", sc))
    return out


def policy_update(agent, prior, optimizer, prompt_ids, prompt_nums, gens, masks, behaviour, adv, n_fresh, cfg):
    """One optimiser step on all sequences (fresh first, replay after), accumulated over micro-batches.
    loss = PPO clip on the sequence log-ratio (stored behaviour logp, clamp 0.7, clip 0.8-1.2)
           + kl_coef * KL(agent || prior) (k3 estimator, per generated token) - ent_coef * entropy (per generated token)."""
    N = len(gens)
    w = torch.where(adv > 0, torch.full_like(adv, cfg.pos_adv_weight), torch.ones_like(adv))
    W = w.sum()
    acc = collections.defaultdict(list)
    optimizer.zero_grad(set_to_none=True)
    for s in range(0, N, cfg.micro_batch):
        sl = slice(s, s + cfg.micro_batch)
        ids, nums, mask, allowed = pack(prompt_ids, prompt_nums, gens[sl], None if masks is None else masks[sl])
        lp, logp_full = sequence_log_probs(agent, ids, nums, allowed)
        with torch.no_grad():
            plp, _ = sequence_log_probs(prior, ids, nums, allowed)
        n_tok = mask.sum(1)
        seq = (lp * mask).sum(1)
        raw = seq - behaviour[sl]
        ratio = torch.exp(raw.clamp(-cfg.log_ratio_clamp, cfg.log_ratio_clamp))
        a = adv[sl]
        surrogate = -torch.min(ratio * a, ratio.clamp(0.8, 1.2) * a) * w[sl]
        logp_safe = logp_full if allowed is None else logp_full.masked_fill(~allowed, 0.0)       # 0 * -inf would be NaN
        entropy = (-(logp_safe.exp() * logp_safe).sum(-1) * mask).sum(1) / n_tok
        d = plp - lp
        kl = (((d.exp() - 1 - d) * mask).sum(1)) / n_tok
        loss = surrogate.sum() / W + (cfg.kl_coef * kl.sum() - cfg.ent_coef * entropy.sum()) / N
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            return dict(nan=True)
        loss.backward()
        acc["kl_seq"].append(((lp.detach() - plp) * mask).sum(1))
        acc["entropy"].append(entropy.detach())
        acc["raw"].append(raw.detach())
    grad_norm = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip))
    if not np.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        return dict(nan=True)
    optimizer.step()
    kl_seq, entropy, raw = (torch.cat(acc[k]) for k in ("kl_seq", "entropy", "raw"))
    ratio_fresh = torch.exp(raw[:n_fresh])
    in_band = ((ratio_fresh >= 0.8) & (ratio_fresh <= 1.2)).float().mean() * 100
    return dict(nan=False, kl_to_prior=float(kl_seq[:n_fresh].mean()), entropy=float(entropy.mean()), log_ratio_mean=float(raw.mean()),
                percent_in_band=float(in_band), grad_norm=grad_norm)


def probe_health(rows, n_batch):
    """PROBE HEALTHY / FAILED over the last 20 steps: validity > 20%, no NaN, fresh samples in band, replay <= 40%, docking not worse."""
    w, first = rows[-20:], rows[:20]
    validity = float(np.mean([r["n_valid"] / n_batch for r in w]))
    finite = all(np.isfinite([r[k] for k in ("kl_to_prior", "entropy", "grad_norm", "reward_mean")]).all() for r in w)
    band = float(np.mean([r["percent_in_band"] for r in w]))
    replay = max(r["replay_share"] for r in w)
    dock_w = float(np.nanmean([r["docking_mean"] for r in w]))
    dock_0 = float(np.nanmean([r["docking_mean"] for r in first]))
    worse = len(rows) > 20 and dock_w > dock_0 + 0.25
    problems = [m for m, bad in (("validity<=20%", validity <= 0.20), ("NaN", not finite), ("in_band<95", band < 95), ("replay>40%", replay > 0.40),
                                 ("docking worse", worse)) if bad]
    print(f"PROBE {'HEALTHY' if not problems else 'FAILED ' + ','.join(problems)} @ step {rows[-1]['step'] + 1}: validity {validity:.2f} | in_band {band:.1f}% | "
          f"max replay share {replay:.2f} | docking mean {dock_w:.2f} (first 20 steps {dock_0:.2f}) | kl {np.nanmean([r['kl_to_prior'] for r in w]):.4f} | "
          f"grad_norm {np.nanmean([r['grad_norm'] for r in w]):.2f}", flush=True)
    return not problems


def run_target(target, args, prior, voc, tk, txn, keys, split, oracles, gnn, tmp, state_dict):
    dp = pickle.loads(txn.get(keys[split[target]]))
    toks, nums = tk.tokenize(protein=dp["protein_atom_name"], prot_coords=dp["protein_pos"].numpy() - dp["ligand_center_of_mass"].numpy())
    toks.append("<LIG_START>")
    nums.append(1)
    prompt_ids = torch.tensor(voc.encode(toks), dtype=torch.long).unsqueeze(0).cuda()
    prompt_nums = torch.tensor(nums, dtype=torch.float).unsqueeze(0).cuda()
    L = prompt_ids.shape[1]
    budget = min(args.max_new_tokens, 2048 - L - 1)
    receptor_pdb, rec = receptor_paths(dp["protein_filename"], args.data_path)
    for required in (receptor_pdb, rec):
        if not os.path.exists(required):
            raise FileNotFoundError(f"receptor file missing: {required}")
    center = [float(x) for x in dp["ligand_center_of_mass"]]
    out_dir = os.path.join(args.out, f"target{target}")
    os.makedirs(out_dir, exist_ok=True)
    agent = LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()
    agent.load_state_dict(state_dict)
    agent.eval()
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.learning_rate, weight_decay=0.01)
    norm, objectives = Normalizer(), mo_mpo.create_objectives()
    ctx = dict(cache={}, counters=collections.Counter(), oracles=oracles, gnn=gnn, rec=rec, center=center, tmp=tmp)
    json.dump(dict(args=vars(args), prompt_tokens=L, token_budget=budget, receptor=rec, center=center,
                   thresholds={s: mo_mpo.get_constraint_thresholds(s) for s in (1, 2, 3)}, weights={s: mo_mpo.get_curriculum_weights(s) for s in (1, 2, 3)},
                   stage_boundaries=[args.n_steps // 3, 2 * args.n_steps // 3],
                   note="torsion 5/4/3 and energy 300/200/150 gate values are reconstructed (the lost originals could not be recovered); SA gate = baseline P90/P80/P70"),
              open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    print(f"target {target}: prompt {L} tokens, budget {budget}, stages at {args.n_steps // 3} and {2 * args.n_steps // 3}; "
          f"gate values torsion 5/4/3, energy 300/200/150 are RECONSTRUCTED, SA gate = baseline P90/P80/P70", flush=True)
    f = open(os.path.join(out_dir, "metrics.csv"), "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    rows, replay_buffer, archive, valid_hist, baseline, healthy = [], [], [], collections.deque(maxlen=20), None, True
    end_id = voc[dec.END]
    for step in range(args.n_steps):
        t0 = time.time()
        stage = mo_mpo.resolve_stage(step, args.n_steps)
        weights = mo_mpo.get_curriculum_weights(stage)
        th = mo_mpo.get_constraint_thresholds(stage, float(np.mean(valid_hist)) if len(valid_hist) >= 20 else 1.0)
        # ---- sample (structured decoder, KV cache) and store the behaviour log-prob
        rows_ids, _, step_masks = dec.sample_smiles(agent, voc, prompt_ids, prompt_nums, args.batch_size, structured=True, valence=True, top_k=10,
                                                    temperature=1.0, max_new_tokens=budget, return_masks=True)
        gens, gen_masks, valid, smiles = [], [], [], []
        for i, r in enumerate(rows_ids):
            term = end_id in r
            g = r[:r.index(end_id) + 1] if term else r
            gen_masks.append(step_masks[i, :len(g)].clone())
            s = valid_smiles(voc.decode(g[:-1] if term else g), term)
            gens.append(g)
            valid.append(s is not None)
            smiles.append(s or "")
        behaviour = behaviour_logp(agent, prompt_ids, prompt_nums, gens, gen_masks)
        # ---- score: cheap terms and gates first, dock the survivors
        results = score_batch(smiles, valid, th, ctx)
        scored = [(i, sc) for i, (st, sc) in enumerate(results) if st == "scored"]
        norm.update({n: [sc[n] for _, sc in scored if sc.get(n) is not None] for n in OBJECTIVES})
        rewards = np.array([-1.0 if st == "invalid" else (0.0 if st == "gate_fail" else sum(weights[n] * norm(n, sc.get(n)) for n in weights))
                            for st, sc in results])
        not_invalid = np.array([st != "invalid" for st, _ in results])
        if not_invalid.any():                                   # baseline excludes invalid molecules by status (not by reward value)
            m = float(rewards[not_invalid].mean())
            baseline = m if baseline is None else 0.9 * baseline + 0.1 * m
        baseline = 0.0 if baseline is None else baseline
        adv_fresh = np.clip(rewards - baseline, -3.0, 3.0)      # clipped once
        # ---- replay: random sample from the top 32 by reward, <= 40% of the sequences, advantage recomputed against the current baseline
        top = sorted(replay_buffer, key=lambda e: -e["reward"])[:32]
        k = min(args.replay, len(top), int(0.4 / 0.6 * len(gens)))
        chosen = random.sample(top, k) if k else []
        all_gens = gens + [e["gen"] for e in chosen]
        all_masks = gen_masks + [e["mask"] for e in chosen]
        all_beh = torch.cat([behaviour, torch.tensor([e["behaviour"] for e in chosen], device=behaviour.device)]) if chosen else behaviour
        all_adv = torch.tensor(np.concatenate([adv_fresh, np.clip([e["reward"] - baseline for e in chosen], -3.0, 3.0)]), dtype=torch.float).cuda()
        stats = policy_update(agent, prior, optimizer, prompt_ids, prompt_nums, all_gens, all_masks, all_beh, all_adv, len(gens), args)
        # ---- bookkeeping: replay buffer, Pareto archive (only the size cap prunes it, never a stage change), metrics
        for i, sc in scored:
            if rewards[i] > 0:
                replay_buffer.append(dict(smiles=smiles[i], gen=gens[i], mask=gen_masks[i], behaviour=float(behaviour[i]), reward=float(rewards[i])))
            entry = {n: sc.get(n) for n in OBJECTIVES}
            entry["smiles"] = smiles[i]
            dominated, kept = mo_mpo.filter_dominated_solutions_from_archive(entry, archive, objectives)
            if not dominated and not any(a["smiles"] == smiles[i] for a in archive):
                archive = kept + [entry]
                if len(archive) > 100:
                    archive = mo_mpo.prune_archive_by_hypervolume(archive, objectives, 100)
        best = {}
        for e in sorted(replay_buffer, key=lambda e: -e["reward"]):
            best.setdefault(e["smiles"], e)
        replay_buffer = sorted(best.values(), key=lambda e: -e["reward"])[:100]
        valid_hist.append(sum(valid) / len(valid))
        mean_of = lambda n: float(np.mean([sc[n] for _, sc in scored if sc.get(n) is not None])) if scored else float("nan")  # noqa: E731
        nan = stats.get("nan", False)
        stat = lambda k_: float("nan") if nan else stats[k_]  # noqa: E731
        row = dict(step=step, stage=stage, n_valid=sum(valid), n_gate_pass=len(scored), reward_mean=float(rewards.mean()), docking_mean=mean_of("docking"),
                   bioactivity_mean=mean_of("bioactivity"), qed_mean=mean_of("qed"), sa_mean=mean_of("sa"), steric_mean=mean_of("steric"),
                   torsion_mean=mean_of("torsion"), energy_mean=mean_of("energy"), kl_to_prior=stat("kl_to_prior"), entropy=stat("entropy"),
                   log_ratio_mean=stat("log_ratio_mean"), percent_in_band=stat("percent_in_band"), grad_norm=stat("grad_norm"),
                   replay_share=len(chosen) / len(all_gens), archive_size=len(archive), wall_seconds=time.time() - t0)
        rows.append(row)
        writer.writerow({k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in row.items()})
        f.flush()
        print(f"step {step:3d} stage {stage} valid {row['n_valid']:2d}/{len(gens)} pass {row['n_gate_pass']:2d} reward {row['reward_mean']:+.3f} "
              f"dock {row['docking_mean']:+.2f} kl {row['kl_to_prior']:.4f} band {row['percent_in_band']:.0f}% replay {row['replay_share']:.2f} {row['wall_seconds']:.0f}s", flush=True)
        if (step + 1) % 50 == 0 or step + 1 == args.n_steps:
            print(f"SUMMARY step {step + 1}: counters {dict(ctx['counters'])} | archive {len(archive)} | replay buffer {len(replay_buffer)} | baseline {baseline:.3f}", flush=True)
            ctx["counters"].clear()
        if args.probe and (step + 1) % 20 == 0:
            healthy = probe_health(rows, len(gens)) and healthy
        if step % 10 == 0:
            torch.cuda.empty_cache()
    f.close()
    json.dump(archive, open(os.path.join(out_dir, "archive.json"), "w"), indent=1, default=float)
    return healthy


def main():
    ap = argparse.ArgumentParser(description="LigandGPT RL fine-tuning")
    ap.add_argument("--ckpt_load_path", default="final.pt")
    ap.add_argument("--vocab_path", default="./ProtLigVoc.txt")
    ap.add_argument("--data_path", default="./data")
    ap.add_argument("--out", default="runs/rl")
    ap.add_argument("--targets", type=int, nargs="+", default=[0])
    ap.add_argument("--n_steps", type=int, default=700)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--replay", type=int, default=8, help="replay sequences per step (capped so replay is <= 40%% of the sequences)")
    ap.add_argument("--pos_adv_weight", type=float, default=1.0)
    ap.add_argument("--kl_coef", type=float, default=1.0)
    ap.add_argument("--ent_coef", type=float, default=0.005)
    ap.add_argument("--log_ratio_clamp", type=float, default=0.7)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true", help="print PROBE HEALTHY/FAILED every 20 steps")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    voc, tk = read_vocabulary(args.vocab_path), ProtLigTokenizer()
    state_dict = torch.load(args.ckpt_load_path)
    prior = LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()
    prior.load_state_dict(state_dict)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    print(f"parameters: {sum(p.numel() for p in prior.parameters()):,}", flush=True)
    txn, keys = get_lmdb_data(os.path.join(args.data_path, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"))
    split = torch.load(os.path.join(args.data_path, "crossdocked_pocket10_pose_split.pt"), weights_only=False)["test"]
    oracles, gnn, tmp = sf.initialize_scoring_system()
    if gnn is None:
        raise RuntimeError("bioactivity GNN failed to load")
    ok = True
    for t in args.targets:
        ok = run_target(t, args, prior, voc, tk, txn, keys, split, oracles, gnn, tmp, state_dict) and ok
    if args.probe:
        print("PROBE HEALTHY (all windows)" if ok else "PROBE FAILED (at least one window failed)", flush=True)


if __name__ == "__main__":
    main()
