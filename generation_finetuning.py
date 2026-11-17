"""LigandGPT RL fine-tuning — rebuilt experiment layer.

Arms: no_curriculum, curriculum_weights, curriculum_pairs, adaptive.
Common 75-step warmup (V + Z only) for every arm.
Fixed per-target reward bounds from prior calibration.
Resume from atomic rolling checkpoint.
"""
import argparse
import collections
import copy
import csv
import hashlib
import json
import math
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

OBJECTIVES_ORDER = ["docking", "bioactivity", "qed", "sa", "energy", "torsion"]

FIELDS = [
    "step", "phase", "stage", "n_valid", "n_total", "validity",
    "heavy_atoms_mean", "size_hit_rate",
    "docking_mean", "docking_norm_mean", "bioactivity_mean", "bioactivity_norm_mean",
    "qed_mean", "qed_norm_mean", "sa_mean", "sa_norm_mean",
    "energy_mean", "energy_norm_mean", "torsion_mean", "torsion_norm_mean",
    "steric_mean",
    "reward_mean", "advantage_mean", "kl_to_prior", "entropy",
    "log_ratio_mean", "percent_in_band", "grad_norm",
    "replay_fraction", "baseline_value",
    "dock_fail_embed", "dock_fail_meeko", "dock_fail_nopose", "dock_fail_timeout",
    "wall_seconds",
]


# ---------------------------------------------------------------------------
# Reward normalisation: FIXED per-target bounds
# ---------------------------------------------------------------------------
class FixedNormalizer:
    """Normalise each objective to [0,1] using fixed bounds (P5-P95 of prior).
    Higher is always better after normalisation."""

    def __init__(self, bounds: dict):
        self.bounds = bounds  # {obj_name: (lo, hi, minimize)}

    def __call__(self, name, x):
        if x is None:
            return 0.0
        lo, hi, minimize = self.bounds[name]
        span = max(hi - lo, 1e-6)
        n = float(np.clip((x - lo) / span, 0.0, 1.0))
        return 1.0 - n if minimize else n

    @staticmethod
    def load(path):
        d = json.load(open(path))
        bounds = {}
        for name, info in d["bounds"].items():
            bounds[name] = (info["lo"], info["hi"], info["minimize"])
        return FixedNormalizer(bounds)


def default_bounds_for_target(target_id):
    """Hard-coded calibrated bounds for target 0 (from earlier runs). Other targets need calibration."""
    if target_id == 0:
        return {
            "docking":     {"lo": -8.96,  "hi": -4.94,  "minimize": True},
            "bioactivity": {"lo": -9.828, "hi": -5.186, "minimize": True},
            "qed":         {"lo": 0.094,  "hi": 0.778,  "minimize": False},
            "sa":          {"lo": 3.70,   "hi": 8.56,   "minimize": True},
            "energy":      {"lo": 14.8,   "hi": 578.0,  "minimize": True},
            "torsion":     {"lo": 0.37,   "hi": 4.465,  "minimize": True},
        }
    return None


# ---------------------------------------------------------------------------
# Size term Z
# ---------------------------------------------------------------------------
def size_term(n_heavy):
    if n_heavy is None or n_heavy < 8 or n_heavy > 48:
        return 0.0
    if 18 <= n_heavy <= 38:
        return 1.0
    if n_heavy < 18:
        return max(0.0, (n_heavy - 8) / 10.0)
    return max(0.0, (48 - n_heavy) / 10.0)


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
ARM_NAMES = ["no_curriculum", "curriculum_weights", "curriculum_pairs", "adaptive"]

STAGE_WEIGHTS_CW = {
    1: {"docking": 0.60, "bioactivity": 0.06, "qed": 0.07, "sa": 0.07, "energy": 0.14, "torsion": 0.06},
    2: {"docking": 0.40, "bioactivity": 0.14, "qed": 0.13, "sa": 0.13, "energy": 0.14, "torsion": 0.06},
    3: {"docking": 0.35, "bioactivity": 0.09, "qed": 0.08, "sa": 0.08, "energy": 0.28, "torsion": 0.12},
}
FIXED_WEIGHTS_NC = {"docking": 0.450, "bioactivity": 0.097, "qed": 0.093, "sa": 0.093, "energy": 0.187, "torsion": 0.080}

PAIRS_STAGES = {
    1: ["docking", "bioactivity"],
    2: ["docking", "bioactivity", "qed", "sa"],
    3: ["docking", "bioactivity", "qed", "sa", "torsion", "energy"],
}
ADAPTIVE_ORDER = ["docking", "bioactivity", "qed", "sa", "torsion", "energy"]


def compute_quality(arm, stage, active_objs, norm, scores):
    """Compute QUALITY term for a single scored molecule."""
    if arm == "no_curriculum":
        return 9.0 * sum(FIXED_WEIGHTS_NC[o] * norm(o, scores.get(o)) for o in OBJECTIVES_ORDER)
    elif arm == "curriculum_weights":
        w = STAGE_WEIGHTS_CW[stage]
        return 9.0 * sum(w[o] * norm(o, scores.get(o)) for o in OBJECTIVES_ORDER)
    elif arm == "curriculum_pairs":
        return sum(1.5 * norm(o, scores.get(o)) for o in active_objs)
    elif arm == "adaptive":
        return sum(1.5 * norm(o, scores.get(o)) for o in active_objs)
    return 0.0


def quality_divisor(arm, stage, active_objs):
    """D value = 1.0 + 1.5 + sum of active quality coefficients."""
    base = 1.0 + 1.5  # V + Z
    if arm == "no_curriculum":
        return base + 9.0
    elif arm == "curriculum_weights":
        return base + 9.0
    elif arm == "curriculum_pairs":
        return base + 1.5 * len(active_objs)
    elif arm == "adaptive":
        return base + 1.5 * len(active_objs)
    return base


# ---------------------------------------------------------------------------
# Core functions (kept from previous version with minimal changes)
# ---------------------------------------------------------------------------
def get_lmdb_data(lmdb_path):
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
    txn = env.begin()
    return txn, list(txn.cursor().iternext(values=False))


def sequence_log_probs(model, encoded_sequences, numbers_sequences, allowed=None):
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
        mask[i, L - 1:L - 1 + len(g)] = 1.0
        if allowed is not None:
            allowed[i, L - 1:L - 1 + len(g)] = masks[i][:len(g)]
    return ids.cuda(), nums.cuda(), mask.cuda(), None if allowed is None else allowed.cuda()


@torch.no_grad()
def behaviour_logp(agent, prompt_ids, prompt_nums, gens, masks=None, chunk=8):
    out = []
    for s in range(0, len(gens), chunk):
        ids, nums, mask, allowed = pack(prompt_ids, prompt_nums, gens[s:s + chunk], None if masks is None else masks[s:s + chunk])
        lp, _ = sequence_log_probs(agent, ids, nums, allowed)
        out.append((lp * mask).sum(1))
    return torch.cat(out)


def valid_smiles(tokens, terminated):
    if not terminated:
        return None
    smi = "".join(tokens)
    mol = Chem.MolFromSmiles(smi) if smi else None
    return smi if mol is not None and mol.GetNumAtoms() >= 4 else None


def policy_update(agent, prior, optimizer, prompt_ids, prompt_nums, gens, masks, behaviour, adv, n_fresh, cfg):
    N = len(gens)
    W = float(N)
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
        surrogate = -torch.min(ratio * a, ratio.clamp(0.8, 1.2) * a)
        logp_safe = logp_full if allowed is None else logp_full.masked_fill(~allowed, 0.0)
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


def reserve_gpu(gib=4.2):
    while True:
        try:
            torch.empty(int(gib * 2**30), dtype=torch.uint8, device="cuda")
            return
        except torch.OutOfMemoryError:
            time.sleep(5)


# ---------------------------------------------------------------------------
# Scoring (gates are LOG ONLY — all valid molecules get scored and docked)
# ---------------------------------------------------------------------------
def score_batch(smiles_list, valid_flags, ctx):
    """Score all valid molecules. Gates are log-only. Returns list of (status, scores_dict, dock_fail_cause)."""
    cache = ctx["cache"]
    uniq = list(dict.fromkeys(s for s, v in zip(smiles_list, valid_flags) if v))
    # cheap scores
    for s in uniq:
        if s not in cache:
            try:
                md = sf.prepare_molecule_with_conformer(s)
            except Exception:
                md = None
            p = sf.evaluate_property_scores(md, ctx["oracles"]["qed"], ctx["oracles"]["sa"]) if md is not None else None
            cache[s] = dict(qed=p and p.qed, sa=p and p.synthetic_accessibility,
                            steric=p and p.steric_clash, torsion=p and p.torsion_strain,
                            energy=p and p.internal_energy,
                            _has_conformer=md is not None and md.has_conformer if md else False)
    # dock ALL valid molecules (no gate filtering)
    to_dock = [s for s in uniq if "docking" not in cache[s]]
    dock_fails = collections.Counter()
    with ThreadPoolExecutor(sf.cpu_quota()) as pool:
        def dock_one(s):
            try:
                return sf.perform_qvina_docking(s, ctx["rec"], ctx["center"], ctx["tmp"])
            except Exception:
                return None
        for s, d in zip(to_dock, pool.map(dock_one, to_dock)):
            cache[s]["docking"] = d
            if d is None:
                if not cache[s].get("_has_conformer"):
                    dock_fails["embed"] += 1
                else:
                    dock_fails["nopose"] += 1
    # bioactivity for all valid
    bio_need = [s for s in uniq if "bioactivity" not in cache[s]]
    if bio_need:
        for s, b in zip(bio_need, sf.predict_bioactivity_for_batch(bio_need, ctx["gnn"])):
            cache[s]["bioactivity"] = b
    out = []
    for s, v in zip(smiles_list, valid_flags):
        if not v:
            out.append(("invalid", None, None))
            continue
        sc = cache[s]
        out.append(("scored", sc, None))
    return out, dock_fails


# ---------------------------------------------------------------------------
# Checkpoint / Resume
# ---------------------------------------------------------------------------
def save_checkpoint(path, step, agent, optimizer, rng_state, baseline, stage, arm_state, replay_buffer):
    torch.save({
        "step": step,
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "baseline": baseline,
        "stage": stage,
        "arm_state": arm_state,
        "replay_buffer": [(e["smiles"], e["gen"], e["behaviour"], e["reward"]) for e in replay_buffer],
    }, path + ".tmp", _use_new_zipfile_serialization=True)
    os.replace(path + ".tmp", path)


def load_checkpoint(path, agent, optimizer):
    ckpt = torch.load(path, weights_only=False)
    agent.load_state_dict(ckpt["agent"])
    optimizer.load_state_dict(ckpt["optimizer"])
    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    torch.random.set_rng_state(ckpt["rng_torch"])
    torch.cuda.set_rng_state(ckpt["rng_cuda"])
    replay = [dict(smiles=s, gen=g, behaviour=b, reward=r) for s, g, b, r in ckpt["replay_buffer"]]
    return ckpt["step"], ckpt["baseline"], ckpt["stage"], ckpt["arm_state"], replay


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def run_target(target, args, prior, voc, tk, txn, keys, split, oracles, gnn, tmp, state_dict):
    dp = pickle.loads(txn.get(keys[split[target]]))
    toks, nums = tk.tokenize(protein=dp["protein_atom_name"],
                             prot_coords=dp["protein_pos"].numpy() - dp["ligand_center_of_mass"].numpy())
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

    # Load normaliser
    bounds_path = os.path.join(args.out, f"calibrated_bounds_target{target}.json")
    if not os.path.exists(bounds_path):
        bounds_path = os.path.join("runs", "baseline", f"calibrated_bounds_target{target}.json")
    if os.path.exists(bounds_path):
        norm = FixedNormalizer.load(bounds_path)
        print(f"target {target}: loaded bounds from {bounds_path}", flush=True)
    else:
        db = default_bounds_for_target(target)
        if db is None:
            raise FileNotFoundError(f"No calibrated bounds for target {target}. Run calibrate_bounds first.")
        bounds = {k: (v["lo"], v["hi"], v["minimize"]) for k, v in db.items()}
        norm = FixedNormalizer(bounds)
        print(f"target {target}: using hard-coded bounds", flush=True)

    agent = LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()
    agent.load_state_dict(state_dict)
    agent.eval()
    optimizer = torch.optim.AdamW(agent.parameters(), lr=args.learning_rate, weight_decay=0.01)
    ctx = dict(cache={}, oracles=oracles, gnn=gnn, rec=rec, center=center, tmp=tmp)

    # Arm config
    arm = args.arm
    warmup_steps = args.warmup_steps
    total_steps = args.n_steps
    stage_len = args.stage_length

    # adaptive state
    arm_state = {
        "active_objs": [],
        "phase_start": 0,
        "introductions": [],
    }
    if arm == "adaptive":
        arm_state["active_objs"] = [ADAPTIVE_ORDER[0]]
    stage = 0  # 0 = warmup

    replay_buffer = []
    baseline = None
    start_step = 0
    valid_ma = collections.deque(maxlen=15)

    # Resume
    ckpt_a = os.path.join(out_dir, "ckpt_a.pt")
    ckpt_b = os.path.join(out_dir, "ckpt_b.pt")
    if args.resume:
        for cp in (ckpt_b, ckpt_a):
            if os.path.exists(cp):
                try:
                    start_step, baseline, stage, arm_state, replay_buffer = load_checkpoint(cp, agent, optimizer)
                    start_step += 1
                    print(f"RESUMED from {cp} at step {start_step}", flush=True)
                    break
                except Exception as e:
                    print(f"WARNING: failed to load {cp}: {e}", flush=True)

    # Config
    config = dict(
        args=vars(args), prompt_tokens=L, token_budget=budget, receptor=rec, center=center,
        arm=arm, warmup_steps=warmup_steps, total_steps=total_steps, stage_length=stage_len,
        bounds=bounds_path if os.path.exists(bounds_path) else "hard-coded",
    )
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    # Run meta
    try:
        git_commit = os.popen("git -C " + os.path.dirname(os.path.abspath(__file__)) + " rev-parse HEAD 2>/dev/null").read().strip()
    except Exception:
        git_commit = "unknown"
    run_meta = dict(
        git_commit=git_commit, seed=args.seed, target=target,
        pocket_id=dp["protein_filename"],
        torch_version=torch.__version__, numpy_version=np.__version__,
    )
    json.dump(run_meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=1)

    # CSV
    csv_path = os.path.join(out_dir, "metrics.csv")
    csv_existed = os.path.exists(csv_path) and args.resume
    f_csv = open(csv_path, "a" if csv_existed else "w", newline="")
    writer = csv.DictWriter(f_csv, fieldnames=FIELDS)
    if not csv_existed:
        writer.writeheader()

    # Detail log
    detail_path = os.path.join(out_dir, "detail.jsonl")
    f_detail = open(detail_path, "a" if csv_existed else "w")

    end_id = voc[dec.END]

    print(f"target {target}: prompt {L}, budget {budget}, arm={arm}, warmup={warmup_steps}, total={total_steps}", flush=True)

    for step in range(start_step, total_steps):
        t0 = time.time()
        in_warmup = step < warmup_steps

        # Determine phase/stage and active objectives
        if in_warmup:
            phase = "warmup"
            active_objs = []
            cur_stage = 0
        else:
            phase = arm
            post_warmup_step = step - warmup_steps
            if arm == "no_curriculum":
                cur_stage = 1
                active_objs = list(OBJECTIVES_ORDER)
            elif arm == "curriculum_weights":
                cur_stage = 1 + post_warmup_step // stage_len
                cur_stage = min(cur_stage, 3)
                active_objs = list(OBJECTIVES_ORDER)
            elif arm == "curriculum_pairs":
                cur_stage = 1 + post_warmup_step // stage_len
                cur_stage = min(cur_stage, 3)
                active_objs = list(PAIRS_STAGES[cur_stage])
            elif arm == "adaptive":
                active_objs = list(arm_state["active_objs"])
                cur_stage = len(active_objs)

        # Detect stage switch (reset baseline & replay)
        prev_stage = stage
        stage = cur_stage if not in_warmup else 0
        if step == warmup_steps or (not in_warmup and stage != prev_stage and step > warmup_steps):
            baseline = None
            replay_buffer = []
            print(f"  step {step}: stage switch {prev_stage}->{stage}, baseline & replay reset", flush=True)

        # ---- Sample ----
        rows_ids, _, step_masks = dec.sample_smiles(
            agent, voc, prompt_ids, prompt_nums, args.batch_size,
            structured=True, valence=True, top_k=10, temperature=1.0,
            max_new_tokens=budget, return_masks=True)
        gens, gen_masks, valid, smiles_list = [], [], [], []
        for i, r in enumerate(rows_ids):
            term = end_id in r
            g = r[:r.index(end_id) + 1] if term else r
            gen_masks.append(step_masks[i, :len(g)].clone())
            s = valid_smiles(voc.decode(g[:-1] if term else g), term)
            gens.append(g)
            valid.append(s is not None)
            smiles_list.append(s or "")
        beh = behaviour_logp(agent, prompt_ids, prompt_nums, gens, gen_masks)

        # ---- Score ----
        results, dock_fails = score_batch(smiles_list, valid, ctx)

        # Compute rewards
        rewards = np.zeros(len(results))
        details = []
        for i, (st, sc, _) in enumerate(results):
            mol = Chem.MolFromSmiles(smiles_list[i]) if valid[i] else None
            n_heavy = mol.GetNumHeavyAtoms() if mol else None
            V = 1.0 if valid[i] else 0.0
            Z = size_term(n_heavy)

            if in_warmup:
                R = 1.0 * V + 1.5 * Z
            elif st == "invalid":
                R = 0.0
            else:
                quality = compute_quality(arm, stage, active_objs, norm, sc)
                R = 1.0 * V + 1.5 * Z + quality

            rewards[i] = R
            detail = dict(step=step, idx=i, smiles=smiles_list[i], valid=valid[i],
                          n_heavy=n_heavy, V=V, Z=Z, reward=float(R))
            if sc is not None:
                for o in OBJECTIVES_ORDER:
                    detail[o] = sc.get(o)
                    detail[o + "_norm"] = norm(o, sc.get(o))
                detail["steric"] = sc.get("steric")
            details.append(detail)

        # Advantage
        D = 2.5 if in_warmup else quality_divisor(arm, stage, active_objs)
        m = float(rewards.mean())
        baseline = m if baseline is None else 0.9 * baseline + 0.1 * m
        adv_raw = (rewards - baseline) / D

        # z-score of quality part among valid (post-warmup only)
        if not in_warmup:
            qual_vals = []
            for i, (st, sc, _) in enumerate(results):
                if valid[i] and sc is not None:
                    qual_vals.append(compute_quality(arm, stage, active_objs, norm, sc))
            if len(qual_vals) >= 2:
                qmean, qstd = np.mean(qual_vals), max(np.std(qual_vals), 1e-6)
                z_scores = np.zeros(len(results))
                qi = 0
                for i, (st, sc, _) in enumerate(results):
                    if valid[i] and sc is not None:
                        z_scores[i] = (qual_vals[qi] - qmean) / qstd
                        qi += 1
                adv_raw = adv_raw + 0.5 * z_scores

        adv_fresh = np.clip(adv_raw, -3.0, 3.0)

        # ---- Replay ----
        top = sorted(replay_buffer, key=lambda e: -e["reward"])[:32]
        k = min(args.replay, len(top), int(0.4 / 0.6 * len(gens)))
        chosen = random.sample(top, k) if k else []
        all_gens = gens + [e["gen"] for e in chosen]
        all_masks = gen_masks + [e.get("mask") for e in chosen]
        all_beh = torch.cat([beh, torch.tensor([e["behaviour"] for e in chosen], device=beh.device)]) if chosen else beh
        replay_adv = np.clip([(e["reward"] - baseline) / D for e in chosen], -3.0, 3.0) if chosen else []
        all_adv = torch.tensor(np.concatenate([adv_fresh, replay_adv]) if chosen else adv_fresh, dtype=torch.float).cuda()

        # ---- Policy update ----
        try:
            stats = policy_update(agent, prior, optimizer, prompt_ids, prompt_nums,
                                  all_gens, all_masks, all_beh, all_adv, len(gens), args)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            print(f"  step {step}: OOM in policy_update, retrying", flush=True)
            stats = policy_update(agent, prior, optimizer, prompt_ids, prompt_nums,
                                  all_gens, all_masks, all_beh, all_adv, len(gens), args)

        # ---- Replay buffer update ----
        for i, (st, sc, _) in enumerate(results):
            if valid[i] and rewards[i] > 0:
                entry = dict(smiles=smiles_list[i], gen=gens[i], mask=gen_masks[i],
                             behaviour=float(beh[i]), reward=float(rewards[i]))
                replay_buffer.append(entry)
        best = {}
        for e in sorted(replay_buffer, key=lambda e: -e["reward"]):
            best.setdefault(e["smiles"], e)
        replay_buffer = sorted(best.values(), key=lambda e: -e["reward"])[:100]

        # ---- Validity MA ----
        v_rate = sum(valid) / len(valid)
        valid_ma.append(v_rate)
        if len(valid_ma) >= 15 and step >= warmup_steps and np.mean(valid_ma) < 0.25:
            print(f"  WARNING: validity collapse! MA(15)={np.mean(valid_ma):.3f} at step {step}", flush=True)

        # ---- Adaptive: check introduction ----
        if arm == "adaptive" and not in_warmup and len(arm_state["active_objs"]) < len(ADAPTIVE_ORDER):
            newest_obj = arm_state["active_objs"][-1]
            steps_in_phase = step - arm_state["phase_start"]
            if steps_in_phase >= args.adaptive_min_phase:
                # check metric
                # We need prior stats for threshold - use bounds
                lo, hi, minimize = norm.bounds[newest_obj]
                prior_mean = (lo + hi) / 2
                prior_sd = (hi - lo) / 3.29  # P5-P95 ~ 3.29 SD
                threshold = prior_mean - 0.25 * prior_sd if minimize else prior_mean + 0.25 * prior_sd
                # average over last 15 steps from details
                # simplified: use current step's mean
                recent_vals = [d[newest_obj] for d in details if d.get("valid") and d.get(newest_obj) is not None]
                if recent_vals:
                    metric_val = np.mean(recent_vals)
                    beats = (metric_val < threshold) if minimize else (metric_val > threshold)
                    forced = steps_in_phase >= args.adaptive_forced_intro
                    if beats or forced:
                        next_idx = len(arm_state["active_objs"])
                        if next_idx < len(ADAPTIVE_ORDER):
                            new_obj = ADAPTIVE_ORDER[next_idx]
                            arm_state["active_objs"].append(new_obj)
                            arm_state["phase_start"] = step
                            arm_state["introductions"].append(dict(
                                step=step, objective=new_obj,
                                metric_value=float(metric_val), threshold=float(threshold),
                                forced=forced
                            ))
                            baseline = None
                            replay_buffer = []
                            print(f"  step {step}: ADAPTIVE introduced {new_obj} (metric={metric_val:.3f}, threshold={threshold:.3f}, forced={forced})", flush=True)
                            # Save checkpoint at introduction
                            torch.save(agent.state_dict(), os.path.join(out_dir, f"ckpt_intro_{new_obj}_step{step}.pt"))

        # ---- Metrics ----
        scored = [(i, sc) for i, (st, sc, _) in enumerate(results) if st == "scored" and sc is not None]
        heavy_valid = []
        for i in range(len(smiles_list)):
            if valid[i]:
                mol = Chem.MolFromSmiles(smiles_list[i])
                if mol:
                    heavy_valid.append(mol.GetNumHeavyAtoms())
        size_hits = sum(1 for h in heavy_valid if 18 <= h <= 38)
        mean_of = lambda n: float(np.mean([sc[n] for _, sc in scored if sc.get(n) is not None])) if scored else float("nan")
        norm_mean_of = lambda n: float(np.mean([norm(n, sc.get(n)) for _, sc in scored if sc.get(n) is not None])) if scored else float("nan")
        nan = stats.get("nan", False)
        stat = lambda k_: float("nan") if nan else stats[k_]

        row = dict(
            step=step, phase=phase, stage=stage,
            n_valid=sum(valid), n_total=len(valid),
            validity=v_rate,
            heavy_atoms_mean=float(np.mean(heavy_valid)) if heavy_valid else float("nan"),
            size_hit_rate=size_hits / max(len(heavy_valid), 1),
            docking_mean=mean_of("docking"), docking_norm_mean=norm_mean_of("docking"),
            bioactivity_mean=mean_of("bioactivity"), bioactivity_norm_mean=norm_mean_of("bioactivity"),
            qed_mean=mean_of("qed"), qed_norm_mean=norm_mean_of("qed"),
            sa_mean=mean_of("sa"), sa_norm_mean=norm_mean_of("sa"),
            energy_mean=mean_of("energy"), energy_norm_mean=norm_mean_of("energy"),
            torsion_mean=mean_of("torsion"), torsion_norm_mean=norm_mean_of("torsion"),
            steric_mean=mean_of("steric"),
            reward_mean=float(rewards.mean()),
            advantage_mean=float(adv_fresh.mean()),
            kl_to_prior=stat("kl_to_prior"), entropy=stat("entropy"),
            log_ratio_mean=stat("log_ratio_mean"), percent_in_band=stat("percent_in_band"),
            grad_norm=stat("grad_norm"),
            replay_fraction=len(chosen) / len(all_gens),
            baseline_value=baseline or 0.0,
            dock_fail_embed=dock_fails.get("embed", 0),
            dock_fail_meeko=dock_fails.get("meeko", 0),
            dock_fail_nopose=dock_fails.get("nopose", 0),
            dock_fail_timeout=dock_fails.get("timeout", 0),
            wall_seconds=time.time() - t0,
        )
        writer.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        f_csv.flush()

        for d in details:
            f_detail.write(json.dumps(d, default=float) + "\n")
        f_detail.flush()

        print(f"step {step:3d} {phase}{'(s'+str(stage)+')' if not in_warmup else ''} "
              f"valid {sum(valid):2d}/{len(valid)} "
              f"reward {row['reward_mean']:+.3f} "
              f"dock {row['docking_mean']:+.2f} "
              f"kl {row['kl_to_prior']:.4f} "
              f"band {row['percent_in_band']:.0f}% "
              f"replay {row['replay_fraction']:.2f} "
              f"{row['wall_seconds']:.0f}s", flush=True)

        # Rolling checkpoint (alternate A/B)
        cp = ckpt_a if step % 2 == 0 else ckpt_b
        save_checkpoint(cp, step, agent, optimizer, None, baseline, stage, arm_state, replay_buffer)

        # Milestone checkpoints
        if step + 1 == warmup_steps:
            torch.save(agent.state_dict(), os.path.join(out_dir, f"ckpt_warmup_end.pt"))
            print(f"  saved warmup-end checkpoint", flush=True)
        if not in_warmup and arm in ("curriculum_weights", "curriculum_pairs"):
            post = step - warmup_steps
            if stage_len > 0 and post > 0 and post % stage_len == 0:
                torch.save(agent.state_dict(), os.path.join(out_dir, f"ckpt_stage{stage}_step{step}.pt"))
        if step + 1 == total_steps:
            torch.save(agent.state_dict(), os.path.join(out_dir, f"ckpt_final_step{step}.pt"))

        # Adaptive: check end condition
        if arm == "adaptive" and not in_warmup:
            all_active = len(arm_state["active_objs"]) >= len(ADAPTIVE_ORDER)
            last_intro_step = arm_state["introductions"][-1]["step"] if arm_state["introductions"] else warmup_steps
            if all_active and (step - last_intro_step) >= 100:
                post = step - warmup_steps
                if post >= 450:
                    print(f"  step {step}: ADAPTIVE complete (all objectives active, last introduced ≥100 steps ago)", flush=True)
                    torch.save(agent.state_dict(), os.path.join(out_dir, f"ckpt_final_step{step}.pt"))
                    break
            post = step - warmup_steps
            if post >= 700:
                print(f"  step {step}: ADAPTIVE forced end at max steps", flush=True)
                torch.save(agent.state_dict(), os.path.join(out_dir, f"ckpt_final_step{step}.pt"))
                break

    f_csv.close()
    f_detail.close()
    if arm_state["introductions"]:
        json.dump(arm_state["introductions"], open(os.path.join(out_dir, "adaptive_introductions.json"), "w"), indent=1)
    return True


def main():
    ap = argparse.ArgumentParser(description="LigandGPT RL fine-tuning")
    ap.add_argument("--ckpt_load_path", default="final.pt")
    ap.add_argument("--vocab_path", default="./ProtLigVoc.txt")
    ap.add_argument("--data_path", default="./data")
    ap.add_argument("--out", default="runs/rl")
    ap.add_argument("--targets", type=int, nargs="+", default=[0])
    ap.add_argument("--arm", default="no_curriculum", choices=ARM_NAMES)
    ap.add_argument("--n_steps", type=int, default=525)
    ap.add_argument("--warmup_steps", type=int, default=75)
    ap.add_argument("--stage_length", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--replay", type=int, default=8)
    ap.add_argument("--kl_coef", type=float, default=1.0)
    ap.add_argument("--ent_coef", type=float, default=0.005)
    ap.add_argument("--log_ratio_clamp", type=float, default=0.7)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe", action="store_true")
    # Adaptive params
    ap.add_argument("--adaptive_min_phase", type=int, default=40)
    ap.add_argument("--adaptive_forced_intro", type=int, default=100)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    reserve_gpu()

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

    for t in args.targets:
        run_target(t, args, prior, voc, tk, txn, keys, split, oracles, gnn, tmp, state_dict)


if __name__ == "__main__":
    main()
