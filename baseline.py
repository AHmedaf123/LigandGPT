"""Baseline from the prior: structured decoder (syntax + valence, top-k 10, T=1), targets 0 and 3, 128 samples each.
Scores valid molecules with the real scorers (QuickVina via Meeko, GNN, QED, SA, steric, torsion, UFF energy), applies the
stage-1 gates (torsion 5, energy 300, SA reject > 6.0), measures docking time and docking noise (3 seeds on 20 molecules).
Run: .venv/bin/python baseline.py      Writes runs/baseline/baseline.json"""
import json
import math
import os
import subprocess  # noqa: F401
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import diagnose_generation as dg          # classify(), load_prompt(), generate(); also puts the repo on sys.path and chdirs to it
import scoring_function as sf
from data_utils import receptor_paths
from tokenizer import ProtLigTokenizer
from vocabulary import read_vocabulary
from model import LigandGPT
import torch

N, BATCH, SEED, WORKERS, TARGETS = 128, 16, 2000, sf.cpu_quota(), (0, 3)
PLAN_WORKERS = 60                                         # worker count the RL step is projected for
SAMPLES = "runs/baseline/samples.json"
DATA = "data"
TMAP = json.load(open(f"{DATA}/target_map.json"))
GATES = dict(torsion=5.0, energy=300.0, sa=6.0)           # stage-1 gates: reject torsion > 5, energy > 300, sa > 6.0
OBJ = ["docking", "bioactivity", "qed", "sa", "steric", "torsion", "energy"]


def dock(smiles, target, seed=1, exhaustiveness=4):
    """Same steps as scoring_function.perform_qvina_docking (which now passes --cpu 1), plus --seed. -> (score, seconds)"""
    t0 = time.time()
    m = TMAP[str(target)]
    rec = receptor_paths(m["protein_filename"], DATA)[1]
    with tempfile.TemporaryDirectory() as td:
        mol = sf.prepare_ligand_molecule(smiles)
        sdf, lig, out = f"{td}/l.sdf", f"{td}/l.pdbqt", f"{td}/o.pdbqt"
        if mol is None or not sf.write_molecule_to_sdf(mol, sdf) or not sf.convert_sdf_to_pdbqt(sdf, lig):
            return None, time.time() - t0
        cmd = sf.build_qvina_command(rec, lig, out, m["center"], [20, 20, 20])
        cmd[cmd.index("--exhaustiveness") + 1] = str(exhaustiveness)
        ok = sf.execute_qvina_docking(cmd + ["--seed", str(seed)])
        return (sf.parse_docking_score_from_file(out) if ok else None), time.time() - t0


def stats(values):
    v = [x for x in values if x is not None]
    return dict(n=len(v), mean=float(np.mean(v)) if v else None, std=float(np.std(v, ddof=1)) if len(v) > 1 else None)


def passes(s):
    return not ((s["energy"] is not None and s["energy"] > GATES["energy"]) or (s["sa"] is not None and s["sa"] > GATES["sa"])
                or (s["torsion"] is not None and s["torsion"] > GATES["torsion"]))


def main():
    os.makedirs("runs/baseline", exist_ok=True)
    voc, tk = read_vocabulary("./ProtLigVoc.txt"), ProtLigTokenizer()
    model = LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()
    model.load_state_dict(torch.load("final.pt"))
    model.eval()
    kw = dict(top_k=10, temperature=1.0, valence=True)

    # ---- 1. sample (cached in runs/baseline/samples.json so a rerun of the scoring does not resample)
    samples, batch_seconds = [], []
    if os.path.exists(SAMPLES):
        samples, batch_seconds = json.load(open(SAMPLES))
    for t in TARGETS if not samples else ():
        ids, nums, _ = dg.load_prompt(t, voc, tk)
        budget = min(200, 2048 - ids.shape[1])
        for b in range(N // BATCH):
            t0 = time.time()
            rows = dg.generate("struct", kw, model, voc, ids, nums, BATCH, budget, SEED + b)
            batch_seconds.append(time.time() - t0)
            for toks, term in rows:
                cat, ok, _ = dg.classify(toks, term)
                samples.append(dict(target=t, smiles="".join(toks), n_tokens=len(toks), category=cat, valid=ok))
        print(f"target {t}: sampled, valid {sum(s['valid'] for s in samples if s['target'] == t)}/{N}", flush=True)
        json.dump([samples, batch_seconds], open(SAMPLES, "w"))
    del model
    torch.cuda.empty_cache()

    # ---- 2. non-docking scores for valid molecules
    oracles, gnn, _ = sf.initialize_scoring_system()
    valid = [s for s in samples if s["valid"]]
    t0 = time.time()
    for s in valid:
        md = sf.prepare_molecule_with_conformer(s["smiles"])
        p = sf.evaluate_property_scores(md, oracles["qed"], oracles["sa"]) if md is not None else None
        s.update(qed=p and p.qed, sa=p and p.synthetic_accessibility, steric=p and p.steric_clash, torsion=p and p.torsion_strain, energy=p and p.internal_energy)
    bio = sf.predict_bioactivity_for_batch([s["smiles"] for s in valid], gnn)
    for s, b in zip(valid, bio):
        s["bioactivity"] = b
    props_seconds = (time.time() - t0) / max(len(valid), 1)

    # ---- 3. docking: every valid molecule, one worker per available CPU, one core each (exhaustiveness 4, seed 1)
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(lambda s: dock(s["smiles"], s["target"]), valid))
    pool_wall = time.time() - t0
    for s, (score, secs) in zip(valid, res):
        s["docking"], s["dock_seconds"] = score, secs
    print(f"docked {len(valid)} molecules in {pool_wall:.0f}s wall with {WORKERS} workers (cgroup CPU quota)", flush=True)
    seq = [dock(s["smiles"], s["target"])[1] for s in valid[:10]]            # uncontended, one at a time

    # ---- 4. docking noise: 20 valid, dockable molecules (10 per target), seeds 1-3
    pick = [s for t in TARGETS for s in [x for x in valid if x["target"] == t and x["docking"] is not None][:10]]

    def noise(exh):
        t0 = time.time()
        with ThreadPoolExecutor(WORKERS) as ex:
            out = list(ex.map(lambda a: dock(a[0]["smiles"], a[0]["target"], seed=a[1], exhaustiveness=exh), [(s, sd) for s in pick for sd in (1, 2, 3)]))
        wall, per = time.time() - t0, [o[1] for o in out]
        sc = np.array([o[0] if o[0] is not None else np.nan for o in out]).reshape(len(pick), 3)
        sd = np.nanstd(sc, axis=1, ddof=1)
        return dict(exhaustiveness=exh, molecules=len(pick), mean_sd_kcal=float(np.nanmean(sd)), max_sd_kcal=float(np.nanmax(sd)), mean_dock_seconds=float(np.mean(per)), wall_seconds=wall)
    noise4 = noise(4)
    noise8 = noise(8) if noise4["mean_sd_kcal"] > 0.5 else None

    # ---- 5. summarise
    def block(rows_all):
        v = [s for s in rows_all if s["valid"]]
        g = [s for s in v if passes(s)]
        return dict(samples=len(rows_all), valid=len(v), validity=len(v) / len(rows_all),
                    mean_tokens_all=float(np.mean([s["n_tokens"] for s in rows_all])), mean_tokens_valid=float(np.mean([s["n_tokens"] for s in v])) if v else None,
                    gate_pass_of_valid=len(g) / max(len(v), 1), gate_pass_of_samples=len(g) / len(rows_all),
                    gate_rejects_of_valid=dict(sa_gt_6=sum(s["sa"] is not None and s["sa"] > GATES["sa"] for s in v), torsion_gt_5=sum(s["torsion"] is not None and s["torsion"] > GATES["torsion"] for s in v),
                                               energy_gt_300=sum(s["energy"] is not None and s["energy"] > GATES["energy"] for s in v), steric_gt_18_not_in_gate=sum(s["steric"] is not None and s["steric"] > 18 for s in v),
                                               missing_conformer=sum(s["sa"] is not None and s["torsion"] is None for s in v)),
                    sa_gate_rejection_rate=sum(s["sa"] is not None and s["sa"] > GATES["sa"] for s in v) / max(len(v), 1),
                    objectives={k: stats([s.get(k) for s in v]) for k in OBJ})
    valid_n = sum(s["valid"] for s in samples)
    t_mean = float(np.mean([s["dock_seconds"] for s in valid]))
    batch_time = float(np.mean(batch_seconds))
    n_valid_124 = round(124 * valid_n / len(samples))
    sampling_s = math.ceil(124 / BATCH) * batch_time
    t_max = float(np.max([s["dock_seconds"] for s in valid]))
    out = dict(config=dict(decoder="structured syntax+valence, top-k 10, T=1.0", samples_per_target=N, batch=BATCH, seed_base=SEED, targets=list(TARGETS), gates=GATES,
                           docking="QuickVina2 exhaustiveness 4, 20 A box at ligand centre, --cpu 1", workers=WORKERS, cgroup_cpu_quota=WORKERS),
               per_target={str(t): block([s for s in samples if s["target"] == t]) for t in TARGETS}, pooled=block(samples),
               docking_time=dict(mean_s=t_mean, median_s=float(np.median([s["dock_seconds"] for s in valid])),
                                 p95_s=float(np.percentile([s["dock_seconds"] for s in valid], 95)), sequential_uncontended_mean_s=float(np.mean(seq)),
                                 pool_wall_s=pool_wall, molecules=len(valid), none_scores=sum(s["docking"] is None for s in valid)),
               non_docking_scoring_seconds_per_molecule=props_seconds, sampling_seconds_per_batch16=batch_time,
               projection_rl_step_batch124=dict(
                   expected_valid_molecules=n_valid_124, sampling_s=sampling_s, non_docking_scoring_s_single_process=n_valid_124 * props_seconds,
                   on_this_machine_4_cpus=dict(docking_s=n_valid_124 * t_mean / WORKERS, total_s=sampling_s + n_valid_124 * props_seconds + n_valid_124 * t_mean / WORKERS),
                   if_60_cpus_were_available=dict(docking_s=math.ceil(n_valid_124 / PLAN_WORKERS) * t_max, total_s=sampling_s + n_valid_124 * props_seconds + math.ceil(n_valid_124 / PLAN_WORKERS) * t_max),
                   note="sampling = ceil(124/16) sequential chunks of 16 (measured); docking uses the measured per-molecule time with --cpu 1; excludes the policy update"),
               docking_noise_exh4=noise4, docking_noise_exh8=noise8, molecules=valid)
    json.dump(out, open("runs/baseline/baseline.json", "w"), indent=1, default=float)
    p = out["pooled"]
    print(f"validity {p['validity']:.3f} | tokens {p['mean_tokens_all']:.0f} | gate pass (of valid) {p['gate_pass_of_valid']:.3f} | SA>6 rejects {p['sa_gate_rejection_rate']:.3f}")
    print("noise exh4:", noise4, "\nnoise exh8:", noise8)


if __name__ == "__main__":
    main()
