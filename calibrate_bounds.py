"""Calibrate reward bounds from frozen prior samples.

Samples until at least MIN_DOCKED molecules are docked per target, scores them,
and writes runs/baseline_v2/calibrated_bounds_target<N>.json with P5-P95 bounds.
Also saves 200 scored molecules per target as prior reference for eval_final.py.

Run: .venv/bin/python calibrate_bounds.py --targets 0 1 2 3
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from scipy import stats as sp_stats

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import decoding as dec
import scoring_function as sf
from data_utils import receptor_paths
from model import LigandGPT
from tokenizer import ProtLigTokenizer
from vocabulary import read_vocabulary

OBJ_INFO = {
    "docking":     {"minimize": True},
    "bioactivity": {"minimize": True},
    "qed":         {"minimize": False},
    "sa":          {"minimize": True},
    "energy":      {"minimize": True},
    "torsion":     {"minimize": True},
}
MIN_DOCKED = 300
BATCH = 16
SEED_BASE = 7000
DATA = "data"
OUT_DIR = "runs/baseline_v2"


def sample_and_score_until(model, voc, tk, target, txn, keys, split, oracles, gnn, tmp, min_docked=300):
    dp = pickle.loads(txn.get(keys[split[target]]))
    toks, nums = tk.tokenize(protein=dp["protein_atom_name"],
                             prot_coords=dp["protein_pos"].numpy() - dp["ligand_center_of_mass"].numpy())
    toks.append("<LIG_START>")
    nums.append(1)
    ids = torch.tensor(voc.encode(toks), dtype=torch.long).unsqueeze(0).cuda()
    pn = torch.tensor(nums, dtype=torch.float).unsqueeze(0).cuda()
    budget = min(200, 2048 - ids.shape[1])
    end_id = voc[dec.END]
    rec_pdb, rec = receptor_paths(dp["protein_filename"], DATA)
    center = [float(x) for x in dp["ligand_center_of_mass"]]

    all_samples = []
    n_sampled = 0
    n_valid = 0
    n_docked = 0
    fail_counts = {"embed": 0, "meeko": 0, "nopose": 0, "timeout": 0, "no_conformer": 0}
    batch_idx = 0

    while n_docked < min_docked:
        torch.manual_seed(SEED_BASE + batch_idx)
        rows, done, _ = dec.sample_smiles(model, voc, ids, pn, BATCH, structured=True, valence=True,
                                          top_k=10, temperature=1.0, max_new_tokens=budget, return_masks=True)
        batch_valid = []
        for i, r in enumerate(rows):
            n_sampled += 1
            term = end_id in r
            g = r[:r.index(end_id)] if term else r
            smi = "".join(voc.decode(g))
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi) if smi else None
            ok = mol is not None and mol.GetNumAtoms() >= 4 and term
            entry = dict(target=target, smiles=smi, valid=ok,
                         n_heavy=mol.GetNumHeavyAtoms() if mol and ok else None)
            all_samples.append(entry)
            if ok:
                n_valid += 1
                batch_valid.append(entry)

        # Cheap scores
        for s in batch_valid:
            md = sf.prepare_molecule_with_conformer(s["smiles"])
            p = sf.evaluate_property_scores(md, oracles["qed"], oracles["sa"]) if md is not None else None
            s.update(qed=p and p.qed, sa=p and p.synthetic_accessibility,
                     steric=p and p.steric_clash, torsion=p and p.torsion_strain,
                     energy=p and p.internal_energy)
            if md is None or not (hasattr(md, 'has_conformer') and md.has_conformer):
                fail_counts["no_conformer"] += 1

        # Bioactivity
        bio = sf.predict_bioactivity_for_batch([s["smiles"] for s in batch_valid], gnn)
        for s, b in zip(batch_valid, bio):
            s["bioactivity"] = b

        # Docking (subprocess-based with timeout)
        to_dock = [s["smiles"] for s in batch_valid if s.get("smiles")]
        if to_dock:
            docked = sf.dock_batch_with_timeout(to_dock, rec, center, tmp,
                                                per_mol_timeout=180, n_workers=sf.cpu_quota())
            for s in batch_valid:
                score = docked.get(s["smiles"])
                s["docking"] = score
                if score is not None:
                    n_docked += 1

        batch_idx += 1
        print(f"  target {target}: batch {batch_idx}, sampled={n_sampled}, valid={n_valid}, docked={n_docked}/{min_docked}",
              flush=True)

    scored = [s for s in all_samples if s["valid"] and s.get("docking") is not None]
    stats = dict(n_sampled=n_sampled, n_valid=n_valid, n_docked=n_docked, fail_counts=fail_counts)
    return scored, all_samples, dp, stats


def compute_bounds(scored, target):
    bounds = {}
    for obj, info in OBJ_INFO.items():
        vals = [s[obj] for s in scored if s.get(obj) is not None]
        if len(vals) < 10:
            print(f"  WARNING: target {target}, {obj}: only {len(vals)} values", flush=True)
            continue
        p5, p95 = np.percentile(vals, [5, 95])
        mean, sd = np.mean(vals), np.std(vals)
        bounds[obj] = {"lo": float(p5), "hi": float(p95), "minimize": info["minimize"],
                       "mean": float(mean), "sd": float(sd), "n": len(vals)}
        print(f"  {obj}: P5={p5:.3f} P95={p95:.3f} mean={mean:.3f} SD={sd:.3f} (n={len(vals)})", flush=True)

    # docking vs heavy atoms linear fit
    pairs = [(s["n_heavy"], s["docking"]) for s in scored
             if s.get("docking") is not None and s.get("n_heavy") is not None]
    fit = None
    if len(pairs) >= 10:
        x, y = zip(*pairs)
        slope, intercept, r, p, se = sp_stats.linregress(x, y)
        fit = dict(intercept=float(intercept), slope=float(slope), r=float(r), n=len(pairs))
        print(f"  docking vs heavy atoms: intercept={intercept:.3f}, slope={slope:.4f}, r={r:.3f} (n={len(pairs)})", flush=True)

    return bounds, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--min_docked", type=int, default=MIN_DOCKED)
    args = ap.parse_args()

    min_docked_target = args.min_docked

    voc, tk = read_vocabulary("./ProtLigVoc.txt"), ProtLigTokenizer()
    model = LigandGPT(vocab_size=len(voc), d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, context_length=2048).cuda()
    model.load_state_dict(torch.load("final.pt"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    import lmdb as lmdb_
    env = lmdb_.open(os.path.join(DATA, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"),
                     subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    split = torch.load(os.path.join(DATA, "crossdocked_pocket10_pose_split.pt"), weights_only=False)["test"]

    oracles, gnn, tmp = sf.initialize_scoring_system()
    os.makedirs(OUT_DIR, exist_ok=True)

    for t in args.targets:
        print(f"\n=== Target {t} ===", flush=True)
        t0 = time.time()
        scored, all_samples, dp, stats = sample_and_score_until(
            model, voc, tk, t, txn, keys, split, oracles, gnn, tmp,
            min_docked=min_docked_target)

        bounds, fit = compute_bounds(scored, t)
        out = dict(target=t, n_scored=len(scored), stats=stats, bounds=bounds,
                   docking_vs_heavy_atoms_fit=fit)
        out_path = os.path.join(OUT_DIR, f"calibrated_bounds_target{t}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)

        # Save 200 scored molecules as prior reference (fixed seed selection)
        rng = np.random.RandomState(42)
        ref_200 = rng.choice(scored, size=min(200, len(scored)), replace=False).tolist()
        ref_path = os.path.join(OUT_DIR, f"prior_reference_target{t}.json")
        with open(ref_path, "w") as f:
            json.dump(ref_200, f, indent=1)

        elapsed = time.time() - t0
        print(f"  target {t}: {len(scored)} scored, {elapsed:.0f}s, wrote {out_path}", flush=True)
        print(f"  stats: {stats}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
