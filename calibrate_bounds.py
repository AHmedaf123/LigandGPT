"""Calibrate reward bounds from frozen prior samples.

Samples at least 128 molecules per target with the structured decoder, scores them,
and writes runs/baseline/calibrated_bounds_target<N>.json with P5-P95 bounds for each objective.
Also logs the docking-vs-heavy-atoms linear fit as a diagnostic.

Reuses existing baseline samples (runs/baseline/samples.json) for targets that match.

Run: .venv/bin/python calibrate_bounds.py --targets 0 1 2 3
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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
N_SAMPLES = 128
BATCH = 16
SEED_BASE = 3000
DATA = "data"


def sample_prior(model, voc, tk, target, txn, keys, split):
    dp = pickle.loads(txn.get(keys[split[target]]))
    toks, nums = tk.tokenize(protein=dp["protein_atom_name"],
                             prot_coords=dp["protein_pos"].numpy() - dp["ligand_center_of_mass"].numpy())
    toks.append("<LIG_START>")
    nums.append(1)
    ids = torch.tensor(voc.encode(toks), dtype=torch.long).unsqueeze(0).cuda()
    pn = torch.tensor(nums, dtype=torch.float).unsqueeze(0).cuda()
    budget = min(200, 2048 - ids.shape[1])
    end_id = voc[dec.END]
    samples = []
    for b in range(N_SAMPLES // BATCH):
        torch.manual_seed(SEED_BASE + b)
        rows, done, _ = dec.sample_smiles(model, voc, ids, pn, BATCH, structured=True, valence=True,
                                          top_k=10, temperature=1.0, max_new_tokens=budget, return_masks=True)
        for i, r in enumerate(rows):
            term = end_id in r
            g = r[:r.index(end_id)] if term else r
            smi = "".join(voc.decode(g))
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi) if smi else None
            ok = mol is not None and mol.GetNumAtoms() >= 4 and term
            samples.append(dict(target=target, smiles=smi, valid=ok,
                                n_heavy=mol.GetNumHeavyAtoms() if mol and ok else None))
    return samples, dp


def score_samples(samples, target, dp, oracles, gnn, tmp):
    valid = [s for s in samples if s["valid"]]
    rec_pdb, rec = receptor_paths(dp["protein_filename"], DATA)
    center = [float(x) for x in dp["ligand_center_of_mass"]]
    # cheap scores
    for s in valid:
        md = sf.prepare_molecule_with_conformer(s["smiles"])
        p = sf.evaluate_property_scores(md, oracles["qed"], oracles["sa"]) if md is not None else None
        s.update(qed=p and p.qed, sa=p and p.synthetic_accessibility,
                 steric=p and p.steric_clash, torsion=p and p.torsion_strain,
                 energy=p and p.internal_energy)
    # bioactivity
    bio = sf.predict_bioactivity_for_batch([s["smiles"] for s in valid], gnn)
    for s, b in zip(valid, bio):
        s["bioactivity"] = b
    # docking
    workers = sf.cpu_quota()
    def dock_one(s):
        try:
            return sf.perform_qvina_docking(s["smiles"], rec, center, tmp)
        except Exception:
            return None
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        docks = list(pool.map(dock_one, valid))
    for s, d in zip(valid, docks):
        s["docking"] = d
    print(f"  target {target}: docked {len(valid)} in {time.time()-t0:.0f}s", flush=True)
    return valid


def compute_bounds(scored, target):
    bounds = {}
    for obj, info in OBJ_INFO.items():
        vals = [s[obj] for s in scored if s.get(obj) is not None]
        if len(vals) < 5:
            print(f"  WARNING: target {target}, {obj}: only {len(vals)} values", flush=True)
            if obj == "docking":
                bounds[obj] = {"lo": -10.0, "hi": -3.0, "minimize": info["minimize"]}
            elif obj == "bioactivity":
                bounds[obj] = {"lo": -12.0, "hi": -4.0, "minimize": info["minimize"]}
            else:
                bounds[obj] = {"lo": 0.0, "hi": 1.0, "minimize": info["minimize"]}
            continue
        p5, p95 = np.percentile(vals, [5, 95])
        bounds[obj] = {"lo": float(p5), "hi": float(p95), "minimize": info["minimize"]}
        print(f"  {obj}: P5={p5:.3f} P95={p95:.3f} (n={len(vals)})", flush=True)

    # docking vs heavy atoms linear fit
    pairs = [(s["n_heavy"], s["docking"]) for s in scored if s.get("docking") is not None and s.get("n_heavy") is not None]
    fit = None
    if len(pairs) >= 10:
        x, y = zip(*pairs)
        slope, intercept, r, p, se = sp_stats.linregress(x, y)
        fit = dict(intercept=float(intercept), slope=float(slope), r=float(r), n=len(pairs))
        print(f"  docking vs heavy atoms: intercept={intercept:.3f}, slope={slope:.4f}, r={r:.3f}", flush=True)

    return bounds, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()

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
    os.makedirs("runs/baseline", exist_ok=True)

    # Try to reuse existing samples
    existing_samples = None
    samples_path = "runs/baseline/samples.json"
    if os.path.exists(samples_path):
        raw = json.load(open(samples_path))
        if isinstance(raw, list) and len(raw) == 2:
            existing_samples = raw[0]

    for t in args.targets:
        print(f"\n=== Target {t} ===", flush=True)
        # Check if we can reuse existing scored samples
        reuse = None
        if existing_samples:
            t_samples = [s for s in existing_samples if s["target"] == t and s["valid"] and s.get("docking") is not None]
            if len(t_samples) >= 20:
                print(f"  reusing {len(t_samples)} existing scored samples", flush=True)
                # add n_heavy
                from rdkit import Chem
                for s in t_samples:
                    if "n_heavy" not in s:
                        mol = Chem.MolFromSmiles(s["smiles"])
                        s["n_heavy"] = mol.GetNumHeavyAtoms() if mol else None
                reuse = t_samples

        if reuse is None:
            dp_data = pickle.loads(txn.get(keys[split[t]]))
            samples, dp = sample_prior(model, voc, tk, t, txn, keys, split)
            scored = score_samples(samples, t, dp_data, oracles, gnn, tmp)
        else:
            scored = reuse

        bounds, fit = compute_bounds(scored, t)
        out = dict(target=t, n_scored=len(scored), bounds=bounds, docking_vs_heavy_atoms_fit=fit)
        out_path = f"runs/baseline/calibrated_bounds_target{t}.json"
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"  wrote {out_path}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
