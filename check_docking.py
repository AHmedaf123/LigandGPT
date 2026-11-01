"""Docking check for targets 0 and 3 (QuickVina 2 + Meeko receptor/ligand PDBQT + the RL scoring function).

1. Redocking: the crystal ligand of each target is prepared with Meeko, docked with the RL settings (20 A box centred on the
   ligand centre of mass from the lmdb, exhaustiveness 4) and its heavy-atom RMSD to the crystal pose is measured (atom
   correspondence from Meeko's INDEX MAP, no symmetry correction, so RMSD is if anything pessimistic). Three QVina seeds.
2. RL path: scoring_function.perform_qvina_docking (the call the RL loop makes, unseeded) on valid molecules from
   results/generation_diagnosis/samples.csv, repeated 3 times to show the run-to-run noise of the docking score.

Writes results/docking_check/docking_check.csv.   Run: .venv/bin/python check_docking.py
"""
import csv
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import lmdb  # noqa: E402
import torch  # noqa: E402
from meeko import MoleculePreparation, PDBQTWriterLegacy  # noqa: E402
from rdkit import Chem  # noqa: E402

import scoring_function as sf  # noqa: E402
from data_utils import receptor_paths  # noqa: E402

DATA = os.path.join(REPO, "data")
TARGETS = (0, 3)


def parse_poses(path):
    """[(score, {pdbqt_serial: xyz heavy atoms})] for every MODEL in a QVina output file."""
    poses, score, atoms = [], None, {}
    for line in open(path):
        if line.startswith("REMARK VINA RESULT"):
            score = float(line.split()[3])
        elif line.startswith("ATOM") and line.split()[-1] not in ("HD", "H"):
            atoms[int(line[6:11])] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ENDMDL"):
            poses.append((score, atoms))
            score, atoms = None, {}
    return poses


def index_map(pdbqt_text):
    toks = " ".join(l[len("REMARK INDEX MAP"):] for l in pdbqt_text.splitlines() if l.startswith("REMARK INDEX MAP")).split()
    return {int(toks[i + 1]): int(toks[i]) - 1 for i in range(0, len(toks), 2)}     # pdbqt serial -> rdkit atom index


def rmsd(pose_atoms, mapping, crystal_xyz):
    d = [np.sum((xyz - crystal_xyz[mapping[s]]) ** 2) for s, xyz in pose_atoms.items() if s in mapping and mapping[s] < len(crystal_xyz)]
    return float(np.sqrt(np.mean(d)))


def load_targets():
    split = torch.load(os.path.join(DATA, "crossdocked_pocket10_pose_split.pt"), weights_only=False)["test"]
    env = lmdb.open(os.path.join(DATA, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"), subdir=False, readonly=True,
                    lock=False, readahead=False, meminit=False, max_readers=256)
    out = {}
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        for t in TARGETS:
            out[t] = pickle.loads(txn.get(keys[split[t]]))
    return out


def main():
    os.makedirs("results/docking_check", exist_ok=True)
    rows = []
    dps = load_targets()
    gen = list(csv.DictReader(open("results/generation_diagnosis/samples.csv")))
    with tempfile.TemporaryDirectory() as tmp:
        for t, dp in dps.items():
            _, rec = receptor_paths(dp["protein_filename"], DATA)
            center = [float(x) for x in dp["ligand_center_of_mass"]]
            ligand_file = os.path.join(DATA, "test_set", dp["ligand_filename"])
            crystal = Chem.SDMolSupplier(ligand_file, removeHs=False)[0]
            n_heavy = crystal.GetNumHeavyAtoms()
            crystal_xyz = crystal.GetConformer().GetPositions()
            mol_h = Chem.AddHs(crystal, addCoords=True)
            setup = MoleculePreparation(rigid_macrocycles=True).prepare(mol_h)[0]
            text, ok, err = PDBQTWriterLegacy.write_string(setup, add_index_map=True)
            assert ok, err
            mapping = index_map(text)
            lig = os.path.join(tmp, f"crystal_{t}.pdbqt")
            open(lig, "w").write(text)
            # ---- 1. redocking of the crystal ligand
            for seed in (1, 2, 3):
                out = os.path.join(tmp, f"crystal_{t}_{seed}_out.pdbqt")
                cmd = sf.build_qvina_command(rec, lig, out, center, [20, 20, 20]) + ["--seed", str(seed)]
                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                dt = time.time() - t0
                poses = parse_poses(out) if r.returncode == 0 else []
                best = poses[0] if poses else (None, {})
                rm = [rmsd(a, mapping, crystal_xyz) for _, a in poses]
                rows.append(dict(target=t, kind="redock_crystal", label=os.path.basename(ligand_file), heavy_atoms=n_heavy, rep=seed,
                                 score=best[0], rmsd_top_pose=(rm[0] if rm else None), rmsd_best_of_modes=(min(rm) if rm else None), seconds=round(dt, 1)))
                print(rows[-1], flush=True)
            # ---- 2. RL path on generated molecules, 3 repeats each
            smis = [r["smiles"] for r in gen if r["setting"] == "struct_valence_k10" and r["target"] == str(t) and r["strict_valid"] == "1"][:6]
            for smi in smis:
                for rep in range(3):
                    t0 = time.time()
                    sc = sf.perform_qvina_docking(smi, rec, center, tmp)
                    rows.append(dict(target=t, kind="rl_path_generated", label=smi, heavy_atoms=Chem.MolFromSmiles(smi).GetNumHeavyAtoms(), rep=rep,
                                     score=sc, rmsd_top_pose=None, rmsd_best_of_modes=None, seconds=round(time.time() - t0, 1)))
                print(f"  t{t} {smi[:48]:48s} scores {[r['score'] for r in rows if r['label'] == smi]}", flush=True)
    with open("results/docking_check/docking_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
