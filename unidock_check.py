"""Uni-Dock 1.1.0 (prebuilt CUDA 12.0 binary) vs QuickVina2 on the 20 docking-noise molecules of the baseline (10 per target, targets 0 and 3).
Nothing is adopted: this only measures speed and agreement. Ligands are prepared exactly as in baseline.py (RDKit embed seed 42 -> SDF -> Meeko
PDBQT), the receptor is the same Meeko PDBQT, the box is the RL box (20 A at the ligand centre). QVina is re-docked (exhaustiveness 4, seed 2,
one job per cgroup CPU) for a matched timing and as the noise ceiling of the correlation (seed 1 = the stored baseline scores).
Run: .venv/bin/python unidock_check.py      Writes results/unidock/unidock_check.json"""
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import baseline as bl                      # dock(), TARGETS, TMAP, DATA (also puts the repo on sys.path and chdirs to it)
import scoring_function as sf
from data_utils import receptor_paths

UD_DIR = "docking/unidock-1.1.0"
UD = os.path.join(UD_DIR, "unidock-1.1.0-cuda120-linux-x86_64")
MODES = ("fast", "balance", "detail")


def proc_gpu_mib(pid):
    """GPU memory of one process from nvidia-smi (the card is shared, so total memory.used is dominated by other tenants); None if not listed."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = [l.replace(" ", "").split(",") for l in out.splitlines() if l.strip()]
    return max([int(m) for p, m in rows if p == str(pid)], default=None)


def run_unidock(rec, center, ligs, mode, outdir):
    """-> (scores in ligs order, wall seconds of the successful attempt, peak GPU MiB of the process, stdout tail, returncode, attempts).
    One GPU batch for all ligands of the target. The GPU is shared and other tenants sometimes fill it completely (the binary then dies at
    start-up or at 'Computing Vina grid' with a CUDA error), so a CUDA failure is retried every 3 s (max 45 min)."""
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.abspath(os.path.join(UD_DIR, "lib")))      # libgomp.so.1 only (symlink to torch's copy)
    cmd = [UD, "--receptor", rec, "--gpu_batch", *ligs, "--search_mode", mode, "--scoring", "vina", "--num_modes", "1", "--seed", "1",
           "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]), "--size_x", "20", "--size_y", "20", "--size_z", "20",
           "--dir", outdir, "--max_gpu_memory", "2048"]
    for attempt in range(1, 901):
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        peak = None
        while p.poll() is None:
            m = proc_gpu_mib(p.pid)
            peak = m if m is None or peak is None else max(peak, m)
            time.sleep(0.2)
        wall, text = time.time() - t0, p.stdout.read()
        if p.returncode == 0 or not ("CUDA" in text or "Computing Vina grid" not in text) or attempt == 900:
            break
        time.sleep(3)
    scores = []
    for l in ligs:
        out = os.path.join(outdir, os.path.basename(l).replace(".pdbqt", "_out.pdbqt"))
        scores.append(sf.parse_docking_score_from_file(out) if os.path.exists(out) else None)
    return scores, wall, peak, text[-300:], p.returncode, attempt


def corr(a, b):
    ok = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    x, y = np.array([o[0] for o in ok]), np.array([o[1] for o in ok])
    return dict(n=len(ok), pearson=float(pearsonr(x, y)[0]) if len(ok) > 2 else None, spearman=float(spearmanr(x, y)[0]) if len(ok) > 2 else None,
                mean_diff_b_minus_a=float(np.mean(y - x)), rmse=float(np.sqrt(np.mean((y - x) ** 2))))


def main():
    os.makedirs("results/unidock", exist_ok=True)
    os.makedirs(os.path.join(UD_DIR, "lib"), exist_ok=True)
    link = os.path.join(UD_DIR, "lib", "libgomp.so.1")
    if not os.path.lexists(link):
        os.symlink(os.path.join(os.path.dirname(torch.__file__), "lib", "libgomp.so.1"), link)
    base = json.load(open("runs/baseline/baseline.json"))["molecules"]
    pick = [s for t in bl.TARGETS for s in [x for x in base if x["target"] == t and x["docking"] is not None][:10]]    # the baseline's noise set
    assert len(pick) == 20
    qv1 = [s["docking"] for s in pick]                                       # stored baseline scores (QVina, seed 1)

    # ---- QuickVina re-dock, seed 2: matched timing + noise ceiling
    t0 = time.time()
    with ThreadPoolExecutor(sf.cpu_quota()) as ex:
        res = list(ex.map(lambda s: bl.dock(s["smiles"], s["target"], seed=2), pick))
    qvina_wall = time.time() - t0
    qv2 = [r[0] for r in res]

    out = dict(binary=UD, molecules=[dict(target=s["target"], smiles=s["smiles"], qvina_seed1=s["docking"], qvina_seed2=q) for s, q in zip(pick, qv2)],
               qvina=dict(workers=sf.cpu_quota(), wall_s_20_molecules=qvina_wall, mean_job_s=float(np.mean([r[1] for r in res])),
                          stored_baseline_sequential_uncontended_mean_s=json.load(open("runs/baseline/baseline.json"))["docking_time"]["sequential_uncontended_mean_s"]),
               seed1_vs_seed2=corr(qv1, qv2), modes={})
    with tempfile.TemporaryDirectory() as td:
        ligs, t0 = {}, time.time()                                           # ligand preparation (same code path as the QVina jobs)
        for i, s in enumerate(pick):
            mol = sf.prepare_ligand_molecule(s["smiles"])
            sdf, pq = f"{td}/m{i}.sdf", f"{td}/m{i}.pdbqt"
            ok = mol is not None and sf.write_molecule_to_sdf(mol, sdf) and sf.convert_sdf_to_pdbqt(sdf, pq)
            ligs[i] = pq if ok else None
        out["ligand_prep_s_20_molecules"] = time.time() - t0
        for mode in MODES:
            uni, per_target = [None] * 20, {}
            wall, peak = 0.0, 0
            for t in bl.TARGETS:
                idx = [i for i, s in enumerate(pick) if s["target"] == t and ligs[i]]
                m = bl.TMAP[str(t)]
                rec = receptor_paths(m["protein_filename"], bl.DATA)[1]
                od = f"{td}/{mode}_t{t}"
                os.makedirs(od)
                sc, w, pk, tail, rc, tries = run_unidock(rec, m["center"], [ligs[i] for i in idx], mode, od)
                for i, v in zip(idx, sc):
                    uni[i] = v
                wall, peak = wall + w, max(peak, pk or 0)
                per_target[t] = dict(wall_s=w, peak_gpu_mib=pk, attempts=tries, returncode=rc, none_scores=sum(v is None for v in sc), tail=tail if rc or None in sc else "")
                print(f"{mode} target {t}: {w:.1f}s, rc {rc}, attempts {tries}, process GPU memory {pk} MiB, scores {[None if v is None else round(v, 2) for v in sc]}", flush=True)
            out["modes"][mode] = dict(wall_s_20_molecules=wall, peak_process_gpu_mib=peak, per_target=per_target, scores=uni,
                                      vs_qvina_seed1=corr(qv1, uni), vs_qvina_seed2=corr(qv2, uni),
                                      vs_qvina_seed1_target0=corr(qv1[:10], uni[:10]), vs_qvina_seed1_target3=corr(qv1[10:], uni[10:]))
    json.dump(out, open("results/unidock/unidock_check.json", "w"), indent=1, default=float)
    q = out["qvina"]
    print(f"\nQVina 4 workers: {q['wall_s_20_molecules']:.0f}s wall for 20 | ligand prep {out['ligand_prep_s_20_molecules']:.0f}s | QVina seed1 vs seed2: {out['seed1_vs_seed2']}")
    for mode, m in out["modes"].items():
        print(f"Uni-Dock {mode}: {m['wall_s_20_molecules']:.1f}s for 20 | process GPU memory {m['peak_process_gpu_mib']} MiB | vs QVina seed1 {m['vs_qvina_seed1']}")


if __name__ == "__main__":
    main()
