"""Comparison table of the 100-step probes (target 0, seed 0) in windows of 20 steps, read from their metrics.csv.
Rows = metric x arm, columns = windows. Means of per-molecule quantities (docking, ligand efficiency, SA, energy, heavy atoms) are weighted by the number
of molecules in each step, so a window mean is a per-molecule mean. SE = standard error of the step means (docking and heavy atoms only).
Run: .venv/bin/python compare_probes.py       Writes results/probe_comparison/comparison.md and comparison.json"""
import csv
import json
import os

import numpy as np

ARMS = [("probe 1 original (curriculum, old code, no size columns)", "runs/probe/target0/metrics.csv"),
        ("curriculum (re-run, new columns)", "runs/probe_curriculum/target0/metrics.csv"),
        ("fixed, time-average of the repo stage vectors", "runs/probe_fixed_avg/target0/metrics.csv"),
        ("fixed, literal weights from the task", "runs/probe_fixed_literal/target0/metrics.csv")]
WINDOWS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
BATCH = 32
# metric -> (column, weight column or None, scale)
METRICS = {"validity": ("n_valid", None, 1 / BATCH), "docking mean (kcal/mol)": ("docking_mean", "n_gate_pass", 1), "heavy atoms, valid molecules": ("heavy_atoms_mean", "n_valid", 1),
           "heavy atoms, docked molecules": ("heavy_atoms_docked_mean", "n_gate_pass", 1), "ligand efficiency (kcal/mol/heavy atom)": ("ligand_efficiency_mean", "n_gate_pass", 1),
           "SA (docked)": ("sa_mean", "n_gate_pass", 1), "UFF energy (docked)": ("energy_mean", "n_gate_pass", 1), "gate-passing molecules per step": ("n_gate_pass", None, 1),
           "kl_to_prior": ("kl_to_prior", None, 1)}
SE_ROWS = ("docking mean (kcal/mol)", "heavy atoms, valid molecules", "heavy atoms, docked molecules")


def load(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows]) for k in rows[0]}


def wmean(x, w):
    ok = np.isfinite(x) & (w > 0) if w is not None else np.isfinite(x)
    return float(np.average(x[ok], weights=None if w is None else w[ok])) if ok.any() else float("nan")


def cell(d, name, lo, hi):
    col, wcol, scale = METRICS[name]
    if col not in d:
        return None, None
    x, w = d[col][lo:hi], None if wcol is None else d[wcol][lo:hi]
    se = float(np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum())) if name in SE_ROWS and np.isfinite(x).sum() > 1 else None
    return wmean(x, w) * scale, se


def main():
    os.makedirs("results/probe_comparison", exist_ok=True)
    data = {a: load(p) for a, p in ARMS}
    out, lines = {}, []
    head = ["metric", "arm"] + [f"{lo}-{hi - 1}" for lo, hi in WINDOWS] + ["last - first"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in METRICS:
        for arm, d in data.items():
            if d is None or METRICS[name][0] not in d:
                continue
            cells = [cell(d, name, lo, hi) for lo, hi in WINDOWS if hi <= len(d["step"])]
            if len(cells) < len(WINDOWS):
                continue
            vals = [c[0] for c in cells]
            out.setdefault(name, {})[arm] = dict(windows=vals, se=[c[1] for c in cells], last_minus_first=vals[-1] - vals[0])
            fmt = lambda c: f"{c[0]:.3f}" + (f" ± {c[1]:.2f}" if c[1] is not None else "")  # noqa: E731
            lines.append("| " + " | ".join([name, arm] + [fmt(c) for c in cells] + [f"{vals[-1] - vals[0]:+.3f}"]) + " |")
    lines += ["", "docking mean over the stage-1 steps (steps 0-32, the same steps in both arms):", ""]
    out["docking_stage1_steps_0_32"] = {}
    for arm, d in data.items():
        if d is not None and len(d["step"]) >= 33:
            m, se = wmean(d["docking_mean"][:33], d["n_gate_pass"][:33]), float(np.nanstd(d["docking_mean"][:33], ddof=1) / np.sqrt(33))
            out["docking_stage1_steps_0_32"][arm] = dict(mean=m, se=se)
            lines.append(f"- {arm}: {m:.3f} ± {se:.2f}")
    open("results/probe_comparison/comparison.md", "w").write("\n".join(lines) + "\n")
    json.dump(out, open("results/probe_comparison/comparison.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
