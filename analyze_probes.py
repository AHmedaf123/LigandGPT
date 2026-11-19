"""Analyze probe results and generate probe_report.md with tables and matplotlib plots.
Run after all probes complete: .venv/bin/python analyze_probes.py
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROBE_DIR = "runs/probe_new"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_metrics(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            d = {}
            for k, v in row.items():
                try:
                    d[k] = float(v) if "." in v or "e" in v.lower() or v == "nan" else int(v)
                except (ValueError, TypeError):
                    d[k] = v
            rows.append(d)
    return rows


def window_stats(rows, start, end, keys):
    subset = [r for r in rows if start <= r["step"] < end]
    out = {}
    for k in keys:
        vals = [r[k] for r in subset if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def pa_report():
    """P-A: Warmup only, compare steps 0-14 vs 60-74."""
    path = os.path.join(PROBE_DIR, "pa_warmup", "target0", "metrics.csv")
    if not os.path.exists(path):
        return "P-A: metrics not found\n"
    rows = load_metrics(path)
    keys = ["validity", "heavy_atoms_mean", "size_hit_rate", "docking_mean", "qed_mean", "sa_mean", "wall_seconds"]
    early = window_stats(rows, 0, 15, keys)
    late = window_stats(rows, 60, 75, keys)
    lines = ["## Probe P-A: Warmup Only (75 steps, target 0, seed 99)\n"]
    lines.append("| Metric | Steps 0–14 | Steps 60–74 | Direction |")
    lines.append("|--------|-----------|------------|-----------|")
    for k in keys:
        e, l = early[k], late[k]
        if k == "sa_mean":
            direction = "↓ better" if l < e else "↑ worse"
        elif k == "docking_mean":
            direction = "↓ better" if l < e else "↑ worse"
        elif k == "wall_seconds":
            direction = ""
        else:
            direction = "↑ better" if l > e else "↓ worse"
        lines.append(f"| {k} | {e:.3f} | {l:.3f} | {direction} |")
    lines.append(f"\nMean wall seconds per step: {np.mean([r['wall_seconds'] for r in rows]):.1f}s")
    lines.append(f"Total steps: {len(rows)}")
    return "\n".join(lines) + "\n"


def pb_report():
    """P-B: Mechanics test."""
    lines = ["## Probe P-B: Mechanics Test (compressed schedule)\n"]
    for arm in ["no_curriculum", "curriculum_weights", "curriculum_pairs", "adaptive"]:
        path = os.path.join(PROBE_DIR, "pb_mechanics", arm, "target0", "metrics.csv")
        if not os.path.exists(path):
            lines.append(f"### {arm}: not found\n")
            continue
        rows = load_metrics(path)
        config_path = os.path.join(PROBE_DIR, "pb_mechanics", arm, "target0", "config.json")
        config = json.load(open(config_path)) if os.path.exists(config_path) else {}
        lines.append(f"### {arm}")
        lines.append(f"- Steps completed: {len(rows)}")
        lines.append(f"- Stages observed: {sorted(set(r.get('stage', '?') for r in rows))}")
        # check adaptive introductions
        intro_path = os.path.join(PROBE_DIR, "pb_mechanics", arm, "target0", "adaptive_introductions.json")
        if os.path.exists(intro_path):
            intros = json.load(open(intro_path))
            lines.append(f"- Adaptive introductions: {len(intros)}")
            for intro in intros:
                lines.append(f"  - step {intro['step']}: {intro['objective']} (metric={intro['metric_value']:.3f}, forced={intro['forced']})")
        # evidence files
        evidence = []
        tdir = os.path.join(PROBE_DIR, "pb_mechanics", arm, "target0")
        for fn in os.listdir(tdir):
            if fn.endswith(".json") or fn.endswith(".csv") or fn.endswith(".pt"):
                evidence.append(fn)
        lines.append(f"- Evidence files: {', '.join(sorted(evidence))}")
        lines.append("")

    # Resume test
    rpath = os.path.join(PROBE_DIR, "pb_mechanics", "resume_test", "target0", "metrics.csv")
    if os.path.exists(rpath):
        rows = load_metrics(rpath)
        lines.append("### Resume Test")
        lines.append(f"- Steps after resume: {len(rows)}")
        if rows:
            lines.append(f"- Last step: {rows[-1]['step']}")
            lines.append(f"- Resume appears {'SUCCESSFUL' if len(rows) > 1 else 'INCOMPLETE'}")
    else:
        lines.append("### Resume Test: not found")
    return "\n".join(lines) + "\n"


def pc_report():
    """P-C: Quality test."""
    lines = ["## Probe P-C: Quality Test (150 steps from warmup checkpoint)\n"]

    # Reference numbers from earlier (lost) run
    lines.append("Reference (earlier lost no_curriculum seed 0):")
    lines.append("- Warmup end: validity 0.71, heavy 26.7, size_hit 0.64, docking -6.54, QED 0.33, SA 5.93")
    lines.append("- After ~300 steps: validity 0.90, heavy 24.5, size_hit 0.87, docking -7.17, QED 0.56, SA 4.68, KL 8.3")
    lines.append("")

    for name, label in [("pc_no_curriculum", "no_curriculum"), ("pc_curriculum_pairs", "curriculum_pairs")]:
        path = os.path.join(PROBE_DIR, name, "target0", "metrics.csv")
        if not os.path.exists(path):
            lines.append(f"### {label}: not found\n")
            continue
        rows = load_metrics(path)
        keys = ["validity", "heavy_atoms_mean", "size_hit_rate", "docking_mean", "bioactivity_mean",
                "qed_mean", "sa_mean", "kl_to_prior", "replay_fraction"]
        late = window_stats(rows, max(0, len(rows) - 30), len(rows), keys)
        lines.append(f"### {label} (last 30 of {len(rows)} steps)")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k in keys:
            lines.append(f"| {k} | {late[k]:.3f} |")
        lines.append(f"\nMean wall seconds: {np.mean([r['wall_seconds'] for r in rows]):.1f}s")
        lines.append("")
    return "\n".join(lines) + "\n"


def plot_pc():
    """Generate plots for P-C metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ["validity", "size_hit_rate", "docking_mean", "qed_mean", "sa_mean", "kl_to_prior"]
    titles = ["Validity", "Size Hit Rate", "Docking (mean)", "QED (mean)", "SA (mean)", "KL to Prior"]

    for name, label, color in [("pc_no_curriculum", "no_curriculum", "blue"),
                                ("pc_curriculum_pairs", "curriculum_pairs", "orange")]:
        path = os.path.join(PROBE_DIR, name, "target0", "metrics.csv")
        if not os.path.exists(path):
            continue
        rows = load_metrics(path)
        steps = [r["step"] for r in rows]
        for ax, metric, title in zip(axes.flat, metrics, titles):
            vals = [r.get(metric, float("nan")) for r in rows]
            ax.plot(steps, vals, label=label, color=color, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "pc_metrics.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved {out_path}")
    return out_path


def projection():
    """Project final grid hours."""
    lines = ["\n## Final Grid Projection\n"]
    # measure step time from P-A
    path = os.path.join(PROBE_DIR, "pa_warmup", "target0", "metrics.csv")
    if os.path.exists(path):
        rows = load_metrics(path)
        mean_s = np.mean([r["wall_seconds"] for r in rows])
    else:
        mean_s = 90.0  # estimate

    lines.append(f"Measured seconds/step: {mean_s:.1f}")
    lines.append("")
    # 4 arms × 4 targets, 1 seed
    # Fixed arms: 525 steps (75 warmup + 450)
    # Adaptive: up to 775 steps (75 warmup + 700)
    n_fixed = 3 * 4 * 525  # 3 fixed arms × 4 targets × 525 steps
    n_adaptive = 4 * 775   # 4 targets × 775 max steps
    total_steps = n_fixed + n_adaptive
    total_hours = total_steps * mean_s / 3600
    lines.append(f"Fixed arms: 3 arms × 4 targets × 525 steps = {n_fixed} steps")
    lines.append(f"Adaptive: 4 targets × 775 max steps = {n_adaptive} steps")
    lines.append(f"Total: {total_steps} steps")
    lines.append(f"At {mean_s:.0f}s/step (sequential): **{total_hours:.1f} hours** ({total_hours/24:.1f} days)")
    lines.append("")
    lines.append("| Run type | Arms | Steps/run | Runs | Total steps | Hours |")
    lines.append("|----------|------|-----------|------|-------------|-------|")
    for arm in ["no_curriculum", "curriculum_weights", "curriculum_pairs"]:
        h = 4 * 525 * mean_s / 3600
        lines.append(f"| {arm} | 1 | 525 | 4 targets | {4*525} | {h:.1f} |")
    h_a = 4 * 775 * mean_s / 3600
    lines.append(f"| adaptive | 1 | ≤775 | 4 targets | ≤{4*775} | ≤{h_a:.1f} |")
    lines.append(f"| **Total** | 4 | | 16 runs | {total_steps} | **{total_hours:.1f}** |")
    return "\n".join(lines) + "\n"


def main():
    report = "# Probe Report\n\n"
    report += pa_report() + "\n"
    report += pb_report() + "\n"
    report += pc_report() + "\n"
    try:
        plot_pc()
        report += "\n![P-C Metrics](pc_metrics.png)\n"
    except Exception as e:
        report += f"\nPlot generation failed: {e}\n"
    report += projection()

    out_path = os.path.join(RESULTS_DIR, "probe_report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
