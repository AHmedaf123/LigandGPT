#!/usr/bin/env python3
"""Evaluate final runs: aggregate training metrics and optionally re-dock top molecules.
Reports distributions — no PASS/FAIL thresholds."""
import argparse, csv, glob, json, os, sys, time
import numpy as np

ARMS = ["no_curriculum", "curriculum_weights", "curriculum_pairs", "adaptive"]
TARGETS = [0, 1, 2, 3]


def find_run_dirs(base, arm, target, seed_pattern="*"):
    pattern = os.path.join(base, f"exp_{arm}_t{target}_s{seed_pattern}")
    return sorted(glob.glob(pattern))


def load_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def safe_float(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def summarize_tail(rows, last_n=30):
    if not rows or len(rows) < last_n:
        return None
    tail = rows[-last_n:]

    def col_stats(col):
        vals = [safe_float(r.get(col)) for r in tail]
        vals = [v for v in vals if v is not None]
        if not vals:
            return {}
        a = np.array(vals)
        return dict(mean=float(np.mean(a)), sd=float(np.std(a)),
                    p5=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)))

    COLS = [
        "validity", "n_valid",
        "docking_mean", "docking_norm_mean",
        "bioactivity_mean", "bioactivity_norm_mean",
        "qed_mean", "qed_norm_mean",
        "sa_mean", "sa_norm_mean",
        "energy_mean", "torsion_mean",
        "heavy_atoms_mean", "size_hit_rate",
        "reward_mean", "kl_seq_sum", "kl_per_token", "entropy",
        "grad_norm", "wall_seconds",
    ]

    out = dict(total_steps=len(rows))
    for c in COLS:
        s = col_stats(c)
        if s:
            out[c] = s
    return out


def print_summary_table(all_results):
    header_cols = ["arm", "target", "seed", "steps",
                   "validity", "docking", "bioactivity", "qed", "sa",
                   "kl_seq", "kl_tok", "reward", "wall_s"]
    fmt = "{:<22} {:>3} {:>4} {:>5}  " + " ".join(["{:>8}"] * 9)
    print(fmt.format(*header_cols))
    print("-" * 120)
    for r in all_results:
        s = r["summary"]
        if s is None:
            print(f"{r['arm']:<22} {r['target']:>3} {r['seed']:>4}  [incomplete or missing]")
            continue

        def g(col):
            v = s.get(col, {})
            return f"{v.get('mean', float('nan')):.3f}" if v else "n/a"

        print(fmt.format(
            r["arm"], r["target"], r["seed"], s["total_steps"],
            g("validity"), g("docking_mean"), g("bioactivity_mean"),
            g("qed_mean"), g("sa_mean"),
            g("kl_seq_sum"), g("kl_per_token"),
            g("reward_mean"), g("wall_seconds"),
        ))


def main():
    p = argparse.ArgumentParser(description="Aggregate final run metrics (no thresholds)")
    p.add_argument("--out", default="runs/final", help="Base directory containing exp_* folders")
    p.add_argument("--window", type=int, default=30, help="Average over last N steps")
    p.add_argument("--arm", help="Filter to this arm")
    p.add_argument("--target", type=int, help="Filter to this target")
    p.add_argument("--json", default=None, help="Write full results JSON to this path")
    args = p.parse_args()

    arms = [args.arm] if args.arm else ARMS
    targets = [args.target] if args.target is not None else TARGETS

    all_results = []
    for target in targets:
        for arm in arms:
            run_dirs = find_run_dirs(args.out, arm, target)
            if not run_dirs:
                all_results.append(dict(arm=arm, target=target, seed="?", summary=None))
                continue
            for rd in run_dirs:
                seed = rd.rsplit("_s", 1)[-1] if "_s" in rd else "?"
                csv_path = os.path.join(rd, f"target{target}", "metrics.csv")
                rows = load_metrics(csv_path)
                summary = summarize_tail(rows, args.window) if rows else None
                all_results.append(dict(arm=arm, target=target, seed=seed, summary=summary))

    print_summary_table(all_results)

    n_complete = sum(1 for r in all_results if r["summary"] is not None)
    n_total = len(all_results)
    print(f"\n{n_complete}/{n_total} runs have enough steps for summary")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Full results written to {args.json}")


if __name__ == "__main__":
    main()
