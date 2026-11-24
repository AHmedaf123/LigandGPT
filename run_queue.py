#!/usr/bin/env python3
"""Sequential grid: 4 arms × 4 targets × N seeds.
Order: for each target, run the four arms before moving to the next target.
Skips finished runs, resumes from checkpoint up to 3 times, dry-run mode."""
import argparse, os, subprocess, sys

ARMS = ["no_curriculum", "curriculum_weights", "curriculum_pairs", "adaptive"]
TARGETS = [0, 1, 2, 3]
WARMUP_STEPS = 75
POST_WARMUP = {"no_curriculum": 450, "curriculum_weights": 450, "curriculum_pairs": 450, "adaptive": 700}
STAGE_LENGTH = {"no_curriculum": 450, "curriculum_weights": 150, "curriculum_pairs": 150, "adaptive": 700}
MAX_RETRIES = 3


def run_steps(arm):
    return WARMUP_STEPS + POST_WARMUP[arm]


def run_dir(base, arm, target, seed):
    return os.path.join(base, f"exp_{arm}_t{target}_s{seed}")


def metrics_path(base, arm, target, seed):
    return os.path.join(run_dir(base, arm, target, seed), f"target{target}", "metrics.csv")


def is_finished(base, arm, target, seed):
    p = metrics_path(base, arm, target, seed)
    if not os.path.exists(p):
        return False
    with open(p) as f:
        n = sum(1 for _ in f) - 1
    return n >= run_steps(arm)


def current_step(base, arm, target, seed):
    p = metrics_path(base, arm, target, seed)
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return max(sum(1 for _ in f) - 1, 0)


def build_cmd(arm, target, seed, base, prior_ckpt, bounds_dir):
    out = run_dir(base, arm, target, seed)
    bounds = os.path.join(bounds_dir, f"calibrated_bounds_target{target}.json")
    n = run_steps(arm)
    cmd = [
        sys.executable, "generation_finetuning.py",
        "--ckpt_load_path", prior_ckpt,
        "--targets", str(target), "--seed", str(seed),
        "--arm", arm,
        "--n_steps", str(n), "--warmup_steps", str(WARMUP_STEPS),
        "--stage_length", str(STAGE_LENGTH[arm]),
        "--bounds_file", bounds,
        "--out", out,
        "--resume",
    ]
    if arm == "adaptive":
        cmd += ["--adaptive_min_phase", "40", "--adaptive_forced_intro", "100", "--adaptive_k", "0.25"]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/final")
    p.add_argument("--prior_ckpt", default="final.pt",
                    help="Pretrained model checkpoint (prior and agent starting weights)")
    p.add_argument("--bounds_dir", default="runs/baseline_v2",
                    help="Dir containing calibrated_bounds_targetN.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[5],
                    help="Seeds to run (default: [5])")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--arm", help="Run only this arm")
    p.add_argument("--target", type=int, help="Run only this target")
    args = p.parse_args()

    arms = [args.arm] if args.arm else ARMS
    targets = [args.target] if args.target is not None else TARGETS
    seeds = args.seeds

    total = len(arms) * len(targets) * len(seeds)
    done = 0

    for target in targets:
        for arm in arms:
            for seed in seeds:
                tag = f"exp_{arm}_t{target}_s{seed}"
                if is_finished(args.out, arm, target, seed):
                    step = current_step(args.out, arm, target, seed)
                    print(f"[SKIP] {tag}: already done ({step}/{run_steps(arm)} steps)")
                    done += 1
                    continue

                cmd = build_cmd(arm, target, seed, args.out, args.prior_ckpt, args.bounds_dir)
                step = current_step(args.out, arm, target, seed)
                print(f"[RUN]  {tag}: {step}/{run_steps(arm)} steps completed")
                if args.dry_run:
                    print(f"       cmd: {' '.join(cmd)}")
                    continue

                for attempt in range(1, MAX_RETRIES + 1):
                    print(f"       attempt {attempt}/{MAX_RETRIES}...")
                    ret = subprocess.call(cmd)
                    if ret == 0 or is_finished(args.out, arm, target, seed):
                        done += 1
                        print(f"[DONE] {tag}")
                        break
                    step = current_step(args.out, arm, target, seed)
                    print(f"       died at step {step}, retrying...")
                else:
                    print(f"[FAIL] {tag}: gave up after {MAX_RETRIES} attempts at step {step}")

    print(f"\nCompleted: {done}/{total}")


if __name__ == "__main__":
    main()
