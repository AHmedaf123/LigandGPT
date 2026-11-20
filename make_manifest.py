#!/usr/bin/env python3
"""Generate a manifest of all run artifacts: checksums, step counts, file sizes."""
import argparse, os, hashlib, json, glob

ARMS = ["no_curriculum", "curriculum_weights", "curriculum_pairs", "adaptive"]
TARGETS = [0, 1, 2, 3]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path):
    if not os.path.exists(path):
        return None
    return dict(path=path, size_mb=round(os.path.getsize(path) / 1e6, 3), sha256=sha256(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/final")
    p.add_argument("--bounds_dir", default="runs/baseline_v2")
    args = p.parse_args()

    manifest = {"runs": [], "bounds": [], "meta": {}}

    for t in TARGETS:
        path = os.path.join(args.bounds_dir, f"calibrated_bounds_target{t}.json")
        info = file_info(path)
        if info:
            manifest["bounds"].append(info)

    run_dirs = sorted(glob.glob(os.path.join(args.out, "exp_*")))
    for d in run_dirs:
        name = os.path.basename(d)
        entry = dict(name=name, dir=d, files=[])

        target_dirs = sorted(glob.glob(os.path.join(d, "target*")))
        for td in target_dirs:
            csv_path = os.path.join(td, "metrics.csv")
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    n = sum(1 for _ in f) - 1
                entry["steps"] = n
                entry["files"].append(file_info(csv_path))

            for pat in ["ckpt_*.pt", "detail.jsonl", "adaptive_introductions.json",
                        "gradient_cosines.jsonl", "config.json"]:
                for fp in sorted(glob.glob(os.path.join(td, pat))):
                    entry["files"].append(file_info(fp))

        manifest["runs"].append(entry)

    out_path = os.path.join(args.out, "manifest.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {out_path}")
    print(f"  {len(manifest['bounds'])} bound files")
    print(f"  {len(manifest['runs'])} run entries")


if __name__ == "__main__":
    main()
