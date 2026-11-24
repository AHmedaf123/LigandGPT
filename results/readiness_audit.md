# LigandGPT Multi-Target Readiness Audit

**Date:** 2026-11-17  
**Auditor:** Claude (automated, read-only)  
**Repo:** `/home/coder/project/LigandGPT`, branch `main`, commit `560ea7c` (2026-11-03)

---

## 1. Repo State

**Branch:** `main`, up to date with `origin/main`, working tree clean (no uncommitted/untracked files).

**Last 5 commits:**
```
560ea7c 2026-11-03 Size-bias audit (offline), QVina seed from the SMILES hash, new stage vectors
9e8f108 2026-11-02 Control probe (no curriculum), 10 s embedding timeout, size columns, Uni-Dock measurement
231a383 2026-11-02 Rebuild the RL core: structured KV-cache sampler, one stage function, gates before docking, behaviour-logp PPO + KL + replay
1e2b98c 2026-11-02 Measure the cost of an RL step (sampling, scoring + docking, update) and project 300/600 steps
340d350 2026-11-02 SA gate from the baseline SA distribution: reject above P90 / P80 / P70 for stages 1 / 2 / 3
```

**Expected files from the brief vs actual:**

| Expected file | Exists? | Actual equivalent |
|---|---|---|
| `probe.py` | NO | `generation_finetuning.py` is the RL training script |
| `run_queue.py` | NO | Does not exist |
| `smoke_test.py` | NO | Does not exist |
| `make_manifest.py` | NO | Does not exist |
| `eval_final.py` | NO | Does not exist |
| `plot_joint.py` | NO | `compare_probes.py` does partial comparison |
| `results/prereg.md` | NO | Does not exist |
| `mo_mpo.py` | YES | Multi-objective reward, curriculum weights, stage logic |
| `scoring_function.py` | YES | All scoring (docking, QED, SA, GNN bioactivity, steric, torsion, energy) |
| `baseline.py` | YES | Prior baseline sampling + scoring |

**Files that ARE present:** `generation_finetuning.py`, `mo_mpo.py`, `scoring_function.py`, `baseline.py`, `model.py`, `tokenizer.py`, `vocabulary.py`, `dataset.py`, `data_utils.py`, `decoding.py`, `diagnose_generation.py`, `compare_probes.py`, `size_audit.py`, `bench_step.py`, `bench_sampling.py`, `check_docking.py`, `GNNmodel.py`, `GNN-finetuning.py`, `GNN-pretraining.py`, `GNNdataset.py`, `unidock_check.py`, `ProtLigVoc.txt`, `final.pt` (367 MB prior checkpoint), `best_model.pt` (44 MB, likely GNN), `requirements.txt`, `tests/` (7 test files).

**.gitignore analysis — what would NOT survive a wipe:**
- `data/` (the entire data directory is gitignored)
- `*.pth` and `*.lmdb` files
- `checkpoints/`, `checkpoints_docking/`, `generated_molecules/`
- `GNN/outputs/`, `GNN/cache/`, `cache/`
- `oracle/` (fpscores.pkl)
- `logs/`

**CRITICAL:** `runs/` is NOT gitignored. Run outputs (metrics.csv, config.json, archive.json, probe.log) are tracked and would survive a wipe. However, the .gitignore ignores `*.pth`, so any saved model checkpoints inside runs/ would be lost. The `final.pt` prior checkpoint is also ignored by `*.pth` — wait, it's `.pt` not `.pth`. Checking: `final.pt` and `best_model.pt` are NOT matched by `*.pth` but are not in .gitignore either. They appear to be tracked.

---

## 2. Recovered Evidence

**Contents of `runs/`:**

| Directory | Files present | Steps completed | Notes |
|---|---|---|---|
| `runs/baseline/` | `baseline.json`, `samples.json`, `step_bench.json`, `step_projection.json` | N/A (sampling, not RL) | Targets 0 and 3, 128 samples each |
| `runs/probe/target0/` | `metrics.csv`, `archive.json`, `config.json` | 100 (steps 0-99) | Curriculum arm, seed 0, target 0 |
| `runs/probe_curriculum/target0/` | `metrics.csv`, `archive.json`, `config.json` | 100 (steps 0-99) | Curriculum re-run, seed 0, target 0 |
| `runs/probe_fixed_literal/target0/` | `metrics.csv`, `archive.json`, `config.json` | 100 (steps 0-99) | Control arm (literal fixed weights), seed 0, target 0 |
| `runs/probe_fixed_avg/target0/` | `metrics.csv`, `archive.json`, `config.json` | 100 (steps 0-99) | Control arm (averaged weights), seed 0, target 0 |

**What is NOT present:**
- No `exp_<arm>_s<seed>/` directories (the naming scheme from the brief does not exist)
- No `detail.jsonl` files in any run
- No `run_meta.json` files in any run
- No `train.log` files (there are `probe.log` files at the arm level, not target level)
- No checkpoints in any run directory
- No 525-step or longer experiment runs — all are 100-step probes

**Conclusion:** Only 100-step probe/diagnostic runs exist. No full experiment runs (525+ steps) have been completed. All probes are on target 0 only.

---

## 3. Target Handling in Code

**How targets are selected:** `generation_finetuning.py` accepts `--targets` as a list of integers (line 389: `ap.add_argument("--targets", type=int, nargs="+", default=[0])`). These are indices into the CrossDocked test split:
```python
split = torch.load("crossdocked_pocket10_pose_split.pt")["test"]  # list of LMDB key indices
dp = pickle.loads(txn.get(keys[split[target]]))  # target is the index into this list
```

**Target IS selectable by command line.** Example: `--targets 0 3` runs targets 0 and 3 sequentially in one process.

**Test split file:** `data/crossdocked_pocket10_pose_split.pt` (present at `/home/coder/project/data/crossdocked_pocket10_pose_split.pt`).

**Pocket length filter:** There is no explicit pocket exclusion filter in `generation_finetuning.py`. The implicit filter is the 2048 context length: `budget = min(args.max_new_tokens, 2048 - L - 1)` (line 270). If a pocket's prompt tokenizes to more than ~1847 tokens (2048 - 200 - 1), the generation budget shrinks. If it exceeds 2047 tokens, the budget goes to 0 or negative and the run would fail or produce nothing useful. Target 0 uses 1641 prompt tokens, leaving a budget of 200.

**Pocket identifiers for indices 0–5:** Not verified — would require loading the LMDB and split file with Python, which is impossible because **no Python environment exists** (see Section 6). The `target_map.json` file (used by `baseline.py`) does NOT exist in the data directory, so pocket-to-PDB-ID mapping is not available without running code. From the probe config, target 0 maps to pocket `BSD_ASPTE_1_130_0` with receptor PDB `2z3h_A_rec`.

**Number of test pockets:** 93 directories exist in `data/test_set/` (not 100 — the brief mentions 100 test pockets with some excluded by a length filter; 7 may have been excluded during data preparation, or the extraction is incomplete).

---

## 4. Per-Target Calibration

**calibrated_bounds.json:** Does NOT exist anywhere in the repo or data directory.

**How normalization actually works:** The `Normalizer` class in `generation_finetuning.py` (line 106) uses:
1. **Static initial bounds** from `mo_mpo.create_objectives()` (line 148–157):
   - docking: [-12.0, 0.0], bioactivity: [-12.0, 0.0], qed: [0.0, 1.0], sa: [1.0, 10.0], energy: [0.0, 200.0], steric: [0.0, 20.0], torsion: [0.0, 5.0]
2. **Dynamic adaptation** with momentum 0.99: each batch's P5/P95 of scored molecules slowly shifts the bounds.

**These bounds are NOT per-target.** They are hardcoded global ranges in `mo_mpo.py`. The dynamic adaptation starts fresh each run (Normalizer is created inside `run_target`), so each run's bounds will naturally drift toward that target's distribution. This is actually reasonable because:
- The initial bounds are wide enough to cover any target.
- The P5/P95 tracking with momentum 0.99 adapts slowly, which is fine for a 525-step run.

**baseline.json** was computed on targets 0 and 3 and contains per-target statistics, but it is used only by `baseline.py` for analysis — `generation_finetuning.py` does not read it.

**Assessment:** Reusing the global initial bounds across targets is NOT a problem because:
1. The bounds are intentionally wide static ranges (not calibrated to a specific target).
2. Dynamic adaptation handles per-target drift automatically.
3. No `calibrated_bounds.json` was ever used by the RL code.

**No change needed for multi-target normalization bounds.**

---

## 5. Queue and Scheduling

### run_queue.py: DOES NOT EXIST

There is no job queue, scheduling, or grid-sweep script. `generation_finetuning.py` runs targets sequentially within a single invocation (`--targets 0 3` runs both in order).

### What exists for launching runs:
```bash
.venv/bin/python generation_finetuning.py \
    --ckpt_load_path final.pt --targets 0 --n_steps 525 --seed 0 \
    --out runs/probe --probe
```
For the control arm (no curriculum):
```bash
.venv/bin/python generation_finetuning.py \
    --ckpt_load_path final.pt --targets 0 --n_steps 525 --seed 0 \
    --fixed_weights avg --out runs/probe_fixed_avg --probe
```

### Output path handling:
The output goes to `<out>/target<N>/` (line 276: `out_dir = os.path.join(args.out, f"target{target}")`). With different `--out` per arm, target folders do NOT collide.

### What would need to change for a multi-target, multi-arm grid:

1. **A queue/driver script** (`run_queue.py`) that:
   - Iterates over (arm, target, seed) combinations
   - Constructs the right `--out` and `--fixed_weights` per arm
   - Skips finished runs (checks if `metrics.csv` exists and has the right number of steps)
   - Resumes crashed runs (NOT SUPPORTED — see below)

2. **Resume support: MISSING.** `generation_finetuning.py` has no `--resume` flag. If a run crashes at step 300, it must restart from step 0. There is no checkpoint saving during training (only `archive.json` is saved at the end). This is a significant risk for 525-step runs (13+ hours each).

3. **The two missing arms:**
   - `curriculum_pairs` (accumulating objectives by stage): NOT IMPLEMENTED in the code
   - `adaptive` (threshold-triggered objective introduction): NOT IMPLEMENTED in the code
   
   The code only supports:
   - **curriculum** (default, no `--fixed_weights`): 3 stages with different weight vectors, all 6 objectives present in all stages
   - **no_curriculum** (`--fixed_weights`): fixed weights throughout, stage-2 gates

### eval_final.py and make_manifest.py: DO NOT EXIST

There are no evaluation or manifest scripts. `compare_probes.py` does a basic comparison of probe metrics but is hardcoded to the specific probe run paths.

### Command for 4 arms × 3 targets × 1 seed:
Cannot be given because only 2 of 4 arms are implemented, and no queue script exists. For the 2 existing arms × 3 targets × 1 seed, one would need to run 6 sequential commands manually.

---

## 6. Environment Readiness

### GPU
- **Model:** NVIDIA L40S
- **VRAM:** 46,068 MiB total, 18,800 MiB used (no running processes — likely cached), 27,268 MiB free
- **Shared:** Yes (the `reserve_gpu()` function at line 223 explicitly waits for 5 GiB to become available, noting "the GPU is shared and a neighbour sometimes fills it completely")

### CPU
- **Cgroup quota:** 400000/100000 = **4 CPUs** (same as earlier hardware)
- **Visible cores:** 80 (host), but cgroup-limited to 4
- **The code correctly uses `cpu_quota()`** to read the cgroup limit

### RAM
- **Total:** 503 GiB, **Available:** 225 GiB (plenty)

### Disk
- **Filesystem:** 1.7 TB total, 865 GB used, 764 GB free (54% used) — plenty

### Python and dependencies

| Component | Status |
|---|---|
| Python / .venv | **MISSING** — no Python interpreter found anywhere on the system. `.venv/` does not exist. `python`, `python3` not on PATH. No conda. |
| PyTorch | **MISSING** (no Python) |
| RDKit | **MISSING** (no Python) |
| Meeko | **MISSING** (no Python) |
| Open Babel (obabel) | **MISSING** — not on PATH |
| PoseBusters | **MISSING** (no Python) |
| QuickVina2 | **PRESENT** at `docking/qvina02` (3.3 MB binary) |
| TDC (PyTDC) | **MISSING** (no Python) |

**The `requirements.txt` documents the needed packages** (torch 2.11.0+cu128, rdkit 2026.9.1, meeko 0.8.0, etc.) and notes a required TDC patch for `rdkit.six`. The venv must be rebuilt before any run.

### Pretrained prior checkpoint
- **Present:** `final.pt` (367 MB) at repo root
- **GNN surrogate:** `best_model.pt` (44 MB) at repo root. Code expects it at `GNN/outputs/pdbbind_finetune/best_model.pt` — this path does NOT exist. Either the best_model.pt at root needs to be moved/symlinked, or the `--gnn_model_path` needs to be overridden (but no such CLI arg exists on `generation_finetuning.py`; it would need to be passed to `initialize_scoring_system()`).

### CrossDocked data

| File | Status |
|---|---|
| `data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` | **PRESENT** (3.8 GB). Download appears complete (wget process no longer running, file size stable). May be incomplete — cannot verify without Python. |
| `data/crossdocked_pocket10_pose_split.pt` | **PRESENT** |
| `data/test_set/` | **PRESENT** (93 pocket directories with `_rec.pdb` and `_rec.pdbqt` files) |
| `data/receptors_pdbqt/` | **MISSING** — the code in `data_utils.py:receptor_paths()` expects PDBQT files at `data/receptors_pdbqt/<dir>/<file>.pdbqt`, but they only exist inside `data/test_set/<dir>/`. A symlink `data/receptors_pdbqt -> data/test_set` or the actual directory is needed. |
| `data/target_map.json` | **MISSING** — used by `baseline.py` only, not by the RL training |
| `oracle/fpscores.pkl` | **MISSING** — TDC downloads it on first use if absent, but needs a Python environment |

### Data download state
The `wget` process downloading the LMDB file is **no longer running**. The file is 3.8 GB and the size has been stable for several minutes. It may have completed successfully, or the download may have been interrupted. Cannot verify LMDB integrity without Python/lmdb.

### Smoke test: SKIPPED
Cannot run a smoke test because **no Python environment exists**. Even if the venv were rebuilt, the missing `data/receptors_pdbqt/` directory would cause docking to fail.

---

## 7. Time Estimate

### Step timing data
From the 100-step probe on target 0:
- **Mean wall time per step:** 76.5 seconds (from `runs/probe/target0/metrics.csv`)
- **Range:** 30–215 seconds (highly variable, dominated by docking; depends on how many molecules pass gates)

From `runs/baseline/step_bench.json` (batch=32):
- Sampling: ~3.8s
- Scoring including docking: ~72s
- Policy update (micro_batch=4): ~0.45s
- **Total estimated:** ~76s per step

The brief estimates 91–95 seconds as validity grows; the probe data shows 76.5s mean over 100 steps. As validity improves in later steps, more molecules pass gates and need docking, so **~90s/step is a reasonable estimate for 525-step runs**.

### Compute hours (at 90 seconds/step, one run at a time)

| Run type | Steps | Hours/run | Formula |
|---|---|---|---|
| Standard arm (525 steps) | 525 | **13.1 h** | 525 × 90 / 3600 |
| Adaptive arm (up to 775 steps) | 775 | **19.4 h** | 775 × 90 / 3600 |

**Grid totals (assuming 3 standard arms + 1 adaptive per target set):**

| Grid | Runs | Computation |Total hours |
|---|---|---|---|
| **(a) 4 arms × 3 targets × 1 seed** | 12 | 9 × 13.1 + 3 × 19.4 | **176 h (7.3 days)** |
| **(b) 4 arms × 3 targets × 3 seeds** | 36 | 27 × 13.1 + 9 × 19.4 | **528 h (22 days)** |
| **(c) 4 arms × 4 targets × 1 seed** | 16 | 12 × 13.1 + 4 × 19.4 | **235 h (9.8 days)** |
| **(d) 4 arms × 4 targets × 3 seeds** | 48 | 36 × 13.1 + 12 × 19.4 | **704 h (29.3 days)** |

**Note:** These estimates assume all 4 arms exist. Currently only 2 arms are implemented, so the actual runnable grids are half these sizes. Also, the adaptive arm does not exist, so all runs would be 525 steps:

| Grid (2 arms only) | Runs | Hours |
|---|---|---|
| 2 arms × 3 targets × 1 seed | 6 | 6 × 13.1 = **79 h (3.3 days)** |
| 2 arms × 3 targets × 3 seeds | 18 | 18 × 13.1 = **236 h (9.8 days)** |
| 2 arms × 4 targets × 1 seed | 8 | 8 × 13.1 = **105 h (4.4 days)** |
| 2 arms × 4 targets × 3 seeds | 24 | 24 × 13.1 = **314 h (13.1 days)** |

---

## 8. Risk Review

### Issues that would silently break a multi-target run

| Risk | Severity | Detail |
|---|---|---|
| **Missing Python environment** | BLOCKER | No `.venv`, no Python on PATH. Nothing can run. |
| **Missing `receptors_pdbqt/` directory** | BLOCKER | `data_utils.py:receptor_paths()` constructs PDBQT paths under `data/receptors_pdbqt/`, but that directory doesn't exist. PDBQT files are inside `data/test_set/` instead. Fix: symlink `data/receptors_pdbqt -> data/test_set`. |
| **GNN checkpoint path mismatch** | BLOCKER | Code expects GNN at `GNN/outputs/pdbbind_finetune/best_model.pt`. File is at repo root `best_model.pt`. Fix: `mkdir -p GNN/outputs/pdbbind_finetune && ln -s ../../../best_model.pt GNN/outputs/pdbbind_finetune/best_model.pt`. |
| **Missing `data` symlink in repo** | BLOCKER | Code uses `--data_path ./data` (default). The repo's `/data` is gitignored as "Symlink to ../data". The symlink doesn't exist. Fix: `ln -s ../data data` inside the repo. |
| **2 of 4 arms not implemented** | MAJOR | `curriculum_pairs` and `adaptive` arms do not exist in the code. Only `curriculum` (default) and `no_curriculum` (`--fixed_weights`) are implemented. |
| **No resume/checkpoint support** | MAJOR | A crash at step 400 of a 13-hour run restarts from step 0. No intermediate checkpoints are saved. |
| **No run_queue.py** | MAJOR | No automated job grid, skip-finished, or resume logic. Must launch each run manually. |
| **No eval_final.py** | MAJOR | No evaluation script for comparing arms across targets. |
| **No make_manifest.py** | MODERATE | No script to generate experiment manifests. |
| **No detail.jsonl output** | MODERATE | Per-molecule detail is not logged (only aggregate metrics.csv). |
| **Shared GPU** | MODERATE | The `reserve_gpu(5)` function blocks until 5 GiB is free. A neighbour's job could delay or block runs. |
| **LMDB integrity unverified** | LOW | Download appears complete (3.8 GB, wget finished) but cannot verify without Python. |
| **target_map.json missing** | LOW | Only used by `baseline.py`, not by RL training. |
| **No warmup phase in code** | INFO | The brief describes a 75-step warmup with `R = 1.0*V + 1.5*Z`. The code has no warmup — it goes straight into the curriculum stages from step 0. The 3-stage curriculum uses `resolve_stage(step, n_steps)` which divides the total steps into equal thirds. |

### Persistence recommendation (DO NOT IMPLEMENT)
After each finished run:
1. `git add runs/<arm>/target<N>/metrics.csv runs/<arm>/target<N>/config.json runs/<arm>/target<N>/archive.json`
2. `git commit -m "Finished <arm> target <N> seed <S>, <steps> steps"`
3. For checkpoints (if added): copy to a separate storage location outside the repo (they would be too large to commit and are gitignored by `*.pth`)
4. Consider a simple shell wrapper that does this automatically at the end of each `generation_finetuning.py` invocation.

---

## Final Verdict Table

| Task | Status | Smallest change needed |
|---|---|---|
| 1. Repo state | NEEDS SMALL CHANGE | Create symlink `data -> ../data` in repo. 6 of 7 expected scripts don't exist (see notes). |
| 2. Recovered evidence | READY (read-only) | Only 100-step probes on target 0 exist. No full experiments. |
| 3. Target handling | READY | `--targets` CLI arg works. Target is an index into the test split. |
| 4. Per-target calibration | READY | Normalization uses wide static initial bounds + dynamic P5/P95 adaptation per run. No change needed. |
| 5. Queue and scheduling | MISSING | Need: `run_queue.py` (job grid), `eval_final.py`, `make_manifest.py`. Two arms (`curriculum_pairs`, `adaptive`) not implemented. No resume support. |
| 6. Environment | MISSING | Python environment must be rebuilt (`.venv`). Create `data/receptors_pdbqt` symlink. Fix GNN checkpoint path. |
| 7. Time estimate | READY | ~13.1 h/run (525 steps), ~90 s/step. Grid (a) = 176 h if all 4 arms existed. |
| 8. Risk review | N/A | See risk table above. |

**Overall: NOT READY.** Three blockers (no Python env, missing data symlinks/paths, 2 of 4 arms unimplemented) and two major gaps (no queue script, no resume support) must be addressed first.

---

## Questions for the Owner

1. **Arms:** The code only implements `curriculum` and `no_curriculum` (fixed weights). The brief describes `curriculum_pairs` (accumulating objectives) and `adaptive` (threshold-triggered). Are these still planned, or should the experiment proceed with only the 2 existing arms?

2. **Warmup phase:** The brief describes a 75-step warmup with `R = 1.0*V + 1.5*Z`. The code has no warmup — it starts the curriculum from step 0. Was warmup removed intentionally, or does it need to be implemented?

3. **Resume support:** A 525-step run takes ~13 hours. If it crashes at step 400, it must restart from scratch. Should we add checkpoint saving and `--resume`? This seems critical for multi-day grids.

4. **Number of seeds:** With 2 arms, 3 targets, 1 seed = 79 hours (3.3 days). With 3 seeds = 236 hours (9.8 days). What's the preferred grid?

5. **GNN checkpoint:** Is `best_model.pt` at repo root the correct GNN bioactivity surrogate? It needs to be at `GNN/outputs/pdbbind_finetune/best_model.pt` for the code to find it.

6. **receptors_pdbqt:** The PDBQT files exist inside `data/test_set/<pocket>/` but the code looks for them at `data/receptors_pdbqt/<pocket>/`. Should we create a symlink, or was there previously a separate prepared-receptor directory?

7. **Target selection:** Which 3–4 target indices should be used? The baseline was run on targets 0 and 3. Should those be included, plus 1–2 new ones? Pocket prompt sizes vary — large pockets may exceed the 2048-token context window.

8. **Evaluation:** How should arms be compared across targets without `eval_final.py`? Is a simple CSV comparison of metrics at step 525 sufficient, or is a formal evaluation script needed?
