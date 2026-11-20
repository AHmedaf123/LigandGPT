# Prelaunch Report — LigandGPT 16-Run Grid

**Date**: 2026-11-20  
**Author**: Experiment rebuild session  
**Status**: READY FOR REVIEW — do NOT start the grid until owner approves

---

## Item 1 — Resume test

**Status**: PASS

**Method**: For each arm, run N steps fresh ("full"), then separately run K steps fresh ("partial") and resume to N. Compare metrics from step K onward between full and resumed runs.

**Test A — curriculum_pairs** (12 steps full, 8 steps partial, resume to 12):
- Core metrics (validity, n_valid, reward, docking, bioactivity, qed, sa, advantage): **exact match** steps 7–11
- SMILES: **exact match** (ordered)
- grad_norm max relative diff: 3.3×10⁻⁶ (expected CUDA non-determinism)

**Test B — adaptive mid-phase** (20 steps full, 10 steps partial, resume to 20):
- Core metrics: **exact match** steps 9–19
- SMILES: **exact match** (ordered)
- grad_norm max relative diff: 5.0×10⁻⁶
- kl_seq_sum max relative diff: 2.7×10⁻⁵ (one step only)

**Note on curriculum_pairs active_objs=[]**: In curriculum_pairs, objectives are introduced by stage schedule, not adaptively. `active_objs` tracks only adaptive introductions, so it stays empty. This is correct.

**Previous test was invalid**: The old test ran 8 steps then set n_steps=8, so resume saw start_step 8 ≥ n_steps 8 and exited immediately. Fixed by running a separate fresh partial run instead of copytree+truncate.

---

## Item 2 — Adaptive milestone redesign

**Problem**: The original threshold (prior mean ± k×SD) was vacuous. SA, torsion and energy pass trivially at step 14 for every k (the prior already satisfies them). Docking and bioactivity never pass. The adaptive arm collapsed to a fixed schedule.

**New design**: Improvement-based milestone. When objective j is introduced at step t_j, record L_j = mean focus metric over 15 steps before t_j. Milestone met when 15-step MA improves by ≥ k × prior_SD relative to L_j.

**Focus metric for docking**: Size residual = `docking_score - (intercept + slope × heavy_atoms)`, removing the size-docking confound.

**Offline simulation** (from probe seed 99 data, 3 arms, 6 objectives × 3 k values):

| k | Organic passes (no_curriculum) | Organic passes (adaptive) | Comment |
|---|-------------------------------|--------------------------|---------|
| 0.10 | 6/6 | 4/6 | Too easy — nearly all pass |
| 0.25 | 3/6 | 3/6 | Good discrimination |
| 0.50 | 2/6 | 1/6 | Too hard — nearly all forced |

**Choice**: k=0.25. Set once from probe data, will not be tuned on final runs.

**Full simulation table**: See `results/prereg.md` (Amendment 2).

---

## Item 3 — Recalibration with 300+ docked molecules

**Status**: COMPLETE — all 4 targets calibrated.

**Changes**:
- `calibrate_bounds.py` rewritten to sample until ≥300 docked molecules per target
- Uses `dock_batch_with_timeout` (subprocess-based, Item 5) for safety
- Records: n_sampled, n_valid, n_docked, fail counts by cause (embed, meeko, nopose, timeout)
- Bounds include: P5, P95, mean, SD, n for each objective
- Saves 200 scored molecules as `prior_reference_target{t}.json`
- Output dir: `runs/baseline_v2/`

### Docked counts per target

| Target | Old n_docked | New n_docked | n_sampled | n_valid | Docking rate |
|--------|-------------|-------------|-----------|---------|-------------|
| 0 (BSD_ASPTE) | 53 | 302 | 1248 | 433 | 70% of valid |
| 1 (GLMU_STRPN) | 41 | 302 | 1424 | 404 | 75% of valid |
| 2 (GRK4_HUMAN) | 23 (was ~7 earlier) | 300 | 4016 | 804 | 37% of valid |
| 3 (GSTP1_HUMAN) | 70 | 304 | 880 | 409 | 74% of valid |

Target 2 required 4016 samples (vs 880–1424 for others) due to its very low docking success rate.

### Old vs new bounds side-by-side

**Docking (minimize, lower = better):**

| Target | Old P5 | New P5 | Old P95 | New P95 | New mean | New SD | Old n | New n |
|--------|--------|--------|---------|---------|----------|--------|-------|-------|
| 0 | -9.11 | -8.50 | -5.80 | -5.40 | -6.78 | 1.05 | 53 | 302 |
| 1 | -7.60 | -8.09 | -4.60 | -4.71 | -6.41 | 1.08 | 41 | 302 |
| 2 | -11.00 | -9.40 | -5.31 | -4.28 | -7.27 | 1.66 | 23 | 300 |
| 3 | -7.10 | -7.50 | -4.15 | -4.30 | -5.92 | 0.95 | 70 | 304 |

**Bioactivity (minimize):**

| Target | Old P5 | New P5 | Old P95 | New P95 | New mean | New SD |
|--------|--------|--------|---------|---------|----------|--------|
| 0 | -10.68 | -9.15 | -6.04 | -6.40 | -8.02 | 0.87 |
| 1 | -9.05 | -9.19 | -5.92 | -5.74 | -7.82 | 1.06 |
| 2 | -17.23 | -8.77 | -7.72 | -6.60 | -8.04 | 0.73 |
| 3 | -8.70 | -8.66 | -4.55 | -4.50 | -6.99 | 1.36 |

Target 2 bioactivity old P5 was -17.23 (extreme outlier from n=23); new P5 is -8.77 — much more stable.

**QED (maximize, higher = better):**

| Target | Old P5 | New P5 | Old P95 | New P95 | New mean | New SD |
|--------|--------|--------|---------|---------|----------|--------|
| 0 | 0.077 | 0.121 | 0.723 | 0.785 | 0.421 | 0.207 |
| 1 | 0.130 | 0.173 | 0.785 | 0.822 | 0.496 | 0.199 |
| 2 | 0.060 | 0.094 | 0.484 | 0.736 | 0.347 | 0.213 |
| 3 | 0.125 | 0.143 | 0.724 | 0.700 | 0.412 | 0.174 |

**SA (minimize):**

| Target | Old P5 | New P5 | Old P95 | New P95 | New mean | New SD |
|--------|--------|--------|---------|---------|----------|--------|
| 0 | 4.06 | 3.32 | 9.39 | 7.81 | 5.62 | 1.39 |
| 1 | 3.37 | 2.91 | 7.75 | 7.42 | 5.17 | 1.36 |
| 2 | 5.63 | 4.45 | 9.52 | 8.49 | 6.68 | 1.23 |
| 3 | 2.32 | 2.62 | 8.22 | 7.60 | 4.86 | 1.52 |

**Energy (minimize):**

| Target | Old P5 | New P5 | Old P95 | New P95 | New mean | New SD |
|--------|--------|--------|---------|---------|----------|--------|
| 0 | 34.3 | 27.6 | 903.4 | 414.1 | 148.1 | 153.2 |
| 1 | 25.3 | 18.3 | 360.6 | 437.6 | 122.4 | 131.6 |
| 2 | 114.7 | 29.8 | 729.2 | 610.7 | 224.2 | 198.5 |
| 3 | 8.5 | 4.6 | 463.0 | 294.7 | 83.3 | 127.9 |

Energy P95 much tighter: target 0 dropped 903→414, target 3 dropped 463→295.

**Torsion (minimize):**

| Target | Old P5 | New P5 | Old P95 | New P95 | New mean | New SD |
|--------|--------|--------|---------|---------|----------|--------|
| 0 | 0.085 | 0.078 | 4.48 | 4.21 | 1.93 | 1.29 |
| 1 | 0.434 | 0.461 | 4.63 | 4.71 | 2.22 | 1.27 |
| 2 | 0.898 | 0.120 | 7.41 | 5.38 | 2.45 | 1.73 |
| 3 | 0.012 | 0.161 | 5.24 | 4.75 | 2.10 | 1.38 |

### Docking vs heavy atoms linear fit

| Target | Old slope | New slope | Old r | New r | Old n | New n |
|--------|----------|----------|-------|-------|-------|-------|
| 0 | -0.046 | -0.007 | -0.43 | -0.08 | 40 | 302 |
| 1 | -0.113 | -0.081 | -0.81 | -0.76 | 31 | 302 |
| 2 | null | -0.016 | null | -0.14 | null | 300 |
| 3 | -0.076 | -0.052 | -0.65 | -0.59 | 51 | 304 |

Target 0 size-docking confound nearly vanished (slope -0.046 → -0.007) with proper sample size. Target 2 had no fit in old data; new fit shows very weak size dependence (slope -0.016).

**Prior docking mean -7.45 vs warmup -6.69**: The prior samples without any reward signal. During warmup, the model is already being updated (warmup is not frozen — it uses docking-only reward to train the model). The difference likely reflects that the warmup policy generates slightly different molecular distributions than the frozen prior.

---

## Item 4 — KL investigation and dual logging

**Finding**: The loss-side KL was always correct (K3 estimator, per-token, kl_coef=1.0). The logged metric was per-sequence sum, which didn't match the loss convention.

**Fix**: Now log both:
- `kl_seq_sum`: per-sequence sum of `(log(agent) - log(prior)) × mask`, averaged over batch
- `kl_per_token`: same, divided by number of tokens per sequence

**Actual KL values** (from probe data, correcting prior report errors):

| Phase | kl_seq_sum | kl_per_token |
|-------|-----------|-------------|
| Warmup step 0 | 0.0 | 0.0 |
| Warmup end (step 74) | ~1.4 | ~0.12 |
| No_curriculum step 150 | ~5.0 | ~0.42 |

The prior report's claim of "KL = 23.7 at warmup end" was **incorrect**. The number 23.7 does not appear in any probe CSV. Maximum warmup KL was 2.48. kl_coef=1.0 is preregistered and will not be changed.

---

## Item 5 — Subprocess-based docking timeouts

**Problem**: ThreadPoolExecutor with `as_completed` timeout doesn't kill stuck threads. A stuck Meeko `prepare()` call hangs indefinitely because threads can't be killed, and the executor's context manager waits for all threads.

**Fix**: Replaced with `multiprocessing.Process` + `Pipe` + `kill`:
- `_dock_worker()`: runs in a child process, docks one molecule, sends result via Pipe
- `dock_batch_with_timeout()`: manages pool of Process workers with per-molecule timeout (default 180s), kills stuck processes with `p.kill()`

**Also**: QVina subprocess timeout increased 60→120s, `start_new_session=True`, process group kill on timeout.

**Tested**: Monkey-patched `prepare_ligand_molecule` to sleep forever. Process killed after 10s. No leftover processes confirmed with `ps aux`.

---

## Item 6 — Extra logging

All additions to metrics.csv row (FIELDS list in generation_finetuning.py):

| Field | Description |
|-------|-------------|
| `kl_seq_sum`, `kl_per_token` | Dual KL metrics (Item 4) |
| `heavy_atoms_mean` | Mean heavy atom count |
| `heavy_atoms_lt18/18_28/28_38/gt38` | Heavy atom histogram bins |
| `size_hit_rate` | Fraction in target range [18, 38] |
| `ligand_efficiency_mean` | docking / heavy_atoms |
| `steric_mean`, `energy_mean`, `torsion_mean` | 3D quality metrics |
| `dock_fail_embed/meeko/nopose/timeout` | Docking failure breakdown |
| `t_sample`, `t_score`, `t_update`, `wall_seconds` | Timing splits per step |

**Additional logging files**:
- `gradient_cosines.jsonl`: pairwise cosine similarity between per-objective policy gradients, every 25 post-warmup steps
- `poses/step{N}_top{K}.pdbqt`: top-3 re-docked poses per step
- `config.json`: full argument dump at run start

---

## Item 7 — eval_final.py

**Rewritten** to report distributions without thresholds:
- `--window` flag (default 30) for last-N-step statistics (mean, SD, P5, P95)
- Reads from new folder structure (`exp_<arm>_t<target>_s<seed>/`)
- No PASS/FAIL — reports numbers for the reader to judge
- `--json` flag for machine-readable output

**Evaluation tools availability**:
- QVina2: available (exhaustiveness 4 during training)
- PoseBusters: installed (v0.6.5)
- PoseCheck: installed (v1.3.1, with pandas version mismatch warning — functional)
- GNINA: **not installed** (optional, 16–30s per molecule on 4 CPU is expensive)

**Not verified**: Timing of evaluation tools on 20 molecules.

---

## Item 8 — Pocket and GNN leakage check

### 8a. Target pocket identities

| Target | UniProt | PDB IDs | Pocket atoms | Prompt tokens | Generation budget |
|--------|---------|---------|-------------|--------------|-------------------|
| 0 | BSD_ASPTE_1_130 | 2z3h, 1wn6 | 409 | 1641 | 200 (fits in 2048) |
| 1 | GLMU_STRPN_2_459 | 4aaw, 4ac3 | 393 | 1577 | 200 (fits in 2048) |
| 2 | GRK4_HUMAN_1_578 | 4yhj | 440 | 1765 | 200 (fits in 2048) |
| 3 | GSTP1_HUMAN_2_210 | 14gs, 20gs | 255 | 1025 | 200 (fits in 2048) |

**No target excluded by prompt length**: Largest prompt (target 2, 1765 tokens) + max_new_tokens (200) = 1965 < context_length (2048).

### 8b. GNN architecture and leakage

The bioactivity GNN (`AdvancedGNN`) is a **ligand-only** model:
- Input: molecular graph (node_input_dim=75, edge_input_dim=12 — standard atom/bond features)
- Output: scalar pK prediction
- **Does not see the protein pocket**

The GNN was fine-tuned on **PDBbind refined set 2020** (from checkpoint config: `index_file: INDEX_refined_data.2020`, `pdbbind_dir: refined-set`), pretrained on ChEMBL.

**PDBbind overlap check — NOT VERIFIED**: The PDBbind refined set index file is not on this machine (`/home/coder/project/data/refined-set/` does not exist). I cannot directly verify whether PDB IDs 2z3h, 1wn6, 4aaw, 4ac3, 4yhj, 14gs, 20gs appear in the PDBbind refined 2020 training set.

**Risk assessment**: Even if these PDB IDs appear in PDBbind, the GNN is a ligand-only model — it predicts general bioactivity from molecular structure, not binding affinity to a specific protein. The GNN does not know which target we're docking against. Its bias (if any) would be toward PDBbind-like molecules in general, not toward ligands for our specific targets. This is a weaker form of leakage than if the GNN saw the protein.

### 8c. CrossDocked2020 train/test split

All 4 targets come from the **test** split of `crossdocked_pocket10_pose_split.pt`. The LigandGPT transformer was pretrained on the train split. The test split contains 100 pockets; we use the first 4 (indices 0–3 in the test split).

---

## Item 9 — Queue restructure

**run_queue.py** rewritten:
- `--seeds` flag (default: `[5]`), supports multiple seeds
- Folder naming: `runs/final/exp_<arm>_t<target>_s<seed>/`
- Order: for each target, run all 4 arms before moving to next target
- `--bounds_dir` flag points to new calibrated bounds (default: `runs/baseline_v2/`)
- `--adaptive_k 0.25` passed to adaptive arm
- Dry-run verified: `python run_queue.py --dry-run --target 0` shows correct commands

---

## Item 10 — Final projection, commit, bundle

**Status**: READY

### Time projection

Per-step wall time from probes (target 0):
- Warmup: ~131s/step (75 steps = ~2.7h)
- Post-warmup no_curriculum: ~117s/step (450 steps = ~14.6h)
- Post-warmup adaptive: ~117s/step (700 steps = ~22.8h)

Per-run estimates:
- no_curriculum/curriculum_weights/curriculum_pairs: 75 + 450 = 525 steps × ~120s = ~17.5h
- adaptive: 75 + 700 = 775 steps × ~120s = ~25.8h

**16-run grid** (sequential, one at a time):
- 12 non-adaptive runs × 17.5h + 4 adaptive runs × 25.8h = 210 + 103 = **~313 hours (~13 days)**

### Checklist summary

| Item | Status |
|------|--------|
| 1. Resume test | PASS — exact metric match, grad_norm diff ≤5×10⁻⁶ |
| 2. Adaptive milestone | Redesigned, k=0.25, offline simulation verified |
| 3. Recalibration | COMPLETE — 300+ docked per target |
| 4. KL dual logging | Fixed and verified |
| 5. Subprocess docking | Tested with stuck-process kill |
| 6. Extra logging | 20+ new metrics fields |
| 7. eval_final.py | Rewritten with --window flag, PoseBusters + PoseCheck installed |
| 8. Pockets + GNN | 4 targets identified, GNN is ligand-only |
| 9. Queue restructure | Dry-run verified |
| 10. Commit + bundle | This report |

### Remaining unverified items

1. **Item 7 (Tool timing)**: Timing of PoseBusters/PoseCheck on 20 molecules not benchmarked
2. **Item 8b (GNN overlap)**: PDBbind refined 2020 index not on disk — cannot verify whether our target PDB IDs appear in GNN training set
3. **GNINA**: Not installed (optional, expensive on 4 CPU)

---

**STOP. Awaiting owner go-ahead before starting the 16-run grid.**
