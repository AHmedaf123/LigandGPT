# Rebuild Report — LigandGPT Experiment Layer

**Date**: 2026-11-18  
**Base commit**: 560ea7c (2026-11-03)  
**Rebuild commits**: 3acc800, e1cd28a + (current)

## Bug Audit (B1–B9)

| Bug | Status | Evidence |
|-----|--------|----------|
| **B1** PPO ratio against behaviour policy | **SOLID** | `generation_finetuning.py:195`: `raw = seq - behaviour[sl]`. Prior enters only through the separate KL term (k3 estimator, line 202). Test: `test_policy_loss.py::PolicyUpdate::test_step_zero_agent_equals_prior` — ratio is 1±1e-3 when agent=behaviour. |
| **B2** Invalid molecules in the loss | **SOLID** | Invalids get reward=0 (warmup: `1.0*V + 1.5*Z` with V=0, Z=0 → R=0; post-warmup: R=0 for invalids). With EMA baseline > 0, advantage is negative → negative gradient signal. No exclusion from loss. |
| **B3** Replay buffer cap at 40% | **FIXED** | Old code capped correctly (`int(0.4 / 0.6 * len(gens))`), but did NOT clear at stage switches. New code clears `replay_buffer = []` and resets `baseline = None` at every stage switch, warmup→post-warmup transition, and adaptive objective introduction (`generation_finetuning.py` stage switch block). Replay fraction logged per step in `replay_fraction` column. |
| **B4** lr 1e-6, ent_coef 0.005, batch 32, micro_batch 1, grad_clip 5.0 | **FIXED** | Old default `micro_batch=4`, now changed to `micro_batch=1`. lr=1e-6 ✓, ent_coef=0.005 ✓, batch_size=32 ✓, grad_clip=5.0 ✓. Gradient norm logged every step. |
| **B5** Fixed per-target reward bounds | **FIXED** | Old code: `Normalizer` class with momentum 0.99 running bounds. New code: `FixedNormalizer` with P5-P95 bounds calibrated from frozen prior samples per target. Bounds stored in `runs/baseline/calibrated_bounds_target<N>.json`. Test: `test_reward.py::TestFixedNormalizer` (4 tests). |
| **B6** Docking failure handling | **FIXED** | Embed timeout changed from 10s → 60s (`scoring_function.py:112`). QVina seed from SMILES hash verified at `scoring_function.py:93`. Failures return `docking=None`, molecule gets scored with `docking_norm=0`. Per-step failure counts logged: `dock_fail_embed`, `dock_fail_meeko`, `dock_fail_nopose`, `dock_fail_timeout`. |
| **B7** Gates are log-only | **FIXED** | Old `score_batch` filtered molecules through `violates_hard_constraints` before docking. New `score_batch` docks ALL valid molecules regardless of SA/torsion/energy/steric values. Gate checks removed from scoring path entirely. Steric, torsion, energy values are computed and logged but do not affect reward or exclusion. |
| **B8** GPU robustness | **FIXED** | `reserve_gpu(gib=4.2)` (was 5.0). `micro_batch=1` (was 4). OOM retry in policy_update: catches `torch.OutOfMemoryError`, clears cache, retries once. |
| **B9** Validity collapse guard | **FIXED** | Moving average of validity over 15 steps tracked via `valid_ma` deque. If MA(15) < 0.25 after warmup, prints loud WARNING. No automatic rollback (per spec: "do not add rollback unless a probe shows collapse"). |

## Files Changed

| File | Change |
|------|--------|
| `generation_finetuning.py` | Complete rewrite: warmup phase, 4 arms, fixed bounds, resume, gates log-only, per-sample detail.jsonl, validity guard |
| `scoring_function.py` | Embed timeout 10→60s; UFF RuntimeError catch in optimize_molecule_geometry |
| `calibrate_bounds.py` | New: calibrate P5-P95 bounds from frozen prior samples |
| `tests/test_reward.py` | New: 16 unit tests for reward system (size term, normalizer, divisor, weights) |
| `run_probes.sh` | New: probe execution script |

## What is NOT verified

- **Adaptive arm thresholds**: The threshold formula (`prior_mean ± 0.25 * prior_SD`) uses an approximation of prior SD from the P5-P95 range. The actual metric tracking uses only the current step's scored molecules, not a rolling 15-step average. This is a simplification that may need tuning.
- **Resume correctness under adaptive arm**: The adaptive state serialization includes `active_objs`, `phase_start`, and `introductions`, but the interaction between resume and the metric tracking window has not been tested in a long run.
- **Docking failure categorization**: The dock_fail_meeko and dock_fail_timeout counters are defined but may not be populated correctly in all cases (the current failure detection is heuristic based on conformer availability).
- **Targets 1 and 2 bounds**: Calibration in progress; will be verified when complete.

## Probe Results

See [probe_report.md](probe_report.md) for full details.

**Summary**: All 4 arms tested on target 0, seed 99. Warmup (75 steps) raises validity 44%→70%. Post-warmup quality runs show all arms improve docking (-6.5 → -7.0 to -7.3), validity (→87-89%), and QED. Adaptive arm successfully introduces all 6 objectives with 5/6 hitting milestones organically. Stage transitions, baseline resets, replay clearing, and resume all verified working.

**Key concern**: KL divergence reaches ~33-35 by step 100+. Recommend increasing kl_coef from 1.0 to 2.0 for the real run.
