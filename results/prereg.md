# Pre-registration Amendments

## Amendment 2: Adaptive Milestone Criterion

**Date**: 2026-11-19  
**Reason**: The original threshold (prior mean ± k SD) was vacuous: sa, torsion and energy pass trivially at step 14 for every k (the prior already satisfies them), while docking and bioactivity never pass. This makes the adaptive arm a fixed schedule (forced_intro steps per phase), defeating its purpose.

### Old design (removed)
When objective j is the newest active, check whether the 15-step mean of objective j exceeds `prior_mean ± k × prior_SD`. If yes, introduce the next objective.

### New design (improvement milestone)
When objective j is introduced at step t_j, record **L_j** = mean of its **focus metric** over the 15 steps before t_j (or the first 15 steps if j is the first objective).

The newest objective's milestone is met when its **15-step moving average** has **improved** by at least `k × prior_SD` relative to L_j, **and** the phase has lasted at least 40 steps.

Forced introduction after 100 steps. End rule unchanged: all six active, last active ≥100 steps, never before 450 post-warmup, hard cap at 700 post-warmup (775 total).

### Focus metrics
- **Docking**: size residual = `docking_score - (intercept + slope × heavy_atoms)`, using the per-target linear fit from calibration. This removes the size-docking confound.
- **All other objectives**: raw metric mean (bioactivity, qed, sa, torsion, energy).

### Starting value: k = 0.25

### Offline simulation (probe seed 99 data only)

Simulated on no_curriculum (150 steps), adaptive probe (100 steps), curriculum_pairs (138 steps).  
"L_j from steps 0-14" = baseline level, then find step where 15-step MA improves by ≥ k × prior_SD.

**no_curriculum (150 steps, all objectives active from start):**

| Objective | L_j | Prior SD | k=0.10 | k=0.25 | k=0.50 |
|-----------|-----|----------|--------|--------|--------|
| docking (residual) | 0.517 | 1.005 | 33 | 42 | never |
| bioactivity | -8.158 | 1.409 | 26 | never | never |
| qed | 0.372 | 0.196 | 21 | 31 | 69 |
| sa | 5.624 | 1.619 | 54 | 61 | never |
| torsion | 2.001 | 1.337 | 20 | 34 | 89 |
| energy | 167.777 | 264.164 | 63 | never | never |

**adaptive probe (100 steps, sequential introduction):**

| Objective | Intro step | L_j | k=0.10 | k=0.25 | k=0.50 |
|-----------|-----------|-----|--------|--------|--------|
| docking (residual) | 0 | 0.588 | 23 | 59 | 91 |
| bioactivity | 30 | -8.234 | 50 | never | never |
| qed | 46 | 0.321 | 60 | 62 | 67 |
| sa | 61 | 6.383 | 75 | 75 | 86 |
| torsion | 76 | 1.304 | 95 | never | never |
| energy | 91 | 223.635 | never | never | never |

**At k=0.25**: 3-4 of 6 objectives pass organically in the no_curriculum curve. In the adaptive curve (sequential introduction), docking, qed, and sa pass; bioactivity, torsion, and energy would need forced introduction. This provides meaningful discrimination between objectives that the model genuinely improves on versus those that need forced scheduling.

**Choice of k=0.25**: At k=0.10, nearly everything passes quickly (too easy). At k=0.50, most objectives never pass (too hard, nearly all forced). k=0.25 gives a reasonable mix of organic milestones and forced introductions. This is set once from probe data (seed 99 only) and will not be tuned on final runs.
