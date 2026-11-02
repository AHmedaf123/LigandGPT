# Size-bias audit

Molecules: 486 collected, 477 after de-duplication by canonical SMILES. prior = baseline 93 + 93 valid of 256 fresh prior samples (unselected, n=186); archive = Pareto archives of the 3 probes (selected survivors, n=291). Sources: {'probe_curriculum': 100, 'probe_fixed_avg': 96, 'probe_fixed_literal': 95, 'baseline': 93, 'fresh': 93}. Heavy atoms: mean 22.2 (prior 28.6, archive 18.1). Bounds: 'initial' = the RL Normalizer's starting ranges, 'adapted' = P5/P95 of the 104 gate-passing prior molecules (where the Normalizer drifts to, momentum 0.99). Spearman is rank-based, so the per-objective correlations are the same under both; only combined rewards depend on the bounds.

## 2. Spearman of the normalised reward with heavy atoms (reward up = better, so a NEGATIVE rho = the term favours SMALLER molecules)

| objective | all [95% CI] | prior only (unselected) | archives only (selected) | weight x slope per +10 atoms, prior, initial bounds | same, adapted bounds | verdict (prior) | verdict (all) |
|---|---|---|---|---|---|---|---|
| docking | +0.69 [+0.62, +0.75] | +0.65 [+0.51, +0.76] | +0.75 [+0.69, +0.81] | +0.018 | +0.070 | toward larger | toward larger |
| bioactivity | +0.70 [+0.63, +0.75] | +0.62 [+0.48, +0.72] | +0.77 [+0.70, +0.82] | +0.009 | +0.026 | toward larger | toward larger |
| qed | -0.19 [-0.27, -0.09] | -0.54 [-0.64, -0.43] | +0.14 [+0.04, +0.24] | -0.011 | -0.018 | toward smaller | toward smaller |
| sa | -0.76 [-0.80, -0.71] | -0.80 [-0.86, -0.73] | -0.69 [-0.74, -0.62] | -0.016 | -0.028 | toward smaller | toward smaller |
| torsion | -0.17 [-0.26, -0.08] | -0.23 [-0.36, -0.09] | -0.07 [-0.20, +0.04] | -0.002 | -0.003 | toward smaller | toward smaller |
| energy | -0.82 [-0.85, -0.78] | -0.69 [-0.78, -0.58] | -0.84 [-0.87, -0.79] | -0.033 | -0.037 | toward smaller | toward smaller |
| steric (weight 0 now, gate only) | -0.56 [-0.63, -0.48] | -0.49 [-0.62, -0.34] | -0.54 [-0.62, -0.45] |  |  | toward smaller | toward smaller |
| COMBINED, stage 2 NEW weights | -0.08 [-0.17, +0.02] | -0.35 [-0.49, -0.20] | +0.20 [+0.09, +0.31] | -0.035 | +0.010 | toward smaller | neutral |
| COMBINED, stage 2 weights of the probes | -0.06 [-0.15, +0.04] | -0.30 [-0.45, -0.14] | +0.23 [+0.10, +0.35] | -0.031 | +0.022 | toward smaller | neutral |
| RL reward of a valid molecule (gate or dock fail = 0), NEW st. 2 | -0.28 [-0.36, -0.19] | -0.53 [-0.62, -0.42] | +0.16 [+0.04, +0.27] | -0.060 | -0.050 | toward smaller | toward smaller |

Verdict rule: |rho| >= 0.15 and a bootstrap CI that excludes 0. 'weight x slope' = the term's contribution to the change of the combined NEW stage-2 reward per +10 heavy atoms (OLS on the complete-case prior molecules; the six terms add up to the COMBINED row; the RL-reward row is the slope of the reward with gate fails = 0).

Mean normalised reward by heavy-atom bin (prior only, initial bounds):

| heavy atoms | n | docking | bioactivity | qed | sa | torsion | energy | combined (scored) | RL reward (gate fail = 0) |
|---|---|---|---|---|---|---|---|---|---|
| <=14 | 23 | 0.44 | 0.47 | 0.53 | 0.69 | 0.83 | 0.87 | 0.57 | 0.57 |
| 15-19 | 40 | 0.53 | 0.63 | 0.49 | 0.60 | 0.60 | 0.58 | 0.57 | 0.39 |
| 20-24 | 39 | 0.54 | 0.69 | 0.46 | 0.52 | 0.59 | 0.47 | 0.54 | 0.30 |
| 25-29 | 31 | 0.59 | 0.70 | 0.46 | 0.46 | 0.57 | 0.37 | 0.54 | 0.32 |
| >=30 | 53 | 0.61 | 0.72 | 0.22 | 0.27 | 0.61 | 0.19 | 0.50 | 0.06 |


## 3. Gate pass rate by heavy-atom tercile (stage 2 gates unless stated; per-gate rates are over all valid molecules, a missing 3D build counts as a fail; the archives passed their gates by construction, so the unselected prior is the relevant block)

**unselected prior samples**

| gate (pass rate) | T1 (5-19 atoms, n=63) | T2 (20-27 atoms, n=63) | T3 (28-132 atoms, n=60) | all |
|---|---|---|---|---|
| 3D build ok | 94% | 95% | 90% | 93% |
| energy <= 200 | 83% | 78% | 43% | 68% |
| torsion <= 4 | 89% | 87% | 82% | 86% |
| SA <= 6.805 | 97% | 90% | 30% | 73% |
| steric <= 15 | 94% | 95% | 90% | 93% |
| ALL cheap gates, stage 2 | 81% | 65% | 20% | 56% |
| all cheap gates, stage 1 | 84% | 81% | 50% | 72% |
| all cheap gates, stage 3 | 71% | 37% | 10% | 40% |
| stage 2 gates and docking ok | 78% | 56% | 17% | 51% |

**all molecules including archives**

| gate (pass rate) | T1 (4-17 atoms, n=181) | T2 (18-23 atoms, n=157) | T3 (24-132 atoms, n=139) | all |
|---|---|---|---|---|
| 3D build ok | 99% | 97% | 95% | 97% |
| energy <= 200 | 98% | 90% | 70% | 87% |
| torsion <= 4 | 98% | 94% | 90% | 94% |
| SA <= 6.805 | 100% | 97% | 64% | 89% |
| steric <= 15 | 99% | 97% | 95% | 97% |
| ALL cheap gates, stage 2 | 97% | 87% | 55% | 82% |
| all cheap gates, stage 1 | 98% | 92% | 75% | 89% |
| all cheap gates, stage 3 | 86% | 65% | 27% | 62% |
| stage 2 gates and docking ok | 97% | 82% | 54% | 79% |

## 4. Validity vs token length (512 prior samples, 8 equal-count bins)

| tokens | n | valid | never terminated (hit the 200-token budget) | valid among terminated | mean RL reward (invalid -1, gate fail 0, stage 2 NEW) |
|---|---|---|---|---|---|
| 12-28 | 65 | 42% | 0% | 42% | -0.35 |
| 29-34 | 73 | 47% | 0% | 47% | -0.36 |
| 35-39 | 60 | 45% | 0% | 45% | -0.40 |
| 40-46 | 63 | 48% | 0% | 48% | -0.38 |
| 47-54 | 64 | 39% | 0% | 39% | -0.50 |
| 55-68 | 60 | 33% | 0% | 33% | -0.65 |
| 69-143 | 64 | 25% | 0% | 25% | -0.74 |
| 144-200 | 63 | 11% | 41% | 19% | -0.89 |

Spearman of token length with validity: all samples -0.21 [-0.29, -0.12]; terminated samples only -0.16 [-0.24, -0.07]; token length with the RL reward (invalid = -1): -0.28 [-0.35, -0.19].

## 5. Docking and ligand efficiency vs heavy atoms (docking is negative = better; ligand efficiency = docking / heavy atoms, also negative = better)

| set | quantity | n | OLS slope per heavy atom | Spearman [95% CI] | Pearson |
|---|---|---|---|---|---|
| all | docking (kcal/mol) | 430 | -0.0954 | -0.69 [-0.75, -0.62] | -0.63 |
| all | ligand efficiency (kcal/mol/atom) | 430 | +0.0134 | +0.92 [+0.90, +0.94] | +0.86 |
| prior | docking (kcal/mol) | 139 | -0.0543 | -0.65 [-0.77, -0.52] | -0.44 |
| prior | ligand efficiency (kcal/mol/atom) | 139 | +0.0102 | +0.92 [+0.89, +0.95] | +0.88 |
| archive | docking (kcal/mol) | 291 | -0.1433 | -0.75 [-0.81, -0.68] | -0.80 |
| archive | ligand efficiency (kcal/mol/atom) | 291 | +0.0152 | +0.91 [+0.89, +0.93] | +0.86 |


Mean by heavy-atom bin (docking and ligand efficiency: all molecules; the last two columns: prior only):

| heavy atoms | n | docking | ligand efficiency | stage 2 gate pass | RL reward |
|---|---|---|---|---|---|
| <=14 | 117 | -5.44 | -0.546 | 100% | 0.57 |
| 15-19 | 116 | -6.59 | -0.387 | 70% | 0.39 |
| 20-24 | 113 | -7.00 | -0.324 | 69% | 0.30 |
| 25-29 | 58 | -7.25 | -0.276 | 58% | 0.32 |
| >=30 | 73 | -8.05 | -0.230 | 15% | 0.06 |


## 6. Candidate fixes tested offline (combined NEW stage-2 reward of the scored molecules vs heavy atoms; adapted bounds, per-atom/LE bounds = P5/P95 of the same gate-passers)

| reward variant | prior: Spearman [95% CI] | prior: slope per +10 atoms | all: Spearman | all: slope |
|---|---|---|---|---|
| current terms | +0.22 [+0.05, +0.40] | +0.010 | +0.37 [+0.28, +0.46] | +0.048 |
| (A) energy and torsion per heavy atom | +0.36 [+0.18, +0.52] | +0.032 | +0.47 [+0.39, +0.55] | +0.071 |
| (C) docking as ligand efficiency | -0.85 [-0.89, -0.80] | -0.165 | -0.84 [-0.87, -0.81] | -0.187 |
| (A)+(C) | -0.82 [-0.87, -0.75] | -0.144 | -0.81 [-0.84, -0.78] | -0.164 |

Positive = the combined reward favours LARGER molecules.

Raw value vs heavy atoms (+ = grows with size):

| quantity | prior: Spearman [95% CI] | all |
|---|---|---|
| UFF energy | +0.71 [+0.60, +0.79] | +0.82 [+0.78, +0.85] |
| torsion strain | +0.23 [+0.10, +0.37] | +0.17 [+0.09, +0.26] |
| energy per heavy atom | +0.44 [+0.30, +0.56] | +0.63 [+0.57, +0.69] |
| torsion per heavy atom | -0.35 [-0.47, -0.22] | -0.29 [-0.38, -0.20] |
| steric | +0.49 [+0.34, +0.62] | +0.56 [+0.49, +0.62] |
| SA | +0.80 [+0.73, +0.85] | +0.76 [+0.71, +0.80] |
| QED | -0.54 [-0.65, -0.43] | -0.19 [-0.27, -0.09] |

UFF energy / heavy_atoms^k is size-neutral in the prior at k = 1.8 (k = 1 is 'per heavy atom').

(A) at the gate: stage 2 pass rate by size tercile (prior), current threshold -> a per-atom threshold with the same overall pass rate:

| gate | current threshold | T1 (small) | T2 | T3 (large) |
|---|---|---|---|---|
| energy gate | 200 (pass 73%) | 83% -> 78% | 78% -> 71% | 43% -> 55% |
| torsion gate | 4 (pass 92%) | 89% -> 78% | 87% -> 90% | 82% -> 90% |


(B) share of molecules below a heavy-atom floor and the RL reward (stage 2 NEW, prior) they receive:

| heavy atoms | prior samples | prior gate-passers | probe archives | RL reward below | RL reward at or above |
|---|---|---|---|---|---|
| < 12 | 8% | 13% | 19% | 0.56 | 0.27 |
| < 15 | 12% | 22% | 32% | 0.57 | 0.25 |
| < 18 | 23% | 36% | 48% | 0.51 | 0.22 |
| < 20 | 34% | 49% | 58% | 0.45 | 0.20 |

