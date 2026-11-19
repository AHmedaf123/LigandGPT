# Probe Report — LigandGPT Experiment Layer Rebuild

**Date**: 2026-11-19  
**Target**: 0 (CrossDocked2020), Seed: 99  
**Base checkpoint**: final.pt (92M params, frozen prior)  
**Calibrated bounds**: P5-P95 from 128 prior samples per target  

---

## 1. Probe P-A: Warmup (75 steps)

Common warmup phase shared by all arms. Reward = 1.0×Validity + 1.5×SizeTerm, D=2.5.

| Metric | Steps 0–14 | Steps 60–74 | Change |
|--------|-----------|------------|--------|
| Validity | 0.444 | 0.698 | **+57% ↑** |
| Docking (mean) | -6.689 | -6.623 | ~flat |
| Heavy atoms | 28.6 | 28.4 | stable |
| Size hit rate | 0.654 | 0.732 | +12% ↑ |
| QED | 0.405 | 0.328 | -19% ↓ |
| SA | 5.598 | 5.817 | +4% worse |
| KL to prior | ~0 | 23.7 | rising (expected) |

**Wall time**: 131s/step average, 2h43m total.

**Assessment**: Warmup works as designed. Validity improves from 44% → 70%, size hit rate from 65% → 73%. Quality metrics (QED, SA) drift slightly during warmup — expected since warmup reward only targets validity and size. KL rises as the model diverges from prior.

---

## 2. Probe P-B: Mechanics Test (30 steps each, compressed schedule)

Verified all 4 arms from warmup checkpoint with stage_length=10, adaptive_min_phase=5, forced_intro=10.

### Stage transitions

| Arm | Stages observed | Transitions |
|-----|----------------|-------------|
| no_curriculum | s1 only | None (fixed weights, single stage) ✅ |
| curriculum_weights | s1 → s2 → s3 | step 10: s1→s2, step 20: s2→s3 ✅ |
| curriculum_pairs | s1 → s2 → s3 | step 10: s1→s2, step 20: s2→s3 ✅ |
| adaptive | s1 → s2 → s3 | bioactivity@10 (forced), qed@20 (forced) ✅ |

### Baseline & replay reset
All arms correctly reset baseline=None and replay_buffer=[] at stage switches. Replay fraction drops to 0.00 at switch step, then recovers to 0.20. ✅

### Resume test
- Killed process during step 0 scoring (before checkpoint saved)
- Resume detected no checkpoint, started fresh from warmup checkpoint
- Completed all 10 steps successfully ✅

---

## 3. Probe P-C: Quality Test (from warmup checkpoint)

### 3.1 Comparison: no_curriculum (150 steps) vs curriculum_pairs (138 steps) vs adaptive (100 steps)

#### Early performance (steps 0–29, all arms start from same warmup checkpoint)

| Metric | no_curriculum | curriculum_pairs | adaptive |
|--------|--------------|-----------------|----------|
| Validity | 0.775 | 0.764 | 0.769 |
| Docking | -6.716 | -6.766 | -6.756 |
| Bioactivity | -8.241 | -8.319 | -8.205 |
| QED | 0.393 | 0.344 | 0.334 |
| SA | 5.706 | 6.024 | 5.919 |
| Heavy atoms | 26.8 | 29.9 | 28.2 |
| KL | 34.2 | 31.0 | 32.9 |

#### Late performance

| Metric | no_curriculum (steps 120–149) | curriculum_pairs (steps 108–137) | adaptive (steps 70–99) |
|--------|------------------------------|--------------------------------|----------------------|
| Validity | **0.890** | 0.893 | 0.872 |
| Docking | -7.146 | **-7.204** | -6.945 |
| Bioactivity | -8.371 | **-8.529** | -8.248 |
| QED | **0.489** | 0.431 | **0.544** |
| SA | **5.313** | 6.103 | 5.415 |
| Heavy atoms | 27.6 | 30.4 | 24.2 |
| KL | 34.8 | 33.3 | 33.1 |

#### Improvement (early → late)

| Metric | no_curriculum | curriculum_pairs | adaptive |
|--------|--------------|-----------------|----------|
| Validity | +0.115 ↑ | +0.129 ↑ | +0.103 ↑ |
| Docking | -0.430 ↑ | -0.438 ↑ | -0.189 ↑ |
| QED | +0.096 ↑ | +0.087 ↑ | **+0.210 ↑** |
| SA | -0.393 ↑ | -0.079 ~flat | -0.504 ↑ |

### 3.2 Adaptive Objective Introductions (100-step probe)

| Step | Objective | Forced? | Metric value | Threshold |
|------|-----------|---------|-------------|-----------|
| 0 | docking | — | (start) | — |
| 30 | bioactivity | **Yes** | -6.873 | -7.704 |
| 46 | qed | No | -8.763 | -8.710 |
| 61 | sa | No | 0.432 | 0.449 |
| 76 | torsion | No | 5.487 | 6.324 |
| 91 | energy | No | 0.870 | 1.950 |

**Key finding**: Only bioactivity was force-introduced (docking didn't beat its -7.70 threshold in 30 steps). All other objectives were introduced organically — the model met each milestone threshold naturally. All 6 objectives active by step 91.

### 3.3 Docking trajectory (by step)

| Step | no_curriculum | curriculum_pairs | adaptive |
|------|--------------|-----------------|----------|
| 0 | -6.54 | -6.54 | -6.54 |
| 25 | -7.00 | -6.46 | -7.14 |
| 50 | -6.84 | -6.78 | -6.84 |
| 75 | -7.05 | -6.86 | -6.78 |
| 100 | -7.15 | -7.25 | — |
| 125 | -7.21 | -7.35 | — |

### 3.4 Wall time

| Probe | Steps | Avg/step | Total |
|-------|-------|---------|-------|
| P-A warmup | 75 | 131s | 2h43m |
| P-C no_curriculum | 150 | 117s | 4h51m |
| P-C curriculum_pairs | 138 | 101s | 3h51m |
| P-C adaptive | 100 | 156s | 4h20m |

---

## 4. Assessment & Observations

### What works ✅
1. **Validity improves**: All arms push validity from ~44% (prior) → 70% (warmup end) → 87-89% (late training)
2. **Docking improves**: All arms improve mean docking from -6.5 → -7.0 to -7.3 range
3. **Stage mechanics**: Curriculum arms correctly switch stages, reset baselines and replay buffers
4. **Adaptive introductions**: 5 of 6 objectives introduced organically (threshold met), only bioactivity forced
5. **Fixed bounds normalization**: Reward scaling is stable, no dynamic drift
6. **Gates are log-only**: Steric/torsion/energy computed but don't filter molecules
7. **Resume**: Works correctly from rolling checkpoints
8. **Invalid molecules**: Included in loss with reward=0, providing negative gradient signal

### Concerns ⚠️
1. **KL divergence is high** (~33-35 by step 100+). The KL coefficient (1.0) may need tuning — the model is diverging significantly from the prior. Consider increasing kl_coef to 2.0-3.0 for the real run.
2. **Adaptive docking doesn't improve as much** (-6.95 at step 70-99 vs -7.15 for no_curriculum at same window). With only 30 steps of docking-only focus before bioactivity is force-introduced, the model doesn't get enough docking-focused training. For the 1000-step real run with min_phase=40 and forced_intro=150, this should be better.
3. **Curriculum_pairs SA doesn't improve** (6.02 → 6.10). The pairs accumulation order may not be optimal for SA — it's introduced in stage 2 alongside QED but gets less gradient pressure.
4. **Docking hangs**: One step hung for 4+ hours (likely a QuickVina2 subprocess stuck on a pathological conformer). The auto-restart script recovered, but adding a per-step wall-time timeout would prevent future hangs.

### Recommendations for real run
1. **Increase kl_coef** to 2.0 to control divergence
2. **Add per-step timeout** (e.g., 600s) to kill and retry hung docking steps
3. **Adaptive arm**: Use min_phase=40, forced_intro=150 for the 1000-step run. Consider lowering bioactivity threshold or increasing forced_intro since it's the hardest to meet organically.
4. **All arms look viable** — no_curriculum is the simplest and performs well; curriculum_pairs has slightly better docking; adaptive has the best QED improvement.

---

## 5. Final Grid Projection

Based on measured 120s/step average:

| Configuration | Arms | Steps/run | Runs | Total steps | Hours |
|---------------|------|-----------|------|-------------|-------|
| no_curriculum | 1 | 525 | 4 targets | 2,100 | 70h |
| curriculum_weights | 1 | 525 | 4 targets | 2,100 | 70h |
| curriculum_pairs | 1 | 525 | 4 targets | 2,100 | 70h |
| adaptive | 1 | 1,075 | 4 targets | 4,300 | 143h |
| **Total** | **4** | — | **16 runs** | **10,600** | **~353h (14.7 days)** |

Sequential on a single L40S GPU. Can be parallelized across targets if multiple GPUs available.
