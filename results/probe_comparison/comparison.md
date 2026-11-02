| metric | arm | 0-19 | 20-39 | 40-59 | 60-79 | 80-99 | last - first |
|---|---|---|---|---|---|---|---|
| validity | probe 1 original (curriculum, old code, no size columns) | 0.397 | 0.519 | 0.639 | 0.719 | 0.758 | +0.361 |
| validity | curriculum (re-run, new columns) | 0.359 | 0.514 | 0.617 | 0.670 | 0.778 | +0.419 |
| validity | fixed, time-average of the repo stage vectors | 0.366 | 0.441 | 0.581 | 0.692 | 0.764 | +0.398 |
| validity | fixed, literal weights from the task | 0.369 | 0.466 | 0.609 | 0.689 | 0.755 | +0.386 |
| docking mean (kcal/mol) | probe 1 original (curriculum, old code, no size columns) | -6.709 ± 0.06 | -6.567 ± 0.07 | -6.289 ± 0.04 | -6.109 ± 0.05 | -6.084 ± 0.04 | +0.625 |
| docking mean (kcal/mol) | curriculum (re-run, new columns) | -6.895 ± 0.10 | -6.531 ± 0.08 | -6.337 ± 0.05 | -6.151 ± 0.05 | -6.110 ± 0.03 | +0.785 |
| docking mean (kcal/mol) | fixed, time-average of the repo stage vectors | -6.597 ± 0.11 | -6.545 ± 0.07 | -6.309 ± 0.05 | -6.194 ± 0.04 | -6.278 ± 0.04 | +0.319 |
| docking mean (kcal/mol) | fixed, literal weights from the task | -6.600 ± 0.10 | -6.445 ± 0.08 | -6.395 ± 0.05 | -6.119 ± 0.04 | -6.103 ± 0.04 | +0.497 |
| heavy atoms, valid molecules | curriculum (re-run, new columns) | 32.043 ± 1.07 | 25.745 ± 0.84 | 24.013 ± 0.75 | 21.844 ± 0.69 | 20.396 ± 0.58 | -11.648 |
| heavy atoms, valid molecules | fixed, time-average of the repo stage vectors | 33.030 ± 1.51 | 28.074 ± 0.94 | 24.336 ± 0.76 | 20.959 ± 0.59 | 20.861 ± 0.46 | -12.169 |
| heavy atoms, valid molecules | fixed, literal weights from the task | 29.945 ± 1.00 | 27.299 ± 0.89 | 24.228 ± 0.94 | 20.347 ± 0.75 | 18.762 ± 0.47 | -11.183 |
| heavy atoms, docked molecules | curriculum (re-run, new columns) | 24.129 ± 0.83 | 20.965 ± 0.55 | 19.980 ± 0.36 | 17.921 ± 0.35 | 16.786 ± 0.27 | -7.343 |
| heavy atoms, docked molecules | fixed, time-average of the repo stage vectors | 22.241 ± 0.99 | 20.429 ± 0.50 | 18.738 ± 0.37 | 17.537 ± 0.32 | 17.473 ± 0.27 | -4.767 |
| heavy atoms, docked molecules | fixed, literal weights from the task | 21.589 ± 0.56 | 20.414 ± 0.52 | 19.201 ± 0.34 | 16.356 ± 0.24 | 16.346 ± 0.27 | -5.243 |
| ligand efficiency (kcal/mol/heavy atom) | curriculum (re-run, new columns) | -0.304 | -0.333 | -0.336 | -0.366 | -0.388 | -0.084 |
| ligand efficiency (kcal/mol/heavy atom) | fixed, time-average of the repo stage vectors | -0.314 | -0.341 | -0.356 | -0.379 | -0.377 | -0.063 |
| ligand efficiency (kcal/mol/heavy atom) | fixed, literal weights from the task | -0.323 | -0.335 | -0.353 | -0.396 | -0.396 | -0.072 |
| SA (docked) | probe 1 original (curriculum, old code, no size columns) | 5.307 | 5.099 | 4.770 | 4.552 | 4.629 | -0.678 |
| SA (docked) | curriculum (re-run, new columns) | 5.395 | 5.118 | 4.946 | 4.585 | 4.490 | -0.904 |
| SA (docked) | fixed, time-average of the repo stage vectors | 5.259 | 4.900 | 4.733 | 4.680 | 4.694 | -0.565 |
| SA (docked) | fixed, literal weights from the task | 5.230 | 5.102 | 4.911 | 4.576 | 4.599 | -0.631 |
| UFF energy (docked) | probe 1 original (curriculum, old code, no size columns) | 92.542 | 77.949 | 59.015 | 46.571 | 49.596 | -42.946 |
| UFF energy (docked) | curriculum (re-run, new columns) | 96.453 | 67.501 | 59.554 | 48.288 | 41.615 | -54.839 |
| UFF energy (docked) | fixed, time-average of the repo stage vectors | 80.559 | 66.927 | 55.290 | 54.440 | 54.478 | -26.081 |
| UFF energy (docked) | fixed, literal weights from the task | 76.022 | 69.169 | 57.193 | 46.377 | 45.779 | -30.242 |
| gate-passing molecules per step | probe 1 original (curriculum, old code, no size columns) | 8.050 | 9.450 | 12.100 | 14.950 | 14.600 | +6.550 |
| gate-passing molecules per step | curriculum (re-run, new columns) | 6.600 | 10.050 | 12.350 | 12.600 | 14.950 | +8.350 |
| gate-passing molecules per step | fixed, time-average of the repo stage vectors | 5.400 | 6.650 | 11.450 | 15.350 | 15.950 | +10.550 |
| gate-passing molecules per step | fixed, literal weights from the task | 5.350 | 7.850 | 10.950 | 15.150 | 18.950 | +13.600 |
| kl_to_prior | probe 1 original (curriculum, old code, no size columns) | 0.381 | 0.971 | 1.994 | 3.717 | 4.504 | +4.123 |
| kl_to_prior | curriculum (re-run, new columns) | 0.367 | 0.848 | 1.617 | 3.271 | 4.614 | +4.247 |
| kl_to_prior | fixed, time-average of the repo stage vectors | 0.323 | 0.734 | 1.277 | 2.275 | 2.645 | +2.323 |
| kl_to_prior | fixed, literal weights from the task | 0.368 | 0.929 | 1.753 | 3.266 | 3.957 | +3.589 |

docking mean over the stage-1 steps (steps 0-32, the same steps in both arms):

- probe 1 original (curriculum, old code, no size columns): -6.659 ± 0.05
- curriculum (re-run, new columns): -6.768 ± 0.07
- fixed, time-average of the repo stage vectors: -6.572 ± 0.07
- fixed, literal weights from the task: -6.540 ± 0.07
