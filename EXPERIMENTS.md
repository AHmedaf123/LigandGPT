# Minimum experiments for a strong accept at Scientific Reports

Ranked by acceptance value per GPU hour. The first three are the ones that matter.
Everything below the line is optional.

## Cost model, read this first

Nobody can tell you the GPU hours until you measure one number: **seconds per
training step**. A step is 8 generated molecules, 8 QuickVina dockings and one
policy update. The dockings run on the CPU and dominate.

Measure it in the first 20 steps of your first run, then:

    hours = (steps x targets x seconds_per_step) / 3600

| seconds/step | one 700-step target | one 200-step target |
|---|---|---|
| 10 | 1.9 h | 0.6 h |
| 20 | 3.9 h | 1.1 h |
| 40 | 7.8 h | 2.2 h |

Two code changes cut this by roughly 3 to 6 times and are worth doing before any
run: parallel docking across the batch, and a score cache that survives between
steps. Both are listed in the prompt for the GPU session.

---

## E0. Verify the numbers the paper states. 0 GPU hours, 10 minutes

Print the exact parameter count, the stage weight table, the stage boundaries,
the archive size and the docking settings out of the loaded code. This closes
every `TODO(VERIFY)` in the paper and costs nothing.

**Do this first.** It includes one check that changes what the other experiments
mean: whether the curriculum stage ever reaches the reward function. In the code
as it stands, `_update_curriculum()` is empty and `self.curriculum_stage` never
leaves 1, so the three-stage schedule may not have been active in the runs that
produced the current tables. Confirm it, then tell me before you run anything
long.

## E1. Curriculum on against curriculum off, 5 seeds. The main experiment

Two arms, 5 seeds each, 3 targets, 200 steps. 30 target-runs.
At 20 s/step with parallel docking, roughly 6 to 10 GPU hours. Run it overnight.

This is the one reviewer R1-W1 asked for and the one the paper currently answers
with an acknowledgement instead of evidence. It also settles the curriculum
question in E0 with numbers rather than argument. Report mean and standard
deviation across seeds, not a single run.

## E2. Baselines through your own scoring pipeline. Near 0 GPU hours

The TargetDiff authors released docked generated molecules for the **same 100
CrossDocked targets and the same split your code loads**:
`pocket2mol_vina_docked.pt`, `ar_vina_docked.pt`, `cvae_vina_docked.pt`,
`targetdiff_vina_docked.pt` and the reference set `crossdocked_test_vina_docked.pt`
([Drive folder](https://drive.google.com/drive/folders/19imu-mlwrjnQhgbXpwsLgA17s1Rv70YS?usp=share_link),
linked from the [TargetDiff repo](https://github.com/guanjq/targetdiff)).

Push those molecules through `scoring_function.py` unchanged and you get four
baselines measured by **your** pipeline, on **your** targets, with **your**
docking settings. That is a far stronger table than quoted numbers from four
different papers, and it costs CPU time only. Highest value per hour in this
list by a wide margin.

What is realistic to actually run, if you were asked to:

- **Pocket2Mol**: weights exist but are behind a README in the `ckpt` folder of
  the repo, and sampling 100 pockets is its own day. Use the released molecules.
- **REINVENT 4**: priors are on Zenodo (doi 10.5281/zenodo.15641296), so they
  are genuinely downloadable. But REINVENT is ligand-based, it has no pocket
  input, and a fair comparison needs its own RL run per target. Not in one day.
- **3DMolFormer**: ICLR 2025, no weights I could find. Keep it as a quoted row.
- **MolCRAFT, BindGPT**: quoted rows. Say in the caption which rows are
  rescored by you and which are quoted.

## E3. Leave out one physical term at a time. 3 arms

Energy, steric, torsion, one at a time, 1 seed, 3 targets, 200 steps.
9 target-runs, roughly 2 to 3 GPU hours.

Do the three physical terms only, not all seven. The paper's headline is
physical validity, so these three are the ones a referee will test. Drop the
term from the weight vector and renormalise the rest.

---

## Optional, in order, if the day allows

**E4. Independent rescoring.** Take the final archive molecules and score them
with something that never entered the reward: QuickVina at exhaustiveness 16
instead of 4, and PoseBusters. Answers "the reward grades its own homework" and
is CPU only. If the paper's PoseBusters number was not produced this way, this
is the experiment that makes it real.

**E5. Rerun under the corrected objective.** Remark 1 admits every reported
number came from the older prior-anchored ratio. One target, 700 steps, one
seed, as a consistency check only. If it is not run, the disclosure in Remark 1
stays exactly as it is.

**E6. Train and test overlap.** CPU only, under an hour. Report how many
PDBbind training proteins share a sequence with the CrossDocked test targets.
The surrogate's own split is random, not by protein, so this number has to be in
the paper one way or the other.

## What I would not bother with

Running any baseline generator from scratch. Re-pretraining anything. More than
3 targets per ablation arm, when 5 seeds on 3 targets says more than 1 seed on
15. More than 200 steps for an ablation: the collapse, if it happens, happens
early.
