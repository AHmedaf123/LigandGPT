# Prompt to paste into the Claude Code session on the GPU machine

Everything between the two lines is the prompt. Upload the paper `.tex`, the
`.bib`, all the code files, `model.pt` and the data directory first, then paste
it.

---

You are working on a journal resubmission. Read this whole brief before you
touch anything. I am the author. My name is Ahmed.

## What this is

LigandGPT is a curriculum-guided reinforcement learning system for
structure-based molecular design. A 12-layer, 12-head, 768-dim transformer
writes SMILES into a protein pocket; a seven-term reward scores each molecule on
docking, bioactivity, QED, synthetic accessibility, steric clash, torsion strain
and internal energy; a three-stage curriculum rotates the weight mass between
the affinity terms, the chemistry terms and the physics terms.

The paper was rejected at AAAI and is being revised for **Scientific Reports**
(Springer Nature). The reviewers' main complaint was that the central claims
rest on acknowledgements rather than evidence, and that some numbers in the text
do not reconcile with the tables. My goal for this session is a paper where no
referee finds a methodology weakness I have not already addressed, and where
every equation matches what the code computes.

I have one day of GPU time. Run only the experiments that are needed to make the
work verifiably strong. Nothing else.

## Files you have

- `sn-article-ligandgpt.tex` and `ligandgpt.bib` — the manuscript. Builds with
  pdflatex + bibtex against `sn-jnl.cls`.
- `generation_finetuning.py` — the RL loop. This is the main entry point.
- `mo_mpo.py` — the reward, the curriculum, the hard constraint gates and the
  Pareto archive.
- `scoring_function.py` — the seven scorers, including QuickVina docking.
- `model.py`, `training_utils.py`, `tokenizer.py`, `vocabulary.py`,
  `dataset.py`, `data_utils.py`, `pretraining.py`.
- `GNNmodel.py`, `GNNdataset.py`, `GNN-pretraining.py`, `GNN-finetuning.py` —
  the bioactivity surrogate.
- `ProtLigVoc.txt` — the 101-token vocabulary.
- `model.pt` — the **pretrained LigandGPT checkpoint**, already trained. It is
  loaded twice in `generation_finetuning.py`: once as the frozen prior and once
  as the starting point for the agent. Pass it as `--ckpt_load_path`. There is
  no pretraining left to do. Do not run `pretraining.py`.
- The bioactivity GNN checkpoint, the `docking/qvina02` binary and the data
  directory with `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`,
  `crossdocked_pocket10_pose_split.pt` and `test_set/`. If any of these is
  missing, say so immediately and stop rather than working around it, because
  every number depends on them.

## Rules I need you to hold

These are not style preferences. Scientific Reports requires a code
availability statement, so a referee can and will read this code after
acceptance. A mismatch found then is a correction notice.

1. **Where an equation and the code disagree, make them agree or state the
   difference.** Never reword a claim so that the mismatch stops being visible.
   If the code is wrong, fix the code and rerun. If the paper is wrong, the
   equation changes. If neither can be settled today, the paper says plainly
   what was run.
2. **Do not change any claim, number, equation or method description in the
   `.tex` on the basis of what the code says until I have approved it.** Write
   every such change into `PAPER_EDITS.md` as a before and after pair with the
   file and function that justifies it, and ask me. You may edit the paper
   freely for framing, wording, ordering and emphasis, and you should, because
   making the writeup stronger is part of the job. The line is: anything that
   changes what the paper asserts goes through me first.
3. **No number in the paper may come from anywhere but a file on disk.** Every
   figure and table entry must trace to a CSV in `results/`. If an experiment
   did not run, its claim does not go in, and the existing `TODO(EXPERIMENT)`
   marker stays.
4. **Do not delete or soften the limitations section.** A blunt limitations
   section is why this paper is defensible. Weaknesses get fixed or disclosed,
   never hidden.
5. **Do not weaken a test, a gate or a threshold to make a result look better.**
   If a run collapses, the collapse is the result.
6. Report what actually happened, including failures, with the output.

## Step 1. Verify the code against the paper. Before any long run

There is one thing to check before everything else.

In `mo_mpo.py`, `update_curriculum_stage(current_step)` sets
`self.current_stage` from the step count, but `_update_curriculum()` is a bare
`pass` and `self.curriculum_stage` is only ever set to 1 in `__init__`.
Meanwhile `_compute_batch_rewards`, `_select_and_add_best_candidate` and the
`get_constraint_thresholds` calls all read **`self.curriculum_stage`**, not
`self.current_stage`.

If I am reading that correctly, the three-stage curriculum never reached the
reward function, and every number in the current paper was produced with stage-1
weights and stage-1 gates for all 700 steps. That is the paper's headline
mechanism, so I need to know before anything else runs.

Confirm it by instrumenting, not by reasoning: print both fields every 50 steps
for a 450-step dry run, or assert on them. Then **stop and tell me what you
found** with the printout. Do not start the long runs and do not edit the paper
until I answer.

The fix, if confirmed, is to have `_update_curriculum()` set
`self.curriculum_stage = self.current_stage`. Once fixed, the comparison between
fixed stage-1 weights and the real schedule becomes experiment E1 below, which
is also the ablation a reviewer demanded. One run answers both.

While you are in there, check each of these against the code and record the
answer in `CODE_VS_PAPER.md`. I have reason to believe several are wrong in the
paper, and I want the list verified from the source rather than from memory:

- Exact transformer parameter count from
  `sum(p.numel() for p in model.parameters())`. The paper says 92M, the README
  says ~85M. Print the real number.
- The seven stage weights, per stage, and whether each column sums to 1.000.
- Stage boundaries. I believe 200 and 400, with a 700-step default.
- Torsion strain: the paper's equation divides by the number of rotatable
  bonds. Check whether `compute_torsion_strain_score` returns a sum or a mean.
- Novelty: the paper uses one minus the **maximum** Tanimoto similarity. Check
  whether the code uses the minimum.
- Diversity: the paper carries a term `D(L)` in the reward. Check whether any
  such term exists, or whether it is a per-batch duplicate penalty instead.
- Internal energy: the paper says MMFF94 on the unrelaxed conformation. Check
  which force field is used and whether the energy is measured before or after
  `optimize_molecule_geometry`.
- Which conformer the physical terms are scored on: the model's own coordinate
  output, or one re-embedded from the SMILES with ETKDG.
- Whether the physical reward terms are normalised to [0,1] or enter raw.
- The policy update: the paper describes PPO with a critic and an advantage.
  Check whether there is a value network anywhere, and whether the advantage is
  a reward minus a moving baseline.
- Whether positive advantages are up-weighted relative to negative ones in the
  loss, and by how much.
- The hard constraint gates in `violates_hard_constraints`: their thresholds per
  stage, and the direction of the synthetic accessibility gate. If the gate
  rejects molecules whose SA is *below* a floor, say so, and note that this
  rejects the easiest-to-synthesise molecules. That is deliberate, it keeps
  trivial fragments out, and it needs to be in the paper.
- Actual optimiser settings in `run_datapoint_generation`, including the
  learning rate actually used as opposed to the `--learning_rate` default.
- Docking: exhaustiveness, box size, timeout and the accepted score range.
- Replay: buffer size, how many are drawn, and the resulting fresh fraction.
- Archive size, the number of weight vectors, and the Das-Dennis divisions.

For each item write: what the paper says, what the code does, and which one you
recommend changing. Put it in `CODE_VS_PAPER.md`. Do not edit the `.tex` yet.

## Step 2. Code changes to make before running

Make these and nothing else. Keep each one small and obvious, and keep the
default behaviour identical to the released code unless the flag is set, so the
runs stay comparable.

1. **The curriculum fix** above, once I confirm.
2. **A `--curriculum` flag** with values `on`, `off`, `reverse`, `fixed3`.
   `off` holds stage 1 for the whole run, which is the no-curriculum arm.
   `reverse` runs the stages 3, 2, 1. Route the stage decision for every mode
   through one function so there is a single source of truth.
3. **A `--drop_terms` flag** taking a comma-separated list of reward term names
   that get weight zero, with the remaining weights renormalised to sum to one.
   Renormalise rather than leaving a hole, so the reward scale, the moving
   baseline and the clip bounds stay comparable across arms.
4. **A `--seed` flag** that seeds python, numpy and torch, and sets
   `cudnn.deterministic`. Note in the log that QuickVina has its own generator
   and is not seeded, so repeat runs agree on the molecules but not to the last
   decimal on the docking score.
5. **A `--targets` flag** accepting `0-15` or `0,3,7`, replacing the hardcoded
   `process_all_datapoints(range(0, 100))`.
6. **A `--no_skip` flag** that disables the step-25 early abandonment of a
   target. That abort is survivorship bias in an ablation: with it on, the arms
   get compared on different sets of targets, and the no-curriculum arm's
   collapse is exactly what gets thrown away. Set it for every ablation arm.
7. **A `metrics.csv` written one row per step** into each target's output
   folder. `process_training_step` already returns everything needed and
   `run_datapoint_generation` throws it away into underscores. Capture it.
   Columns: step, stage, wall_seconds, valid_ratio, n_valid, n_gated, n_unique,
   advantage_mean, advantage_std, reward_baseline, total_loss, policy_loss,
   entropy, entropy_coefficient, kl_to_prior, grad_norm, temperature,
   archive_size, and the batch mean of each of the seven scores plus the best
   docking score. Flush every row; a run that dies at step 300 must leave its
   first 300 rows behind. Also write a `run_config.json` per target with the
   seed, the flags and the resolved stage weights.
8. **Two speed fixes.** These decide whether the day is enough.
   - `calculate_scores` passes `score_cache={}` and `molecule_cache={}` fresh on
     every call, so nothing is reused between steps and the same molecule is
     re-docked every time it reappears. Make both caches module-level and keyed
     by SMILES plus receptor, with a size bound.
   - Docking is sequential over the batch, it is a subprocess, and it is the
     slowest part of a step. Add a `--dock_workers` flag and dock the unique
     SMILES of a batch in parallel into the cache, then let the existing
     per-molecule loop read the results. A thread pool is enough, since the work
     is in a subprocess. Temp file names are already hashed per SMILES, so
     parallel workers will not collide, but check that before you trust it.
     Verify on one target for 10 steps that 1 worker and 4 workers produce the
     same scores for the same molecules.
9. Two small things while you are there: `os.environ["CUDA_VISIBLE_DEVICES"]`
   is set inside `main()` after torch is already imported, so it has no effect;
   and the step log prints `Valid=.../16` while the batch size is 8. Fix both,
   and print the parameter count at startup.

Commit after Step 2 with the diff, or save the originals beside the edited
files. I want to be able to see exactly what changed.

## Step 3. Experiments, in this order

Measure seconds per step in the first 20 steps of the first run and tell me the
number before committing to the rest. Hours are (steps x targets x
seconds_per_step) / 3600, and at 20 s/step a 700-step target is about 4 hours.
If the projection does not fit the day, cut the number of targets first, then
the number of steps, and never the number of seeds.

**E0, no GPU, 10 minutes.** The `CODE_VS_PAPER.md` table from Step 1. This alone
closes every `TODO(VERIFY)` in the manuscript.

**E2 first, because it needs almost no GPU, and start it in the background.**
The TargetDiff authors released docked generated molecules for the same 100
CrossDocked targets and the same split this code loads:
`pocket2mol_vina_docked.pt`, `ar_vina_docked.pt`, `cvae_vina_docked.pt`,
`targetdiff_vina_docked.pt`, plus the reference set
`crossdocked_test_vina_docked.pt`. They are in the Drive folder linked from
https://github.com/guanjq/targetdiff under "sampling results". Download them,
inspect the structure first and print it before you write any parsing code, pull
out the SMILES per target, and push them through our **unmodified**
`scoring_function.py` on the same targets with the same docking settings.

That gives four baselines measured by our own pipeline rather than quoted from
four different papers, which is a much harder table to argue with. Write
`results/baseline_rescoring.csv` with one row per molecule and a summary table
by method. In the paper's table caption, mark clearly which rows we rescored and
which remain quoted from their source papers. Do not quietly mix them.

**E1, the main experiment, overnight.** Curriculum `on` against curriculum
`off`, 5 seeds each, the same 3 targets, 200 steps, `--no_skip` set on both.
30 target-runs. Report mean and standard deviation across seeds for valid ratio,
the seven scores, archive size and the number of gated molecules. This is the
experiment reviewer R1 asked for, and with the curriculum bug fixed it is also
the evidence that the curriculum does anything at all. If the two arms come out
the same, that is the finding, and I need to hear it rather than have it
smoothed over.

**E3, if E1 leaves time.** Leave out one physical term at a time, energy then
steric then torsion, 1 seed, the same 3 targets, 200 steps. Nine target-runs.
Only the three physical terms, not all seven: physical validity is the paper's
headline, so those are the three a referee will test.

**Optional, in this order, only if the day genuinely allows.**

- Independent rescoring: take the final archive molecules and score them with
  something that never entered the reward, QuickVina at exhaustiveness 16
  instead of 4 and PoseBusters. This answers "the reward grades its own
  homework". CPU only. The paper reports a PoseBusters pass rate: check how it
  was computed, and if it was not computed this way, this is the run that makes
  it real.
- Train and test overlap: how many PDBbind proteins used to fine-tune the
  bioactivity surrogate share a sequence with the CrossDocked test targets. The
  surrogate's split is random rather than by protein, so this number belongs in
  the paper either way. CPU, under an hour.
- A single 700-step run under the corrected objective, as a consistency check
  against what Remark 1 in the paper discloses.

Do not run any baseline generator from scratch, do not re-pretrain anything, and
do not expand an ablation past 3 targets. Pocket2Mol weights need a day of their
own; REINVENT's priors are downloadable from Zenodo but REINVENT has no pocket
input and would need its own per-target RL run; 3DMolFormer has no released
weights I could find. Released molecules are the right trade.

## Step 4. The paper

Only after the runs. In this order:

1. `CODE_VS_PAPER.md` and `PAPER_EDITS.md` for me to approve. Equations first:
   every one of the 27 equations should be checked against the function that
   implements it, and the ones that do not match listed with the code's actual
   form.
2. New results into tables and figures, each traceable to a CSV in `results/`.
3. Then the writeup. Scientific Reports wants Introduction, Results,
   Discussion, Methods in that order, no separate related work section, and an
   unstructured abstract under about 200 words. The current file is still in the
   AAAI order with a 270-word abstract. Propose the reorder as its own pass so
   the content edits stay reviewable on their own.
4. Conventions in that file, keep them: a `% [FIX R3-W1]` comment on each edit
   that answers a reviewer point; `% TODO(EXPERIMENT)` for anything not yet run,
   never a claim; simple words; no em dashes; `\caption` before `tabular*`;
   `\botrule` not `\bottomrule`; no `\resizebox` or `\adjustbox`, the class
   wraps tables in threeparttable and those break it; figures use `[H]`, tables
   cannot and use `[!ht]` with a `\FloatBarrier`. Rebuild with pdflatex and
   bibtex and report page count, overfull boxes and undefined references.
5. Never add a bibliography entry you have not verified against the real paper.
   No placeholder entries and no "to be verified" notes. A referee who checks
   one reference and finds nothing stops reading.

## What to report back to me

A single summary with: what the curriculum check found, the seconds per step and
what it let you fit, the E1 table with seed spreads, the E2 baseline table, the
list of equation mismatches with your recommendation for each, and anything that
failed. Say plainly which reviewer complaints are now answered with evidence and
which are still answered with an acknowledgement. I would rather have four
solid experiments and an honest limitations section than eight claims I cannot
defend when a referee opens the repository.

---
