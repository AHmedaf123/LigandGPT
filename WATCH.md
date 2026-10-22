# What to watch while a run is going, and what it means

The run should write one row per step to `metrics.csv` in each target's output
folder. These are the columns worth your attention, in the order you should look
at them.

## The four that tell you to stop the run

**`valid_ratio`** Fraction of the 8 generated strings that parse into a real
molecule. Healthy: 0.5 and up, and not falling. If it sits near 0 for more than
20 steps, the policy has collapsed and the rest of the run is wasted. The code
retries with a higher temperature at step 0 only, so a collapse later goes
unnoticed unless you watch this.

**`n_unique`** Distinct SMILES in the step. If this drops to 1 or 2 and stays
there, the policy found one molecule and is repeating it. That is mode collapse,
and `advantage_mean` can still look fine while it happens, which is why you need
both.

**`grad_norm`** Gradient size before clipping. Spikes over 10 mean the update is
fighting itself. Occasional spikes are normal, a run that spikes every step is
not.

**`n_gated`** Molecules that scored but were thrown out by a hard constraint
(reward -1). A jump here right after a stage boundary, step 200 or step 400,
means the new stage's gates are stricter than what the policy can currently
make. Expect a bump, then a recovery. A bump that never recovers is the failure
mode to report.

## The ones that tell you it is working

**`stage`** Should read 1 before step 200, 2 from 200 to 400, 3 from 400 on.
**If it stays 1 for the whole run, the curriculum is not active** and you must
stop and say so. This is the first thing to check in the first minute.

**`advantage_mean`** Average reward minus the moving baseline. It is a moving
target by construction, so do not expect it to climb forever. What you want is
positive on average and not trending down.

**`entropy`** How undecided the policy is about the next token. It should fall
slowly. A cliff means early convergence, a flat high value means nothing is
being learned.

**`kl_to_prior`** How far the policy has drifted from the pretrained model.
Growing steadily is the point of the run. Growing fast while `valid_ratio` falls
means it is drifting into nonsense.

**`archive_size`** Molecules on the Pareto archive, capped at 100. Should climb
early then settle. Stuck at a low number means almost nothing is passing the
gates.

## The chemistry columns

`docking_mean` and `docking_best` in kcal/mol, lower is better, below -8 is
good. `qed_mean` 0 to 1, higher is better, over 0.6 is drug-like.
`sa_mean` 1 to 10, **lower is easier to synthesise**, and note the hard gate
rejects molecules *below* the stage floor, so very low values disappear by
design. `steric_mean` and `torsion_mean` and `energy_mean` are all
lower-is-better strain measures.

Per stage, this is what should move:

| Stage | Steps | What should improve |
|---|---|---|
| 1 | 0 to 200 | `docking_mean`, `bioactivity_mean` |
| 2 | 200 to 400 | `qed_mean`, `sa_mean`, while docking holds |
| 3 | 400 on | `energy_mean`, `steric_mean`, `torsion_mean`, while the rest hold |

If stage 2 wrecks the docking it gained in stage 1, that is the gradient
conflict the paper discusses, and it is worth a figure rather than a worry.

## First five minutes of any run, checklist

1. `stage` moves at 200 and 400. If not, stop.
2. Parameter count printed at startup matches what the paper says.
3. `wall_seconds` divided by `step` gives your seconds per step. Multiply out
   before you commit to the full run.
4. `valid_ratio` above 0.3 by step 10.
5. `n_gated` is not 8 out of 8.
