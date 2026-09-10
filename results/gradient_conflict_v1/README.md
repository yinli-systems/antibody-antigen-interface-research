# The two objectives agree; the relational one over-steps

Status: a direct gradient measurement at three trained checkpoints, eighty
validation complexes each. It rules one mechanism out and supports another.

Date: 2026-09-10

Machine-readable: [`seed1.json`](seed1.json), [`seed2.json`](seed2.json),
[`seed3.json`](seed3.json).

## Why it was measured

Down-weighting the pair terms improves **both** tasks
(`results/multitask_coupling_v1/`). At least five accounts fit that: gradient
conflict, a compromised shared representation, loss-scale imbalance, plain
optimisation interference, and the pair objective learning relations irrelevant
to node classification.

Gradient conflict is the one usually assumed and the only one that can be
measured directly. For each complex, the gradient of the node objective and the
gradient of the pair objective are taken with respect to the **shared** trunk —
863,494 parameters, excluding the 33,029 of the pair decoder and the 258 of the
node heads — and the cosine between them is reported.

## Result

| seed | cos(∇node, ∇pair) | sd | instances with cos < 0 | ‖∇node‖ | ‖∇pair‖ | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **+0.7000** | 0.1042 | **0%** | 3.376 | 5.024 | 1.49 |
| 2 | **+0.6982** | 0.1043 | **0%** | 3.665 | 5.495 | 1.50 |
| 3 | **+0.7307** | 0.1054 | **0%** | 3.915 | 5.373 | 1.37 |

**Not one of the 240 complexes shows a negative cosine.** The objectives pull
the shared trunk in substantially the same direction, and the pair gradient is
about half again as large.

## What it rules out, and what is left

**Gradient conflict is ruled out.** A cosine of +0.70 with no negative instances
is the opposite of objectives fighting. This was the assumed explanation and it
is wrong.

**Loss-scale imbalance is what survives**, and three independent measurements
agree on it:

1. the gradients are aligned — the tasks are not in tension;
2. the pair gradient is 1.4–1.5x the node gradient in norm at the default
   weighting;
3. reducing the pair weight to a quarter improves node AUPRC by 13.3% **and**
   pair AUPRC by 10.8% together.

Put in one sentence:

> The node and pair objectives agree about where the shared representation
> should go. The default weighting makes the relational term take too large a
> step in that shared direction, which costs both tasks.

That is a less intuitive result than objective conflict and a more checkable
one. It also predicts what the dose curve shows: an interior optimum, not a
trade-off frontier.

The remaining accounts are weakened by the same measurement. A compromised
shared representation would not produce objectives this aligned; a pair
objective learning irrelevant relations would not produce a cosine of 0.70.

## Limits

1. **Gradients at a trained checkpoint, not along the trajectory.** Alignment
   late in training does not prove alignment early, when the representation is
   being formed. A conflict present at initialisation and resolved by
   convergence would not show here.
2. Three seeds, one cohort, one architecture, at `λ = 1`. The measurement has
   not been repeated at the other dose points.
3. Cosine on the concatenated trunk gradient is a single scalar for 863,494
   parameters. Per-layer conflict that cancels in the aggregate would be
   invisible.
4. The node heads are excluded because the pair loss has no gradient with
   respect to them — the cosine there is undefined, which is itself a check that
   the parameter classification is right.

## Provenance

| | |
|---|---|
| script | `scripts/gradient_conflict.py` |
| checkpoints | `esm2_esmif1` seeds 1-3, AsEP `epitope_group` |
| instances | 80 validation complexes per seed |
| node objective | `epitope_bce`, `epitope_dice`, `paratope_bce`, `paratope_dice` at their trained weights |
| pair objective | `edge` at weight 1.0 |
