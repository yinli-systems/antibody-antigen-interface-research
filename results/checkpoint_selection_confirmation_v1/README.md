# Checkpoint selection: stage two, and the `d_model` choice

Registered in `protocols/CHECKPOINT_SELECTION_CONFIRMATION_V1.md`. Read once,
on **group validation only**. The `epitope_group` test split was deliberately
not re-evaluated after the checkpoint-selection hypothesis was discovered, in
accordance with the pre-specified test-access protocol.

## What was replicated

A single-seed sweep had suggested that selecting the checkpoint on total loss
rather than on `partner_rank` changed which capacity was worth having. Ten
paired seeds at each of `d_model` 128 and 256, `--selection-metric total`,
everything else at the defaults.

The quantity is the one the protocol names: **mean per-complex epitope MCC**,
with the threshold chosen per seed on the same validation split.

| | |
|---|---|
| M(128) | 0.2986 |
| M(256) | 0.3265 |
| M(256) − M(128) | **+0.0278**, positive on 8/10 seeds |
| bootstrap 95% CI | [+0.0096, +0.0450] |
| decided | yes, the interval excludes zero |
| **d\*** | **256**, by argmax |

Under the rule, 256 is taken only because the interval excludes zero. Had it
contained zero the rule would have returned 128 whichever mean was larger.

## What these numbers are not

Both figures are **best-on-split**: the threshold is chosen on the split it is
then scored on. That is legitimate for choosing `d_model` -- the same freedom is
given to both arms and the comparison is paired -- and it is not a performance
number. It cannot be placed beside WALLE, PEPNet or any other published row.

The selected thresholds land at 0.78-0.91 on every seed, far from the 0.5 the
evaluation applies by default, which says the model is poorly calibrated for the
fixed convention rather than that it is unusually good at this task. The
distance between the two is roughly the whole difference between these figures
and the frozen group-test numbers (MCC 0.157).

## What follows

Stage three, already submitted: the `d_model` 256 configuration is trained on
`epitope_ratio`, ten seeds. Its threshold will be selected on **ratio
validation** and applied unchanged to ratio test, which is the second and final
registered reading of that split. The first reading, for the original frozen
configuration, gave MCC 0.2584 and is reported in `results/asep_benchmark_v1/`.

Producing the ratio-test predictions is not the reading. The reading is the
single aggregation that applies the validation-selected thresholds, and it runs
once when all ten seeds are on disk.
