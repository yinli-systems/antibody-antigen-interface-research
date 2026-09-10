# AsEP test measurement

Registered 2026-09-10, **before the `esm2_esmif1` arm finished training** and
therefore before its validation numbers existed. The `esm2` arm's validation
numbers were already known and are recorded in
`results/asep_benchmark_v1/`.

## What is being decided

One configuration gets one measurement on AsEP's `epitope_group` test split.

The test split is the benchmark's, not this project's. Reading it is a
measurement, not development. It is worth exactly one reading, and a second
would make the first meaningless, because a number chosen from two readings is
selected on the thing it is supposed to estimate.

## The rule for choosing the configuration

**The arm with the higher mean epitope-node macro MCC on validation, over its
ten seeds, is frozen.** No other criterion, no tie-breaking on a second metric,
no inspection of test-adjacent quantities.

If the two arms' validation intervals overlap, the higher mean still wins. A
coin-flip dressed as a decision rule is still a decision rule, and picking the
simpler arm on a tie would be a preference about model size smuggled in as
statistics.

The frozen configuration is: the arm, its ten seeds, its twelve-epoch budget,
its `partner_rank` checkpoint selection, and the split manifest digest already
recorded. Nothing is tuned afterwards.

## What is measured, and at which operating point

Reported on test, for all ten seeds of the frozen arm:

| quantity | why |
|---|---|
| epitope-node macro MCC at fixed 0.5 | the convention the pipeline already uses |
| epitope-node micro MCC at fixed 0.5, best, and prevalence-matched | the operating point is the largest known uncertainty in any comparison |
| epitope-node AUPRC and AUROC | threshold-free, so comparable without agreeing on an operating point |
| pair-level macro AUPRC | this project's own endpoint, reported for continuity and **not** compared to published work |

Threshold-free metrics are the ones a comparison should lean on. AUPRC and AUROC
do not depend on where anyone put a cut, and the sweep already showed a 1.33x
span in MCC across reasonable operating points on this model alone.

## What will not be claimed

The measurement establishes a number on AsEP's test split for this
configuration. It does not establish a ranking against WALLE or EpiFormer
unless their operating point and metric definition are matched, which has not
been done and is not part of this measurement.

If the test number is worse than validation, that is the recorded result. A
validation-to-test drop is the normal behaviour of a model selected on
validation, and this project selects checkpoints on validation.

## Extraction

The frozen arm needs ESM-IF1 features for the test instances if it is the
`esm2_esmif1` arm. Those are extracted as part of this measurement, counted, and
recorded in the receipt — frozen unsupervised features carry no label, but
parsing a benchmark's test structures is an event worth counting.
