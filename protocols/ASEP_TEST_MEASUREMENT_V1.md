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

---

## Amendment, 2026-09-10: the published convention is now known

Registered **after** the test measurement was submitted and **before** its
results were read. The amendment is driven by what the published papers state
about their own method, not by anything in this project's numbers.

Two facts were located in the literature after this protocol was first written:

**WALLE's epitope-group figures** (Table 2(b) of arXiv 2608.01092):

| | MCC | precision | recall | AUROC | F1 |
|---|---:|---:|---:|---:|---:|
| WALLE | 0.077 ± 0.015 | 0.143 ± 0.017 | 0.266 ± 0.025 | 0.544 ± 0.010 | 0.145 ± 0.014 |

**Their convention**, stated in the same paper: threshold-dependent metrics use
"a probability threshold selected on the validation set to maximize MCC", and
the metrics are pooled over residues rather than averaged per complex.

Both differ from what this protocol assumed:

1. **The comparable quantity is micro, not macro.** Every epitope-node figure
   recorded so far in `results/asep_benchmark_v1/` is a per-complex average.
   Those are not comparable to 0.077 and must not be placed beside it.
2. **The comparable operating point is a validation-selected MCC-maximising
   threshold**, applied unchanged to test. This project's sweep already computes
   a "best" point, but takes it on the split being measured, which would
   overstate: choosing a threshold on test and reporting the metric it maximises
   selects on the quantity being estimated.

The measurement therefore adds one operating point, defined entirely by the
external convention:

> **matched**: the threshold that maximises micro MCC on the *validation* split
> of the frozen arm, applied unchanged to the test split.

This is the only point at which a comparison to the published number is
attempted. The other three -- fixed 0.5, best-on-test, prevalence-matched --
remain reported as before, and the first two bracket how much the convention is
worth.

Nothing else changes. The arm is already frozen, the test measurement is already
submitted, and this amendment cannot alter either.
