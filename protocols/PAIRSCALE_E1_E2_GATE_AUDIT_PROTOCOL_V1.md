# PAIRScale E1/E2 Gate Audit Protocol V1

## Purpose and timing

This protocol was frozen after the real-cache GPU engineering smoke passed and
before any epoch-260 development-holdout result existed. It defines how the
three preregistered screening seeds may advance to a five-seed robustness
audit. It cannot authorize model scaling, checkpoint selection, a confirmatory
claim, or a state-of-the-art claim.

The endpoint-disjoint 192/64 assignment is reused development data. Each seed
must train to the fixed epoch-260 horizon without early stopping. The holdout
may be deserialized once per seed only after the epoch-260 checkpoint and its
receipt are immutable.

## Authenticated analysis unit

The auditor authenticates the assignment, assignment receipt, each final
report, each frozen checkpoint receipt, and the common source/data anchors.
All seed reports must contain the same ordered holdout instance inventory.
Connected antibody--antigen components are reconstructed from the assignment,
not accepted from a report.

For every instance, direction, and score source, the specificity statistic is
the cognate score minus the arithmetic mean of the four score-free decoys.
The four mandatory specificity cells are:

- antibody-to-antigen, raw residual;
- antibody-to-antigen, double-centered residual;
- antigen-to-antibody, raw residual; and
- antigen-to-antibody, double-centered residual.

The three-seed value for an instance is the arithmetic mean of its three seed
values. Confidence intervals use 10,000 paired connected-component bootstrap
draws with PCG64DXSM seed 42017. Seeds are repeated measurements, not
independent bootstrap clusters.

## Frozen three-to-five-seed screening rule

A candidate is eligible to add seeds 101 and 137 only if all conditions below
hold for the fixed three-seed reports:

1. all reports reached epoch 260 with no selection or early stopping;
2. every one of the four specificity cells has an authorized 95% component-
   bootstrap lower bound above zero;
3. at least two of three individual seed means are positive in every cell;
4. the mean per-complex pair-AUPRC improvement over the in-model additive unary
   baseline is at least +0.005 and no seed has a mean improvement below -0.010;
5. the minimum exhaustive R@1 enrichment across the three seeds is at least
   2.0 times the exact random baseline in both directions for both raw and
   centered residual scores; and
6. every required field and artifact agrees across the independent audits.

Failure is retained as evidence and starts a new numbered repair protocol. It
does not permit threshold relaxation, seed replacement, or adding more seeds
until a positive result appears.

## Final Gate S, C, and R boundaries

The screening rule is not Gate S. Gate S still requires five fixed seeds, at
least four positive seeds in every specificity cell, raw and centered
component-bootstrap lower bounds above zero in both directions, and R@1
enrichment of at least 3.0 on a separately authenticated homolog-matched
candidate pool. Exhaustive or length-matched retrieval cannot be relabeled as
homolog-matched evidence.

Gate C requires at least +0.010 mean macro pair AUPRC over the strongest
separable baseline, an authorized paired component-bootstrap lower bound above
zero, and directionally positive P@K and AP-Lift differences. If a training
report lacks separable-baseline P@K or AP-Lift, Gate C remains pending until an
independent frozen-checkpoint audit produces them.

Gate R requires five seeds, the component-aware paired bootstrap, a fixed
component sign-permutation test, and all prespecified protein/peptide, antigen
length, interface size, multichain, and identity-novelty strata. Missing
lineage or novelty metadata is reported as unavailable and cannot silently
pass the gate.

Scaling remains blocked until Gates S, C, and R all pass under their complete
contracts.
