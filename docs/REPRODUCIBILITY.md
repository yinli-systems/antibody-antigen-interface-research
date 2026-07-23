# Reproducibility contract

Every formal result should be reconstructable from an immutable tuple containing:

- source revision and source-tree digest;
- dataset manifest and split-membership digest;
- model and optimizer configuration;
- random seed and complete RNG state when training is resumable;
- checkpoint digest and selection rule;
- raw per-instance scores and targets;
- metric implementation and aggregation unit;
- cluster-aware uncertainty estimates;
- control definitions and failure thresholds.

Intermediate jobs must not inspect the final test set. Model selection, threshold
selection, and ensemble weighting must be frozen using development data before the
final evaluation is launched.

Results from reused benchmarks are labeled as benchmark evidence, not pristine
confirmatory evidence. Negative controls and null results are preserved.
