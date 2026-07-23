# Partner-specific retrieval protocol v1.1 amendment

Locked: 2026-07-23, before any AntiBERTy retrieval result was observed.

This amendment preserves protocol v1 as an immutable historical record. The SHA-256
of that record at amendment time is
`dc9027676c921de861e44308818d6498a3662ed8d7fe8cc64770e8c63fb52380`.

## Reason for the amendment

The exhaustive global-alignment audit showed that an MMseqs2 screen can miss
cross-boundary antibody homologs. A concatenated VH+VL comparison can also dilute a
near-identical single-chain match when the other chain differs. This is a split-validity
issue, not a model result, so it is resolved before any AntiBERTy retrieval score is
observed.

## Stronger development boundary

The preferred fixed development boundary must have zero train-validation crossings at
all four sequence views:

1. concatenated VH+VL antibody sequence;
2. VH sequence alone;
3. VL sequence alone, excluding genuinely missing light chains rather than aligning a
   placeholder; and
4. antigen sequence.

For every view, the primary crossing rule is global Needleman-Wunsch aligned-region
identity at least 80% and reciprocal coverage at least 80% for both sequences. Exact
overlap must also be zero. The CALM-style score, identical aligned residues divided by
the longer raw sequence, is reported separately for direct literature comparability.

If the four-view component graph cannot retain at least 2,500 development-training
instances and a nontrivial validation library, the concat-only split remains a
CALM-comparable sensitivity analysis, but no chain-level homology-clean claim is made.
The feasibility failure and connected-component sizes must be reported rather than
relaxing thresholds after seeing model performance.

## Model-selection firewall

- Architecture selection uses only folds derived from the original training authority.
- Validation and frozen test identifiers are sealed during inner-fold construction.
- Hyperparameters, stopping rules, and the winning-architecture rule are fixed before
  the new development holdout is scored.
- The fixed development holdout is evaluated once for the seed-42 gate in protocol v1.
- Seeds 17 and 73 are permitted only after seed 42 passes every preregistered gate.
- The frozen test remains unopened unless the three-seed development confirmation
  passes and a separate final-evaluation authorization is recorded.

## Additional robustness requirements

The final report must include:

- exact finite-library random expectations and 10,000 degree-preserving permutations;
- cluster-aware confidence intervals in addition to per-anchor bootstrap intervals;
- raw-embedding, random-head, length/composition, unary, and partner-permutation
  controls;
- metrics stratified by nearest training identity and by antibody/antigen family size;
- performance versus candidate-library size, with the complete library as primary;
- deterministic manifests, model-weight hashes, split hashes, per-anchor ranks, and
  checkpoint hashes; and
- an explicit caveat that frozen protein-language-model pretraining data cannot be
  fully audited from downstream artifacts.

## External validity ladder

SAbDab-derived positive-pair retrieval is necessary but cannot establish real binding
specificity because unobserved pairs are not verified non-binders. Broader claims
require a separate, versioned protocol using experimental positive and negative pairs,
starting with AVIDa-SARS-CoV-2 or AVIDa-hIL6. No architecture or threshold may be tuned
on that external test. A future structure-aware residue-pair model must be evaluated as
a distinct task rather than conflated with dual-encoder partner retrieval.
