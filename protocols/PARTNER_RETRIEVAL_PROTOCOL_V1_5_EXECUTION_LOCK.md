# Partner-specific retrieval protocol v1.5 — execution lock

Status: locked before final refit and before any selected AntiBERTy checkpoint is
scored on either corrected boundary.

Date: 2026-07-23 CST

## Purpose

Protocol v1.4 replaced the contaminated v1.3 boundaries with two subsets derived
strictly inside authority validation. This amendment fixes the executable consequence
of that correction and removes optional stopping, duplicate refits, and analyst-chosen
randomness from the confirmation stage.

## Immutable inputs

- Authority split SHA-256:
  `e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b`.
- Concatenated-antibody/antigen boundary SHA-256:
  `87a90b8dbee7ff2bd8b110797346238026e4c50c5796cdc9e5e80c383e676c21`.
- Four-view chain-aware boundary SHA-256:
  `2bf2f0ba0709a466bbe6f57d1ee042e8d2406eb557d9a71cb92fb30d8ce51344`.
- Independent boundary verification SHA-256:
  `3bd8186603d908a99e1fbf00ead3697dd276ccd798b4f91bcbdda81441bd80e8`.
- Common authority-train membership SHA-256:
  `442095581ae9bf3c7440c1a818612298b5f81e5a62ef46f431e0867b9c8b466b`.

The two corrected boundaries contain exactly the same 3,229 training identifiers.
Their 128/88 and 43/173 validation/excluded-validation partitions differ, but neither
excluded partition may enter training or evaluation.

## Single-refit rule

Exactly one seed-42 final checkpoint is fitted on the common authority-train
membership. It uses the architecture selected by the train-only fold screen and the
previously fixed optimization contract:

- AntiBERTy antibody encoder and ESM-2 antigen encoder;
- strict projection-before-length-mean pooling;
- 128-dimensional shared representation;
- batch size 1,024;
- AdamW, learning rate 0.001, weight decay 0.2, betas 0.9/0.98;
- cosine warm restarts with 10-epoch initial period and multiplier 2;
- deterministic seed 42; and
- exactly 58 epochs, with no validation load, metric, early stopping, or checkpoint
  selection.

The concatenated-antibody/antigen manifest is the canonical train-materialization
manifest only because a concrete authenticated file is required. The checkpoint must
bind both corrected boundaries and their common train-membership hash. Training a
second checkpoint for the chain-aware boundary is forbidden; the same final checkpoint
must be used for both evaluations.

## Mandatory evaluation order

Both boundary evaluations are predeclared and mandatory after a successful refit.
The second evaluation is not conditional on the first score or gate. Each boundary is
scored exactly once through a central O_EXCL ledger whose identity includes the final
checkpoint SHA-256, boundary SHA-256, authority SHA-256, and literal validation split.

Before holdout materialization, the checkpoint must verify that the requested boundary,
constraint mode, evaluation path, score-artifact path, and central registry directory
all exactly match the immutable submission manifest. A failed or interrupted attempt
leaves its ledger consumed and cannot be retried under a different output path.

## Locked statistical analysis

For each boundary, the authenticated full score matrix is analyzed once with:

- 10,000 degree-preserving independent permutations;
- 10,000 anchor and joint bipartite-component bootstrap replicates;
- random seed 42;
- complete native candidate libraries as the primary setting;
- candidate-library sizes 10, 25, 50, and 100 with 1,000 deterministic samples as
  secondary descriptive curves, omitting sizes larger than the native library; and
- plus-one Monte Carlo p-values and 95% intervals.

No alternate seed, permutation count, bootstrap count, candidate-size curve, or
subgroup can replace the primary result after scores are visible. Additional analyses
must be labeled post hoc.

## Mandatory falsification controls

One control bundle is fitted on the same common authority-train membership and then
evaluated once on each predeclared boundary. It contains all five locked controls:

1. shared raw ESM-2 mean-pooled cosine similarity;
2. an untrained CALM-shaped random projection head;
3. antibody-only unary scoring;
4. antigen-only unary scoring; and
5. train-only length/amino-acid-composition logistic scoring.

Control evaluation uses the same one-shot and statistical contracts. Missing controls
cannot be treated as neutral evidence.

## Claim boundary

These two subsets remain reused development confirmation because an earlier ESM2
exploration inspected their source authority-validation pool. Passing every gate can
support a rigorous internal robustness result only. It cannot support an independent,
external, prospective, temporal, clinical, or world-leading claim. The frozen SAbDab2
test remains closed; external experimental confirmation follows the separately locked
AVIDa protocol.
