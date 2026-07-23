# Corrected-boundary retrieval preflight

Status: computed before any model checkpoint was trained on or scored against the
corrected boundaries.

Date: 2026-07-23 CST

## Inputs

- Prepared instance table SHA-256:
  `a9131d52323bae40572b7de81c5e41353ccc39a73309b7d7043485350facb009`.
- Concatenated-antibody/antigen boundary SHA-256:
  `87a90b8dbee7ff2bd8b110797346238026e4c50c5796cdc9e5e80c383e676c21`.
- Four-view chain-aware boundary SHA-256:
  `2bf2f0ba0709a466bbe6f57d1ee042e8d2406eb557d9a71cb92fb30d8ce51344`.

This check used only instance identifiers and authenticated antibody/antigen sequence
digests. It did not load a model, checkpoint, embedding, prediction, or metric.

## Effective retrieval libraries

The common authority-training partition contains 3,229 emitted rows and 1,830 unique
antibody--antigen sequence pairs after exact-pair deduplication (1,383 unique antibodies
and 1,172 unique antigens). Both corrected boundaries use exactly this same training
partition.

| Boundary | Emitted validation rows | Unique sequence pairs | Duplicate pairs removed | Unique antibodies | Unique antigens | Maximum antibody degree | Maximum antigen degree |
|---|---:|---:|---:|---:|---:|---:|---:|
| Concatenated antibody + antigen | 128 | 80 | 48 | 66 | 66 | 3 | 6 |
| Concat + VH + VL + antigen | 43 | 36 | 7 | 30 | 26 | 3 | 6 |

## Exact finite-library null fixed before scoring

The exact random expectation accounts for each anchor's number of known positives; it
is not the single-positive harmonic approximation.

| Boundary | Direction | Candidate count | Anchor count | Exact random MRR | Exact random Recall@1 |
|---|---|---:|---:|---:|---:|
| Concatenated antibody + antigen | antibody to antigen | 66 | 66 | 0.081142 | 0.018365 |
| Concatenated antibody + antigen | antigen to antibody | 66 | 66 | 0.080733 | 0.018365 |
| Four-view chain-aware | antibody to antigen | 26 | 30 | 0.163324 | 0.046154 |
| Four-view chain-aware | antigen to antibody | 30 | 26 | 0.157472 | 0.046154 |

The bidirectional exact-random MRRs are 0.080938 and 0.160398, respectively.
These values were computed from boundary membership and the positive-degree graph
before any selected checkpoint score was generated or inspected.

## Interpretation

The complete native library remains the primary retrieval setting. Known positive
degrees must be respected by the exact finite-library null and by degree-preserving
permutations. The 30-by-26 chain-aware library has limited statistical power and is a
strict stress analysis; its point estimate must be accompanied by exact/permutation
inference and wide uncertainty intervals rather than treated as an independent claim.
