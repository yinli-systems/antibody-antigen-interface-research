# Partner-specific retrieval protocol v1.2 amendment

Locked: 2026-07-23, before any AntiBERTy retrieval result was observed.

This amendment records an objective consequence of the v1.1 four-view split rule. It
does not replace or modify the historical v1 and v1.1 files. Their SHA-256 digests are:

- v1: `dc9027676c921de861e44308818d6498a3662ed8d7fe8cc64770e8c63fb52380`
- v1.1: `2e959dacb62a4af5139455c72638842321407c712b96785d1c56d70537c78a41`

## Observed split feasibility before model scoring

The concat-antibody-plus-antigen graph yields a fixed development split of 3,200 train
and 245 validation instances. Exhaustive independent verification has zero primary and
zero CALM-scaled crossings for concatenated VH+VL and antigen.

Adding separate VH and VL constraints produces a giant connected component containing
3,393 of 3,445 development instances. Therefore, only 52 instances can be held out
while preserving all four sequence constraints. This chain-strict validation contains
38 unique concatenated antibodies, 38 unique VH sequences, 33 unique antigens, but only
6 unique VL sequences; 45 of its 52 instances have no VL sequence.

## Two-boundary reporting rule

Neither boundary may be selected or discarded based on retrieval performance.

1. **CALM-comparable development estimate:** the 3,200/245 concat-antibody-plus-antigen
   split is the main quantitative development estimate because it is materially larger
   and matches the published full-sequence CALM comparison more closely.
2. **Four-view homology stress test:** the 3,393/52 concat+VH+VL+antigen split is a
   mandatory robustness analysis. It is not the sole headline benchmark because its
   validation set is small and heavily enriched for antibodies without light chains.

The two results must be presented together with native library sizes, exact chance
expectations, and their different claim boundaries. A strong development claim
requires the preregistered v1 gates on the CALM-comparable split and a positive effect
over the matched random/permutation null in both directions on the four-view stress
test. The small stress-test sample receives exact/permutation inference and confidence
intervals; failure or inconclusiveness must be reported and cannot be replaced by a
different threshold.

## Locked split artifacts

- CALM-comparable split SHA-256:
  `fdfeaf7554a441c69fd66b956874db3e16cb055c1aaa76875659de2f1c5f09d0`
- CALM-comparable independent verifier SHA-256:
  `80aef793487a5f1256f48d33ab01dfa040a085817b947d42022c958ee6031bbb`
- Four-view chain-strict split SHA-256:
  `5d92edd03e3b96edfa727823b2a5bbdf49e80614ef759fe0c6b14cf6c2b6b92b`
- Four-view independent verifier SHA-256:
  `f724e7c8cb99a7f79d76241df077539474977d2946a7c714c956dbc42d822c32`

The frozen SAbDab2 test remains sealed under both analyses.
