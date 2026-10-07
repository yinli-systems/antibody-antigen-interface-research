# ABF-PairContact v1 research plan, revision 2

Status: implementation and evidence plan only; this file is not execution
authority and does not authorize test access.

## Updated decision after repository-wide evidence audit

The repository already contains a useful 3B Stage-A1 train-only precursor:
`abpairfm_stage_a1_geometry_ranking_canary_v17`.  With a frozen backbone and
80 head-only updates, its PDB-disjoint internal holdout changed from:

- macro sampled contact AP: `0.2141694` to `0.2812827` (`+0.0671133`);
- positive-over-negative pairwise accuracy: `0.4982547` to `0.5969855`
  (`+0.0987308`).

This is mechanism evidence, not downstream or blind-test evidence.  The v17
ranking objective compares every positive with every randomly sampled negative
inside a complex.  It does **not** separate unary residue propensity from
partner-specific interaction, does not require row/column-matched negatives,
and uses an auxiliary distance-bin target.  Therefore it supports continuing
with interaction ranking, but it does not resolve the documented marginal
shortcut.

## Minimal next intervention

Reuse the audited v17 frozen-backbone execution and holdout scaffold, replacing
only the contact objective with:

`z_ij = a_i + b_j + r_ij + c`,

where `a_i` and `b_j` are learned unary logits, `r_ij` is the bilinear
interaction logit, and `c` is a global bias.  Optimize:

1. class-balanced binary contact loss on `z_ij`;
2. within-complex ranking on `r_ij` against row-matched and column-matched
   noncontacts;
3. a row/column mean-square centering penalty on sampled `r_ij` values.

The canary must use binary contact labels only.  Full distance bins must not be
used, so any gain cannot be attributed to privileged distance regression.

## Correctness and efficiency gates before GPU execution

- exact deterministic negative sampling;
- no sampled negative may be a positive contact;
- row and column matches must be counted and reported separately;
- ranking must fail closed if no matched comparison exists;
- zero ranking/centering weights must reduce exactly to contact BCE;
- all head parameters must receive finite gradients;
- old Stage-A1 unit tests must remain green;
- the source, protocol, manifest, checkpoint, and environment must be hashed;
- checkpoint save/restore must reproduce the content hash;
- old SAbDab2 test, SABLE, fresh156, and temporal labels must remain closed.

## Execution ladder

1. CPU synthetic and regression gate (completed by the Phase-0 receipt).
2. One-GPU tiny graph/optimizer/checkpoint gate on a frozen train-only slice.
3. Four-GPU, 80-update, one-seed frozen-backbone head comparison against the
   exact v17 sampling/holdout scaffold.
4. Only if residual-ranking metrics improve, a fixed three-arm continuation
   screen: pair MLM, leak-free retrieval, and PairContact.
5. Only after the mechanism and downstream development gates pass, scale the
   frozen objective to a larger 3B token budget.

The pre-existing v17 internal holdout is already exposed and may only be used
as a development mechanism panel.  Final evidence still requires a future,
preregistered, post-cutoff antibody-and-antigen double-cold cohort.
