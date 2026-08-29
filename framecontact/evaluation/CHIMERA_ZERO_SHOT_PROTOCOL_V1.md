# Frozen CHIMERA zero-shot interface protocol

This protocol was frozen before this evaluation read CHIMERA row-level labels
or generated CHIMERA model scores. The machine-readable authority is
`chimera_zero_shot_protocol_v1.json`; its SHA-256 is written to a create-only
receipt before materialization starts.

## Question

Does same-chain spatial-neighbor routing improve zero-shot residue-interface
prediction over dense sequence self-attention when both models use the same
frozen ESM-2 inputs, training data, parameter count, epoch count, and seeds?

The primary comparison is the equal-weight seed-17/42/73 ensemble of fixed
epoch-20 `spatial_topology` checkpoints minus the corresponding
`sequence_dense` ensemble. No CHIMERA datum may select a checkpoint, mode,
threshold, ensemble weight, mapping rule, or exclusion rule.

## External data and scope

CHIMERA is pinned to revision
`54d5a2a1e59a329db7a6c09184652ed225188940`. The co-primary external axes are
`temporal.test` and `antigen_fold.test`. The primary endpoint is the paired
difference in mean per-complex epitope AUPRC. The two hypotheses use Holm
correction. Paratope metrics, pooled metrics, thresholded metrics, the
development-best checkpoints, and `epitope_group.test` are secondary or
exploratory exactly as specified in the JSON authority.

This is an auxiliary zero-shot epitope/paratope labelling evaluation on the
CHIMERA structures. It is not the benchmark's canonical epitope-conditioned CDR
sequence-structure co-design task and must not be described as a CHIMERA design
leaderboard result.

## Mapping and failure policy

Heavy and light chains remain distinct antibody chains. Every antigen chain
also remains distinct. Real N/CA/C coordinates define frames; residues with an
incomplete frame remain present with an explicit missing-frame mask. Provided
epitope/paratope annotations are cross-checked against contact pairs without
using model scores.

All mapping failures are quarantined with a deterministic reason before metrics
are computed. Confirmatory interpretation is forbidden below 95% usable
coverage on either primary split. There is no score-dependent filtering or
post-hoc length filtering.

## Statistics and reporting

Seed probabilities are averaged with equal weights. The paired group bootstrap
uses 10,000 replicates and seed 20260730, preferring an official antigen group
and otherwise using the normalized antigen-sequence SHA-256. Both candidates
and all seeds stay paired. A co-primary split supports the hypothesis only when
the paired 95% interval is above zero and the Holm-adjusted two-sided p-value is
below 0.05.

The report must include absolute metrics, paired effects, intervals, raw and
adjusted p-values, coverage, every exclusion, data and checkpoint digests, and
full versus SAbDab2-overlap-excluded analyses. Null or negative findings remain
reportable outcomes.
