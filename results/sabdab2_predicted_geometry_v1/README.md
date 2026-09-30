# SAbDab2 sequence-predicted-geometry development sensitivity

This compact snapshot reports the frozen six-run, three-seed development
sensitivity analysis in which sequence-predicted backbone geometry replaces
experimental bound-complex coordinates at model input and inference.

The routed median contact, epitope, and paratope mean-per-complex average
precision values are `0.026726`, `0.243782`, and `0.688472`. Relative to the
frozen bound-geometry development reference, the paired median changes are
`-0.001548`, `+0.006660`, and `-0.023270`, respectively. The epitope changes by
seed are `+0.007911`, `+0.006660`, and `-0.013012`, so this is a mixed
development sensitivity result rather than an independent performance claim.

## Evidence boundary

- The 2,565/549 development split is unchanged, and the 549-complex validation
  set is checkpoint-selection data rather than independent confirmation.
- Experimental bound geometry is not used as model input and is not required
  at inference.
- Experimental bound geometry is still used for auxiliary distance
  supervision during training. The experiment therefore does not establish a
  fully structure-free training pipeline.
- No old exposed SAbDab2 test or temporal holdout was loaded, and the
  comparative result was not used for downstream protocol tuning or promotion.
- Checkpoints, prediction structures, logs, pair-level outputs, and internal
  filesystem paths are intentionally excluded from this public snapshot.

The JSON files preserve the exact metrics, seeds, artifact hashes, selected
epochs, and audit booleans. The `source_analysis_sha256` and
`source_receipt_sha256` fields bind this sanitized public snapshot to the
immutable internal analyzer outputs.

`PREDICTED_GEOMETRY_TASK_ASSOCIATION_PUBLIC_V1.json` adds a paired,
per-complex analysis of predicted- versus bound-geometry checkpoints. It
reports PDB-cluster bootstrap intervals and all predeclared geometry-quality
correlates. None of the simple RMSD, graph-overlap, or AF3-score associations
supports a causal mechanism claim; the file is development-only sensitivity
evidence.

## Bilateral Top-K evidence

`BILATERAL_TOPK_READOUT_DEVELOPMENT_PUBLIC_V1.*` records the earlier
fixed-checkpoint readout sensitivity. The separately frozen
`BILATERAL_TOPK_TRAINING_ABLATION_DEVELOPMENT_PUBLIC_V1.*` records eight
from-scratch seed-17 runs for epitope and paratope `K in {1,2,4,8}`. The
from-scratch analyzer passed all integrity checks. Epitope K2 is the most
balanced arm, while paratope K1 and K2 are promising relative to the native
reference. Because this is a single-seed experiment on checkpoint-selection
validation data, it supports an Appendix ablation rather than a universal
optimal-K or independent test claim.
