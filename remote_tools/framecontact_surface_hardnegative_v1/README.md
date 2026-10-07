# FrameContact surface-prior and hard-negative development study

This directory freezes a development-only comparison of two targeted changes to
the no-privileged-distance, sequence-predicted-geometry FrameContact recipe:

1. `surface`: a zero-initialized antigen surface prior computed exclusively from
   within-chain predicted C-alpha density;
2. `hard_negative`: a per-complex ranking loss over the highest-scoring negative
   contact pairs and lowest-scoring positive pairs;
3. `combined`: both changes together.

Each intervention is trained from scratch for the joint baseline checkpoint and
the TopK2 epitope expert at seeds 17, 29, and 43. The original no-distance runs
are reused as the exact comparator and are not rerun.

## Evidence boundary

- Train/validation only: 2,565/549 SAbDab2 development complexes.
- Model geometry comes only from independently sequence-predicted structures.
- No experimental bound coordinates or bound-derived distance auxiliary labels
  are model inputs or training targets. Experimental structures define the
  unchanged contact labels only.
- The old SAbDab2 test and temporal holdout are not loaded.
- SABLE v0.2.0 had already been opened before this study was designed. It is
  permanently measurement-only and is not read, rerun, recalibrated, or used for
  checkpoint selection here. Any selected development arm requires a new,
  prospectively frozen external cohort for confirmation.

## Frozen hyperparameters

- surface radius: 10 Angstrom;
- hard-negative fraction: top 1% of valid negatives per complex;
- maximum selected positives/negatives: 64 each;
- pairwise logit margin: 0.5;
- hard-negative loss weight: 0.25 when enabled;
- all other data, architecture, optimization, early stopping, and routing rules
  match the no-privileged-distance reference.

## Execution order

1. Apply the three patches for each source tree and build authenticated source
   manifests.
2. Run `freeze_surface_hardnegative_v1.py` before any GPU work.
3. Run the two-task GPU canary.
4. Submit the 18-task formal array with an `afterok` dependency on the canary.
5. Run the analyzer only after all 18 tasks complete successfully.

The formal analyzer reports all seed-level metrics and deltas. A development arm
can be marked eligible for future confirmation only if it satisfies a
pre-declared contact- or epitope-improvement gate while respecting the paratope
guardrail. It is never promoted as independently confirmed evidence.

## Systems recovery

The first formal array (`1493976`) failed before any training update because the
generated protocol receipt used `protocol_sha256`, while the frozen trainer
requires `protocol_file_sha256`. Both CUDA canary tasks passed. The failed array
is retained as fail-closed evidence and no result is reused. The corrected
receipt-only recovery uses a new campaign root and reuses only the completed
2/2 canary binding; model code, data, hyperparameters, and scientific protocol
are unchanged.
