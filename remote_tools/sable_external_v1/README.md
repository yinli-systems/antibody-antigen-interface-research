# SABLE v0.2.0 single-opening external evaluation

This directory contains the auditable launch and analysis code for a strictly
external FrameContact evaluation on SABLE v0.2.0. The evaluation is separated
into irreversible evidence stages so that labels cannot influence model,
checkpoint, routing, calibration, cohort, or metric choices.

## Frozen sequence

1. `screen_sable_v0_2_0_label_free_v1.py` defines the double-cold cohort using
   metadata and sequences only. It retained 97 entries from 88 PDB IDs.
2. `build_sable_v0_2_0_af3_inputs_v1.py` and
   `run_sable_v0_2_0_af3_array_v1.sbatch` predict antibody and antigen geometry
   independently. No antibody--antigen complex is predicted jointly.
3. `materialize_sable_v0_2_0_esm2_embeddings_v1.py` produces the frozen
   sequence representations without labels or bound structures.
4. `freeze_sable_v0_2_0_no_distance_checkpoint_binding_v1.py` binds the exact
   six development checkpoints trained without bound-distance auxiliary
   supervision.
5. `SABLE_V0_2_0_MEASUREMENT_SOURCE_MANIFEST_V1.json` SHA-binds the downloader,
   one-time label materializer, six-run evaluator, analyzer, and launchers
   before any bound structure is downloaded.
6. `freeze_sable_v0_2_0_bound_label_opening_v1.py` authorizes at most one label
   opening only after all earlier authorities pass.
7. The bound structures are downloaded from the pinned Zenodo record, contact
   labels are materialized once, the exact six checkpoints are evaluated
   without updates or calibration refitting, and the fixed seed-matched task
   router is analyzed once.

The first materialization process failed closed before any performance metric
was computed because five standard modified residue names were not normalized
and one scFv stored its declared heavy and light segments on the same physical
chain. Recovery v2 was explicitly authorized, SHA-frozen, and mapping-only
preflighted on all 97 entries. It adds only the audited parent-residue map and
maps repeated physical chain IDs once against their concatenated declared
segments; it neither uses wildcard matching nor excludes an entry. The
recovered mapping gate retained all 97 entries and 88 unique PDB IDs.

The immutable external execution plan has SHA256
`d97f0b94863fe9227d80c5675e944091809245b8fab22a89cda81148a8de172a`.
The frozen post-opening measurement source manifest has SHA256
`d67dd8580e4a0e9d4b039223cf2f583e6b48c71f21f33b49b08985e059f9f918`.

## Claim boundary

At inference, the model receives only ESM-2 residue representations and
independently sequence-predicted, chain-internal N/CA/C geometry. Experimental
bound structures are opened only after model and protocol freezing and are
used solely to derive evaluation labels. The old SAbDab2 test and temporal
holdout remain closed. No result from this campaign may be used for tuning or
model selection.
