# Prospective temporal antibody-and-antigen double-cold v2

This campaign begins after the ABF30B FrameContact model, six component
checkpoints, and task router were frozen. It watches future official SAbDab2/PDB
releases using metadata and sequences only. SABLE v0.2.0 and the failed temporal
v1 cohort are explicitly closed evidence and cannot be reused as blind tests.

The opening gate is cumulative and immutable: at least 36 sequence-double-cold
antibody instances from at least 30 unique PDB entries. The first official
weekly snapshot satisfying both minima is used. No threshold relaxation or
candidate cherry-picking is permitted.

`freeze_prospective_temporal_double_cold_v2.py` binds the frozen model and
protocol. The retained initial `screen_sabdab2_release_metadata_v2.py` treated
SAbDab2's `date` field as a release date; SAbDab2 documents that field as the
initial PDB deposition date. Because the initial screen contained zero
post-cutoff candidates and no labels were opened, v2.1 records a transparent
non-scientific authority correction rather than altering the original record.

Future screens use this chain:

1. `fetch_rcsb_release_metadata_v2_1.py` freezes the exact official RCSB
   `rcsb_accession_info.initial_release_date` for every SAbDab2 PDB ID.
2. `freeze_release_date_amendment_v2_1.py` binds the correction to the original
   protocol and zero-candidate initial screen.
3. `screen_sabdab2_rcsb_release_metadata_v2_1.py` uses SAbDab2 for antibody
   annotations and RCSB for release dates. It remains metadata-only and cannot
   authorize label opening.

The preferred final executor is an independent sealed steward following
`THIRD_PARTY_SEALED_EXECUTION_SPEC_V2.md`.
