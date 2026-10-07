# SAbDab2 ESM-2 455-sample handoff

This exporter creates a model-ready, hash-addressed subset of the authenticated
SAbDab2 ESM-2 cache. It preserves the path convention consumed by
`Sabdab2StructureEmbeddingDataset`:

```text
<embedding-root>/<first-two-sequence-sha256>/<sequence-sha256>.pt
```

The output also contains:

- `INSTANCE_EMBEDDING_MAP.csv`: exact instance-to-antibody/antigen tensor map;
- `REQUESTED_INSTANCE_IDS.txt`: sorted exported instance authority;
- `SABDAB2_ESM_EMBEDDING_MANIFEST.json`: subset artifact manifest;
- `EXPORT_RECEIPT.json`: count, source, SHA-256, tensor-schema, length, shape,
  coverage, and finite-value checks.

After an exact 455-sample split authority is available,
`finalize_test455_loader_bundle_v1.py` adds the split JSON, a label-free Parquet
instance index, relative loader paths, and a second receipt. It verifies that
the split test IDs, requested IDs, instance map, and prepared-instance index are
identical before writing anything. The resulting directory can therefore be
passed directly as the embedding root while `SABDAB2_TEST455_INSTANCE_INDEX.parquet`
provides the ordered test-instance metadata.

For the canonical frozen SAbDab2 test split, use `--split-column frozen_split
--split-role test`. When the requester provides its own ID manifest, supply
`--request-ids` and `--request-ids-sha256`; exact IDs then replace split-column
selection. The exporter refuses duplicate/missing IDs, unexpected counts,
noncanonical paths, manifest drift, symbolic sources, non-finite tensors, and
pre-existing outputs.

The finalized ParaCloud bundle is located at:

```text
/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/sabdab2_esm2_frozen_test455_v1
```

Its embedding/index status and provenance are recorded in `CACHE_BUNDLE_RECEIPT.json`.
The separate `EMBEDDING_LOADER_VERIFICATION_RECEIPT.json` binds a production
`EmbeddingStore` smoke check for all 455 Ab/Ag pairs. This package supplies only
the embedding input component; a full model also needs its geometry inputs and
evaluation labels. See `HANDOFF_STATUS_20260904.md` for the remaining requester
cache-schema requirement.
