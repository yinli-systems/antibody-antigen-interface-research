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

For the canonical frozen SAbDab2 test split, use `--split-column frozen_split
--split-role test`. When the requester provides its own ID manifest, supply
`--request-ids` and `--request-ids-sha256`; exact IDs then replace split-column
selection. The exporter refuses duplicate/missing IDs, unexpected counts,
noncanonical paths, manifest drift, symbolic sources, non-finite tensors, and
pre-existing outputs.
