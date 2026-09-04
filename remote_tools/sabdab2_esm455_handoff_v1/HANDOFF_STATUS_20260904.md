# SAbDab2 ESM-2 test-455 handoff status

## Completed export

The authenticated canonical SAbDab2 `frozen_split=test` subset was exported at:

```text
/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/sabdab2_esm2_frozen_test455_v1
```

The directory is directly compatible with the existing sequence-SHA-addressed
embedding loader. It contains 455 instance mappings and 514 unique ESM-2
artifacts (536,892,184 logical bytes).

Verified identities:

- requested instance ID-set SHA-256:
  `9a42ea95391659b068dd1e1070290368a5f4c440e73a66e0a5ee2577c71efcd8`
- source embedding manifest SHA-256:
  `9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545`
- subset embedding manifest SHA-256:
  `67c2ee885771cba395e42809bad9c7b0e3d6d02a323b00059d295b365498758f`
- instance-to-embedding map SHA-256:
  `f95ab224044011cc103a876e088c942952de74f02a2541d92ea7ef131257dc6b`
- artifact tree SHA-256:
  `37d535648204e0d3b8cde291f6566af7cadf1ebaaa9acb9e983eaee3ce8e0e1b`

All artifact hashes, model identity fields, sequence lengths, tensor shapes,
residue-axis coverage, and finite-value checks passed. No structures, task
labels, model predictions, or metrics were read by the exporter.

## Confirmed requester identity and loader bundle

The requester confirmed that the canonical 455 instances are the intended test
set. The independently retained three-way split contains 2,727 training, 481
validation, and 455 test instances; its test ID set exactly matches the exported
mapping and requested-ID authority:

- split JSON SHA-256:
  `232f6d055d66ca15ffc04cb43b0895d322a3d7ce1f38d7fde517296633c220a2`
- test ID-set SHA-256:
  `9a42ea95391659b068dd1e1070290368a5f4c440e73a66e0a5ee2577c71efcd8`

The same remote directory now also contains:

- `SABDAB2_455_SPLIT.json`: the complete train/validation/test assignment;
- `SABDAB2_TEST455_INSTANCE_INDEX.parquet`: a label-free test index with
  instance IDs, Ab/Ag sequence digests, sequence lengths, and split role;
- `LOADER_PATHS.json`: relative paths for direct loader configuration;
- `README.md`: remote consumption instructions;
- `CACHE_BUNDLE_RECEIPT.json`: hashes and exact-set checks for the completed
  bundle.

The final bundle status is `PASS_MODEL_READY_TEST455_LOADER_BUNDLE`. All 514
embedding artifacts were loaded again with the production tensor schema and
passed digest, sequence-length, shape, full residue-axis coverage, and finite
value checks. Experimental coordinates and task labels are not included.

## Scope of the completed handoff

The completed component is the **embedding cache plus split/index**. The
`PASS_MODEL_READY_TEST455_LOADER_BUNDLE` status does not mean that a full
FrameContact inference run was executed. The production `EmbeddingStore` can
consume these tensor paths; `Sabdab2StructureEmbeddingDataset` additionally
needs separate structure and residue-label caches and accepts only train or
validation roles in its current constructor.

The requester's `README_test_split_request_v1.md` remains inaccessible at
`/data/ab02/yangwuyu/framecontact_case_study_20260904/test_split_request_v1/`.
Their confirmation resolves the ID-set question but does not establish the
required full cache schema. Copying that README to the shared ParaCloud handoff
directory is the remaining requirement for exact full-cache assembly. The
preceding chat described the cache as complete too broadly; the boundary above
is the corrected handoff status.

`verify_test455_embedding_loader_v1.py` performs a repeatable smoke check using
the exact `src/interaction_first/runtime.py` module, binding its SHA-256 and
loading both roles of every instance through `EmbeddingStore`. Its separate
receipt records that full model inference and requester schema validation have
not been performed.
