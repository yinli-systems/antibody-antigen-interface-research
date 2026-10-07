# Test-455 embedding handoff: consumption and limits

Use this directory as the sequence-SHA-addressed **embedding root**. It contains
455 instance mappings and 514 unique ESM-2 sequence artifacts, a full split JSON,
and a label-free test index. All paths in `LOADER_PATHS.json` are relative to
this directory; they describe the embedding/index component.

The production API is `interaction_first.runtime.EmbeddingStore(root,
max_cache_gib).load(sequence_sha256, device)`, returning `(chunks, starts,
sequence_length)`. The tensors retain their original chunk boundaries and
1280-dimensional residue representations. Use the existing model's chunk
handling rather than concatenating overlapping windows.

`EMBEDDING_LOADER_VERIFICATION_RECEIPT.json` records 910 Ab/Ag load calls and
checks the split, map, index, sequence lengths and keys, and every artifact SHA.

**Full inference cache assembly is still pending the requester's schema.**
No geometry, masks derived from geometry, residue/contact labels, or model
checkpoint is included here. `Sabdab2StructureEmbeddingDataset` requires
additional inputs and is not an independent test loader for this directory.
The earlier `PASS_MODEL_READY_TEST455_LOADER_BUNDLE` status refers to the
embedding/index layer, not a completed end-to-end inference run.

The requester confirmed the 455 IDs, whose sorted newline-delimited SHA-256 is
`9a42ea95391659b068dd1e1070290368a5f4c440e73a66e0a5ee2577c71efcd8`.
Their full-cache README at `/data/ab02/yangwuyu/framecontact_case_study_20260904/test_split_request_v1/`
is not mounted on ParaCloud. Make that README available beside this bundle to
specify the remaining fields and container format; the ESM embeddings can be
reused without recomputation.
