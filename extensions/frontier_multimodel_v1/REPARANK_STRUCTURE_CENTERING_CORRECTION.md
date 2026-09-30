# RePaRank structure-arm centering correction — 2026-09-30

## Discovery

The completed uncentered structure run (`1642234`) is preserved as a real negative result, but it is **not** the final official-preprocessing comparison. Its evaluator parsed exact N/CA/C/O coordinates and passed raw PDB xyz directly to the official sequence+structure checkpoint.

At official source commit `6910d950c2e483289973be41387c8913b859766a`, RePaRank validation/test uses `CenterMesh()` for every chain. `ComplexDataset.transform_example` applies the transform separately to each chain and `CenterMesh` returns `points - points.mean(dim=(0,1), keepdim=True)`. The official tutorial also centers coordinates before structure inference. RePaRank's `PositionEmbedding` directly encodes continuous xyz with sinusoidal functions, so omission of centering is scientifically material.

## Frozen correction

The corrected run changes exactly one scientific input transformation:

`backbone <- backbone - mean(backbone, axes=(residue, atom))`

independently for heavy, optional light, and antigen chains before the checkpoint forward pass.

Everything else remains frozen:

- official sequence+structure checkpoint SHA-256 `ce95a698e866246b647197f5b12e11d86c2392d462d2cb2a9ed63c57cf51ae0d`;
- official source commit `6910d950c2e483289973be41387c8913b859766a`;
- strict 217-case predicted-chain authority SHA-256 `bf47107325885a7f555767cf9f55f8d8074bdafaaae6956961ee2ea6408cc594`;
- the same 217 development-validation complexes / 68 split-authority components;
- the exact complete antibody-by-antigen candidate grid and frozen 4.5 Å binary contact labels;
- negative Euclidean distance between official complex embeddings as the threshold-free pair score;
- no test/OOD access, no checkpoint tuning, no threshold tuning.

This is a protocol correction, not a hyperparameter search. The uncentered result remains reported and is not overwritten.

## Interpretation boundary

Predicted structures are independent chain-local structures, not a predicted bound antibody-antigen pose. This is consistent with RePaRank's datamodule comment that training augmentation imitates inference with separate protein structures; however, official checkpoint training accession overlap with this development cohort remains unknown. No clean SOTA or OOD claim is authorised.
