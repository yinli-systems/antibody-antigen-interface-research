# Frontier FSDP2 and DCP Engineering Protocol V1

Status: implemented locally; GPU evidence pending.

This protocol adds an independent FSDP2 and PyTorch Distributed Checkpoint
path for the existing R2 reranker. It is deliberately restricted to synthetic,
train-only systems evidence while Gate 0 remains unresolved. It cannot read an
outer fold, frozen test split, biological label, prediction, or model-selection
metric.

## Why this is a separate path

The existing DDP path has a proved fixed-world-size gradient formula and a
single-file immutable checkpoint. That proof cannot be assumed for FSDP2.
FSDP2 changes parameter layout, gradient communication, optimizer-state
layout, and checkpoint semantics. V1 therefore introduces new source,
runtime, launch, parity, and checkpoint receipts instead of modifying a
historical DDP receipt.

## Locked implementation

- Public PyTorch `torch.distributed.fsdp.fully_shard` is applied bottom-up to
  every bidirectional cross-attention block and then to the root reranker.
- The frozen D0 retriever is excluded from sharding.
- The AdamW optimizer is constructed only after sharding.
- FP32 eager is the first correctness mode. BF16 is allowed only after the
  corresponding FP32 run passes.
- `torch.compile`, FP8, asynchronous checkpointing, DeepSpeed, elastic process
  membership, and custom gradient hooks are separate future experiments.
- DCP saves sharded model and optimizer state. Each generation is staged in a
  unique directory, inventoried by exact filename, byte count, and SHA-256,
  published atomically, and made read-only.

The reduced fixture executes the real `ProtocolReranker` R2 code path with two
bidirectional cross-attention layers and the native residue-pair contact head.
It is small enough for a one-process full-global-batch reference. It validates
the objective and distributed mechanics, not full-model capacity or biological
quality.

## Numerical experiment

For a fixed global batch, each rank receives an equal rank-major slice. The
FSDP2 objective is the mean of equal local losses followed by FSDP2's gradient
average. Rank zero separately runs an unsharded model from identical initial
bytes over the full global batch with the same optimizer.

After a frozen number of steps:

1. compare the FSDP2 full model state with the one-process reference;
2. save an immutable sharded model-and-optimizer DCP generation;
3. advance the uninterrupted model by one step;
4. construct and shard a fresh model, restore the DCP generation, and run the
   same next step; and
5. require byte-exact continuation state and loss equality.

The default FP32 gate is `atol=2e-5`, `rtol=2e-4`. The restart gate is exactly
zero. Any non-finite value, missing key, unexpected key, source mismatch,
provenance mismatch, shard tamper, existing output, or rank-zero publication
failure terminates the run.

## Execution order

1. Local static checks and single-process DCP model/AdamW round-trip.
2. One-GPU FP32 reference, save, restore, and continuation.
3. Eight-GPU FP32 FSDP2 parity and same-world-size continuation.
4. Eight-GPU BF16 only after the FP32 gate passes.
5. A separately identified one-to-eight or eight-to-one DCP replay before any
   cross-world-size resharding claim.
6. At least 500 warmed steps before a throughput or efficiency claim.

The first GPU failures are evidence and must be retained. Thresholds, model
shape, optimizer, batch mapping, or source cannot be changed inside the same
attempt to convert a failure into a pass.

## Claim boundary

A passing run proves only synthetic R2 FSDP2 execution and checkpoint
continuation for the exact source, runtime, topology, precision, and world size
in its receipts. It does not prove biological training, partner specificity,
validation quality, test quality, 32/64-GPU scaling, production readiness, or
state of the art.

Authoritative machine-readable settings are in
`protocols/frontier_fsdp2_dcp_engineering_v1.json`.
