# OAS Production Materialization V2

## Purpose

Protocol v2 preserves every authenticated source, filtering, deduplication,
split, test-access, and authorization rule from v1. It changes only the
execution envelope required by the ParaCloud scheduler: a GPU partition job
must reserve one GPU to obtain CPU resources, even though corpus
materialization is CPU-only.

The reserved GPU is not a throughput claim and is not part of the computation.
The batch supervisor, mapper, reducer, finalizer, and independent replay all
fail unless exactly one scheduler GPU is assigned, `CUDA_VISIBLE_DEVICES` is
empty, and the v2 resource policy forbids GPU compute. Each terminal receipt
records that boundary.

## Execution topology

- One scheduler allocation reserves one idle GPU and eight CPU cores.
- One create-only supervisor runs eight logical tasks concurrently.
- Mapping keeps the frozen 32-way authority assignment.
- Reduction keeps all 256 digest buckets and exact validation precedence.
- Existing complete task receipts may be authenticated and skipped; partial or
  malformed task publications fail closed.
- Map and reduce stages remain separate. Reduction is not submitted until all
  32 mapper receipts have passed inventory validation.

This topology avoids reserving 32 or 256 GPUs for CPU-only work while retaining
the original logical partitioning. It does not change the corpus definition.

## Release boundary

Successful mapping and reduction do not authorize training. The finalized
corpus must still pass the independent full-replay auditor and the one-GPU
loader/checkpoint gate. No downstream held-out data is accessed by this stage.

The independent replay is implemented without importing the mapper, reducer,
record-layout module, or tokenizer. It decodes all fixed-width records,
reconstructs exact validation precedence and minimum-ordinal selection,
retokenizes every selected sequence, compares every NumPy value, and verifies
the complete mapper, reducer, shard-manifest, corpus-manifest, source, and
resource-receipt graph. Its create-only pass receipt remains non-authorizing;
only a later one-GPU loader and cross-world-size checkpoint/recovery gate can
authorize the corpus for training.

## Local evidence

The targeted v1/v2 mapper, reducer, finalizer, supervisor, independent replay,
protocol, and Slurm tests pass under two Python hash seeds. Full repository
collection requires the optional local `torch`, `pyarrow`, and `ortools`
dependencies and is therefore a separate environment-level check rather than
evidence produced by this unit.
