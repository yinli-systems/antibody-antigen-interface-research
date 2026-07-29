# Antibody DDP scientific parity gate v2

**Frozen before execution:** 2026-07-29 Asia/Shanghai

**Scope:** eight-GPU systems promotion on the authenticated dense OAS pilot.
This protocol cannot authorize production pretraining, downstream biological
claims, or scale beyond eight GPUs.

## Why v2 is a new protocol

The immutable v1 paired gate failed its full-500-update trajectory thresholds.
Its counterbalanced runs were exactly reproducible within each mode, and every
mode passed the first 20 updates, but compiler and BF16-reduction rounding
differences accumulated into different optimization trajectories. The v1
result remains failed and is not reinterpreted.

PyTorch documents that floating-point addition and multiplication are not
associative and that mathematically identical computations are not guaranteed
to be bitwise identical. PyTorch's published FSDP plus `torch.compile` parity
study compared long training curves and independent benchmark outcomes rather
than demanding identical values at every update. The built-in BF16 DDP hook
also intentionally reduces gradients at lower precision and is documented as
experimental.

V2 therefore keeps the strict 20-update numerical gate and adds a harder,
pre-registered outcome gate: three seeds, a training-excluded MLM panel, paired
per-sequence non-inferiority intervals, learning checks, and the already frozen
efficiency and rank-parity checks.

## Data boundary

The dense pilot contains 8,112 authenticated, exactly deduplicated sequences.
The 812 lowest namespace-bound sample-key digests form the evaluation panel.
The other 7,300 records form the only legal training universe. The split
receipt must bind every original array hash, both ordered index arrays, both
sample-key inventories, and the split implementation.

The dense pilot does not retain a row-level study or subject identifier. The
panel can detect systems-induced degradation on unseen pilot records, but it
cannot establish study-independent antibody generalization. That limitation is
part of the protocol and cannot be removed by a passing result.

## Experimental design

Seeds 17, 42, and 73 each compare safe eager, FP32 compiled, and the selected
BF16 communication candidate. The nine-run order is counterbalanced. Every run
uses the same 500-update training membership for its seed, an isolated compile
cache, deterministic algorithms, exact eight-rank state parity, and per-update
gradient-rank parity.

The evaluation panel is scored before training and after update 500 with four
fixed counter-derived MLM masks. Per-sequence cross entropy and masked accuracy
are retained. Candidate comparisons are paired by sample key after averaging
mask streams and seeds.

The gate requires all of the following:

- all six candidate/reference traces pass the inherited first-20-update loss
  and gradient thresholds;
- every run improves held-out MLM cross entropy by at least 5% and masked
  accuracy by at least one absolute percentage point;
- compiled versus eager, BF16 versus compiled, and BF16 versus eager all pass
  the frozen per-seed margins and paired 95% bootstrap non-inferiority bounds;
- BF16 is at least 1.25 times as fast as eager and 1.20 times as fast as FP32
  compiled for every seed and in the pooled measurement;
- every run passes the memory, step-time stability, provenance, membership, and
  eight-rank parity checks.

A pass only authorizes execution of a separately audited exact-recovery gate.
It does not itself authorize 16- or 32-GPU scaling.

## Primary sources

- PyTorch numerical accuracy:
  <https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html>
- PyTorch FSDP plus compile throughput and parity study:
  <https://pytorch.org/blog/maximizing-training-throughput/>
- PyTorch DDP communication hooks:
  <https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html>
- PyTorch compile placement for DDP:
  <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html>
