# ElasticABF recovery protocol v1

Status: registered for a real-OAS 35M correctness gate; the running 650M
production campaign is immutable and outside this experiment.

## Question

Can one authenticated DDP/DCP checkpoint continue across a 4-to-2-to-4 RTX
4090 topology change without changing the global sample stream, MLM masks,
token-indexed learning-rate coordinate, or terminal trainer coordinate, while
keeping BF16 loss and gradient-norm deviations inside a frozen numerical gate?

## First gate

The gate runs one four-update uninterrupted reference and one segmented replay:

1. two updates on four GPUs;
2. one resumed update on two GPUs; and
3. one resumed update on four GPUs.

Every stage uses the authenticated production OAS training split, ABF-35M,
16,384 target residue tokens per update, micro-batch 16, batch quantum 128,
seed 17, FP32 gradient reduction, fused AdamW, pinned staging, stable length
ordering, and no compilation. The small model and four-update window are a
correctness fixture, not a performance or biological-quality experiment.
The freezer performs the one full 90 GB logical array-hash pass. Each stage
then revalidates the signed manifests, immutable shard tree, file sizes, dtypes,
and shapes, avoiding four redundant full-corpus reads inside one allocation.

## Scientific invariants

- Global update membership is derived from the saved sample cursor and seed,
  independently of world size.
- MLM corruption is counter-derived from sample key, update, position, and
  seed. It does not consume rank-local RNG streams.
- Attention and residual dropout are exactly zero.
- The global target count is reduced before scaling each rank's loss.
- Learning rate is indexed by globally consumed residue tokens.
- The source RNG manifest and every source-rank RNG file are authenticated.
  On a world-size change they are not mapped to unrelated destination ranks;
  destination non-scientific RNG streams are deterministically reseeded.
- Model, optimizer, trainer-state, receipt, corpus, runtime, and source hashes
  are checked before the first resumed optimizer update.
- Optimizer hashes are compared only within the same representation domain:
  canonical DCP payload to canonical DCP payload, and applied native optimizer
  state to the source native optimizer state.

## Acceptance

- The first two same-world updates must have exact loss and gradient-norm
  traces after timing fields are excluded.
- Every update must have identical membership-derived coordinates, selected
  target count, sequence count, consumed tokens, and learning rate.
- Cross-world loss and gradient norm use
  `abs(observed-reference) / (0.005 + 0.02 * abs(reference)) <= 1`.
- DCP payload and applied model/optimizer hashes must equal the authenticated
  source receipt at both topology changes.
- The uninterrupted and 4-to-2-to-4 branches must end with identical model
  tensors, optimizer step counters, and param-group metadata. Moments are
  compared by canonical model parameter name, never process-local IDs. The
  independently rerun gate freezes `exp_avg` limits of `2^-29` absolute,
  `1e-3` relative, and 4096 ULP, plus `exp_avg_sq` limits of `2^-36`
  absolute, `1e-5` relative, and 32 ULP.
- Source RNG inventories and semantic trainer-state digests must form an
  unbroken chain.
- Final consumed tokens, next update, and sample cursor must equal the
  uninterrupted reference exactly.

Failure is retained as evidence. Thresholds, topology, model, steps, data,
seed, optimization, or source cannot be changed inside an attempt.

The optimizer-moment thresholds were calibrated from failed attempt 1457391,
which retained an exact terminal model but exposed maximum differences of
`2^-31` for `exp_avg` and `2^-38` for `exp_avg_sq`. They are registered at
four times those absolute differences before the independent acceptance run;
attempt 1457391 cannot itself pass the revised gate.

## Promotion boundary

Passing this gate authorizes only a longer ABF-650M elastic development gate.
That next gate must add a frozen OAS validation panel, 50 warmup plus 200
measured updates, checkpoint pause/recovery telemetry, and an uninterrupted
four-GPU comparator. No production campaign, downstream test, biological,
model-selection, scaling-efficiency, or SOTA claim follows from this v1 gate.
