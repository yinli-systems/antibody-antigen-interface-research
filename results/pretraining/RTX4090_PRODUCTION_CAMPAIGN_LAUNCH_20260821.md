# RTX 4090 OAS production-pretraining launch — 2026-08-21

Status at `2026-08-21T18:17:16+08:00`: the first production segment is
running on four `gpu_4090` devices. This is a launch and recovery-evidence
record, not a terminal accuracy or throughput result.

## Frozen campaign

- Model preset: `abf_650m` (634,513,920 parameters).
- Corpus authority: 198,345,366 unique train records and 22,272,431,632
  unique train residue tokens.
- Campaign limit: 30,000,000,000 admitted residue tokens, including a planned
  repeat exposure of 7,727,568,368 tokens.
- Execution: four RTX 4090 GPUs, BF16, DDP, fused AdamW, pinned staging,
  compiled transformer blocks, compiled CUDA graphs disabled, and one static
  length bucket at 194 residues.
- Batch contract: micro-batch 16 per GPU, target 131,072 residue tokens per
  optimizer update, batch quantum 256, seed 17.
- Segment contract: at most 3,000 updates and four hours per segment, with a
  one-hour reserve. A successful immutable segment audit authorizes and
  submits exactly one successor; a failed audit cannot continue the chain.
- Validation: a fixed 8,192-record validation panel is evaluated after every
  segment. Downstream transfer and test access are not authorized.

The frozen campaign gate has file SHA-256
`3ab66eff7c8c99f48159ef0d0c32ff81ada9f2db513c28f3fe009560f1b3700e`.
The exact cluster adaptation is recorded in
[`patches/rtx4090_pretraining_campaign.patch`](../../patches/rtx4090_pretraining_campaign.patch).

## Recovery evidence

The accepted four-GPU recovery run compared one continuous four-update run
against a two-update checkpoint followed by an exact two-update resume. The
independent audit passed all four equality conditions:

- final model equality;
- final optimizer equality;
- scientific trace equality;
- terminal-coordinate equality.

The recovery gate SHA-256 is
`41117d571a5ee9da333644a3b27fe2306be3b37bcbeb4cf7b079cd0355370d86`.
The independent recovery audit file SHA-256 is
`58d2da1104eb44c114ad4bbf4f6d5778aeab0f2347abac0312e8a66107028217`,
with receipt payload SHA-256
`816e6a6e3f36e3a03123e9a5cb71ffc6a4e644d9c9b17a79fe7a21cb47868cb1`.

Two earlier attempts are retained as negative systems evidence. The first
failed before training because its login-node runtime manifest did not match
the compute-node environment. The second passed the runtime and hardware
checks but exposed that PyTorch 2.7.1 does not accept the optional
`DistributedDataParallel(batched_grad_copy=...)` argument. The compatibility
change now passes the argument only when the runtime supports it, and fails
closed if the unsupported feature is explicitly requested.

## Efficiency decision

The historical static-194 compiled candidate passed memory, numerical,
validation non-inferiority, replicate-state, replicate-stability, MFU, and
step-stability checks. It did not beat eager execution on relative throughput:
the two throughput-promotion gates remain explicitly false. That negative
result is preserved in the campaign gate.

Compiled static-194 is nevertheless the operationally safe RTX 4090 choice:
the historical eager run used about 23.5 GiB, leaving inadequate headroom on a
24 GiB card, whereas the accepted compiled recovery peaked near 19.6 GiB. The
historical compiled p95 of 1.966 seconds came from different hardware and is
used only as a reference; segment planning applies a 1.8× derating, and the
first RTX 4090 segment will provide the authoritative steady-state throughput.
After compilation, a point-in-time production telemetry sample showed
20,397 MiB used per GPU and approximately 96–100% utilization across the four
cards. This supports allocation health and memory headroom, but it is not a
completed throughput benchmark.

## Claim boundary

This evidence authorizes resumable OAS masked-language-model pretraining only.
It does not establish a terminal validation score, downstream gain,
comparative superiority, biological performance, or SOTA. No SAbDab2, AsEP,
official, frozen, or external test labels are consumed by this campaign.
