# Frontier FSDP2 and DCP evidence

This directory records the first executable FSDP2 and PyTorch Distributed
Checkpoint correctness chain for the reduced R2 reranker fixture. The final
source is commit `afbc0abe03be08a3966edffb88e63c4eb83dc9ca`, with executed source
digest `c5260f1e60d7fdf360d35799dcca5a751cea113cb6ef76a5259776bcf16ee805`.

## Final FP32 results

| Job | GPUs | FSDP2 vs one-process reference | DCP continuation | Result SHA-256 |
|---|---:|---|---|---|
| `1339702` | 1 | max absolute `0`; max scaled error `0` | exact zero over 81 tensors / 44,531 elements | `6d718ab3a89216cd1b100dbef15a82e63703cee04a0a3e55476759605b9ef959` |
| `1339704` | 8 | max absolute `3.5686e-6`; max scaled error `0.1784 < 1` | exact zero over 81 tensors / 44,531 elements | `4769c16470a51f53866e9d8a19b23bcc97995bd84932c4789db73cd83fc29882` |

The elementwise gate is
`abs(observed-reference) / (atol + rtol * abs(reference)) <= 1`, with
`atol=2e-5` and `rtol=2e-4`. Large naive relative error near reference zeros
does not bypass or fail the gate; the recorded scaled error is the authority.

The eight-rank immutable checkpoint contains eight DCP shard files plus
metadata and a create-only receipt. Its receipt content digest is
`37cdc2ec2408e44fbdbbd494e6a5a5e97d5e72f75cf2743a860e5e9654e5a647`.
The checkpoint shards remain on the cluster and are not duplicated in Git.

## Retained failures

- Job `1339683` stopped before model execution because the historical runtime
  lock no longer matched the live compute-node framework state. Its invalid
  verification receipt is retained.
- Job `1339691` stopped before the first optimizer step because public FSDP2
  rejects zero-dimensional parameters. The fix explicitly promotes the R2
  contact bias to a length-one vector for both FSDP2 and the reference without
  changing broadcast mathematics. The failure log is retained.

The compute-node runtime was then re-locked create-only and verified in job
`1339688`. Its manifest file SHA-256 is
`5e8c0ecf201876d0559331299187e4a8b8820124d79bdb24b9973b3b57e24202`.

## What these artifacts do not show

The three-step fixture is intentionally too short for stable throughput or
scaling conclusions; its cold-start timing percentiles are not performance
benchmarks. It does not access SAbDab2 labels, outer folds, or a frozen test.
It does not prove BF16 parity, cross-world-size FSDP2 resharding, 32/64-GPU
efficiency, a full-size model, partner specificity, or biological quality.
Those require separately frozen experiments.

`execution_summary.json` is the compact machine-readable index. Each final
job folder contains the downloaded runtime, launch, environment, GPU trace,
result, checkpoint receipt, log, and remote artifact-hash inventory. The eight
downloaded files per passing job were rehashed locally against that remote
inventory before publication.
