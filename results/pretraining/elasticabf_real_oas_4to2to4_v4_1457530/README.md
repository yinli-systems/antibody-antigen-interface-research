# ElasticABF real-OAS 4-to-2-to-4 gate

Status: **PASS**. Slurm job `1457530` completed on `gpu_4090` in 5 minutes
41 seconds with exit code 0.

This is a systems-correctness result for ABF-35M on the authenticated OAS
training split. It is not a production-pretraining, downstream, biological,
model-selection, or SOTA result.

## Result

| Gate | Result |
| --- | --- |
| DCP chain 4 to 2 to 4 | pass |
| Global sample, token, update, and LR coordinates | exact |
| Same-world prefix loss and gradient norm | exact |
| Cross-world loss maximum scaled error | 0.0 |
| Cross-world gradient-norm maximum scaled error | 0.0 |
| Terminal model tensors | exact |
| Optimizer step and param-group metadata | exact |
| Source rank RNG inventories | authenticated |

The terminal optimizer moments are intentionally reported as numerically
bounded rather than falsely labeled exact. Across 63 named parameter states,
the maximum `exp_avg` difference was `4.656612873077393e-10` (1639 ULP) and
the maximum `exp_avg_sq` difference was `3.637978807091713e-12` (7 ULP).
Both passed thresholds frozen before this independent run: `2^-29`, `1e-3`,
4096 ULP for `exp_avg`, and `2^-36`, `1e-5`, 32 ULP for `exp_avg_sq`.

## Evidence

- Gate protocol SHA-256:
  `2cbaf45ec3d00bbf5c8831471fc56edbbc455c78ea376ac7dd5cfffd196f07ff`
- Audit file SHA-256:
  `c5cdb0facf244c10d29548c04e1a1d2fc24d0124731711348801f5e4037d11e4`
- Optimizer diagnostic SHA-256:
  `f00dfe16dc6fe9593505f77da382a58e49585e1672b597c71a32ee94896b3c61`
- Audit payload receipt SHA-256:
  `9980c098db5b2ab584ffaa43c167596a787cfa73fe2d18eb1d52d6f7400c6a2c`

The repository contains the canonical protocol, audit, optimizer diagnostic,
run receipts, trainer states, GPU telemetry, logs, and remote
artifact manifest. Multi-hundred-megabyte DCP shards remain in the immutable
remote run root recorded by the receipts and are not duplicated in Git.

Attempt `1457391` is retained remotely as a calibration failure. It first
showed that process-local optimizer hashes are not a valid cross-topology
terminal criterion and cannot be retroactively accepted by this gate.
