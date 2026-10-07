# Pair-loss canary v1

This frozen, matched PINDER-Train-only canary compares the existing
adaptive-balanced residue-pair loss with a class-balanced focal loss. The
decoder, Stage-A0 checkpoint, 512-system training subset, 16-system
PDB-disjoint holdout, initialization seed, optimizer, auxiliary losses, and
64-update budget are identical. No test or temporal cohort was accessed.

## Result

| Metric after 64 updates | Adaptive-balanced control | Balanced focal |
|---|---:|---:|
| Contact AP | 0.002367 | 0.002217 |
| Contact AUROC | 0.641964 | 0.633216 |
| Contact MRR | 0.005411 | 0.004548 |
| Contact Recall@1 | 0 | 0 |
| Matched-minus-swapped mean logit | 0.061008 | 0.046414 |

The pre-registered paired difference-in-differences result for focal minus
adaptive-balanced was `-0.0001504 [-0.0003098, -0.0000154]` for contact AP,
`-0.008748 [-0.015029, -0.001861]` for contact AUROC, and
`-0.000863 [-0.001740, -0.000121]` for contact MRR. Partner-swap separation
also decreased by `-0.014594 [-0.025029, -0.005213]`.

Decision: `FAIL_BALANCED_FOCAL_PAIR_RANKING_GATE`; retain the existing
adaptive-balanced pair loss and do not promote focal loss to B32K.

## Integrity

- Analysis SHA-256: `9fee352a5c049d206e7ae1727ee0c0a3a43f38f7c997138aa3a51148b2639923`
- Adaptive-balanced result SHA-256: `6ca20e7d1e6b56034a9baf40d652016f2d02e09f058f0e3e7d24048a1dbaa54d`
- Balanced-focal result SHA-256: `fb3b1c50c1f9ad6dad2c49503517632bca5b7d216d90b4a6ac8d203fbb4a9477`
- Frozen source SHA-256: `b47eff405951c07c08feaf6b4928e0daea040617356ae729b71619fd301aff4c`
- `test_access=false`

The machine-readable paired analysis is in `ANALYSIS_V1.json`; each arm's
execution record is in its corresponding `RESULT.json` and `PROGRESS.json`.
