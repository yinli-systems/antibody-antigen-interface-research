# Checkpoint-aligned historical test replay

Status: **PASS_ROUTED_HISTORICAL_TEST549**. Preparation `1557224`, all six
`1557225_[0-5]` evaluations and analyzer `1557226` completed with exit `0:0`.
Every component ran on an NVIDIA GeForce RTX 4090.

This corrects a handoff mismatch, not the model: the old455 ID set contains
272 training, 73 validation and 110 test instances under these checkpoints'
actual antigen-fold split. It is not a 94-case-shortened version of the real549
test set. No model was retrained/reselected; no threshold or temperature was refit.

The matched bundle contains 549 instances, 503 unique ESM-2 artifacts,
independent AF3-predicted chain-internal geometry and six frozen checkpoints.
Experimental contact labels are evaluation targets only. Some AF3 predictions
use homolog MSAs, so this is not a query-only structure-prediction claim.

## Three-seed routed medians

| Task | AUROC | Pooled AP | MPC-AP | Precision | Recall | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Contact | 0.92580 | 0.01001 | 0.02667 | 0.01502 | 0.09249 | 0.03461 |
| Epitope | 0.67184 | 0.12599 | 0.22031 | 0.10767 | 0.64497 | 0.12475 |
| Paratope | 0.95399 | 0.63403 | 0.69859 | 0.56869 | 0.74347 | 0.61144 |

Route: Contact/Paratope from baseline, Epitope from seed-matched TopK2 expert,
seeds17/29/43. Each column is independently median-aggregated over seeds.
Micro threshold metrics above use the **frozen validation-selected thresholds**.
Component outputs separately provide raw-sigmoid0.5 and calibrated-probability0.5.

All 549 complexes contribute to pooled metrics. **MPC-AP averages 548 eligible
complexes**: one complex with zero positive contacts (and therefore zero positive
residue marginals) is explicitly ineligible for positive-retrieval AP. It was
not removed from inference, the test set or pooled metrics. AP means sklearn
`average_precision_score`, not trapezoidal PR-AUC.

## Reuse / handoff

```text
/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/sabdab2_antigen_fold_test549_predicted_geometry_v2/
```

Start with its `README.md` and `run_bundle.py`. `READY` contains production-loader
and all-six-forward verification; `FULL_SUMMARY` contains the final routed report
and a nine-row per-seed CSV. `evaluation/0` through `evaluation/5` retain all
3,294 individual prediction archives (logits, labels, masks), component results
and hash receipts. The two source snapshots are included; inference paths are
portable and do not rely on the original absolute provenance paths.

Bundle manifest SHA-256:
`3e370ff3ebae1f6261865ace9b71666385ff5722d07f08388c5a490365400958`.
Result SHA-256:
`26822d1b921f3566112b0f8498e230e41f5d2cbe487f7b5474e77c24d67a79d2`.
Receipt SHA-256:
`e906ec9f91f1ede54611d65851eb5c641f0c870be2685ee1b31aa3bc2dd40f04`.
Remote CSV SHA-256:
`baea9bd532fed3d7f3fd1021d97e48867edcb28bf9905e2f6c7c78e56026fff7`.

This is an **already exposed historical antigen-fold evaluation**, not a new
blind or proven antibody-and-antigen sequence-double-cold test. Do not mix these
549 metrics with baselines evaluated on the original455, SABLE97 or MIPE560.
Do not choose a new checkpoint based on the values in this report. The
colleague's 455/Chai-1 table is not reproduced or validated by this run; its exact
script, ID list, geometry manifest and score export are needed for reconciliation.

The old455 packages and failed preparation evidence remain unchanged. Corrected
source, regression tests and submission receipts are under
`remote_tools/aligned_test549_handoff_v1` on the teacher-handoff branch.
