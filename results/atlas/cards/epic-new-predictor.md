# October 4 report: predictor swap versus matched retraining

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `report-only` — Newer narrative claim; matching terminal campaign artifacts not identified in the inspected HEAD trees

Inference-time validation OpenDDE swap and matched retraining are separate. Reported ensemble test epitope AP:AF3 0.2321 versus OpenDDE 0.2157; a long-entity training fallback remains AF3.

| Field | Value |
|---|---|
| workstream | EPIC newer report |
| cohort | Reported 321 validation swap; separate 455 retrained test |
| inputs | AF3/OpenDDE monomer predictors |
| seeds | Reported 3 seeds per condition |
| estimand | Paired AP differences; inference swap is not retraining |

## What this does not establish

Reconcile 3,015-complete wording with the fallback. CI crossing zero is not equivalence. Newer manuscript claim: matching terminal campaign artifacts were not identified in the 30 inspected HEAD trees; they may exist off-Git. Attach run membership, source/split/selection hashes, final receipts and per-seed/per-complex outputs before promotion to artifact-verified headlines.

## Pinned evidence

1. [paper/neurips_2026_framecontact/REVIEWER_RESPONSE_20261004.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/paper/neurips_2026_framecontact/REVIEWER_RESPONSE_20261004.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `dd878e9b097515b3f6ff51c596d9c8d14287dadd`; SHA-256 `7524e311c438085891ebc856df4fd2afdc322e9bc32862fcf4f6673b5f6f3cce`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
