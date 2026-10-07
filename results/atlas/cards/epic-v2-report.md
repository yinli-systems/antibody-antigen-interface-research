# October 4 report: ensemble gains and selection-transfer failure

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `report-only` — Newer narrative claim; matching terminal campaign artifacts not identified in the inspected HEAD trees

Reported AP 0.0214→0.0346 contact,0.2148→0.2502 epitope,0.6841→0.7215 paratope. Roughly half contact/epitope gain is seed ensembling. Contact-selected ensemble instead scores 0.0240 versus baseline ensemble 0.0273.

| Field | Value |
|---|---|
| workstream | EPIC newer report |
| cohort | Reported cluster-disjoint 324 validation and official 455 test |
| inputs | Reported averaging/regularization/feature ensemble; distinct from earlier PLM screen |
| seeds | Reported 3-seed arms; 75-improvement-model count unresolved |
| estimand | Separate single model, unchanged seed ensemble, new-component ensemble, and task-specific validation choice |

## What this does not establish

Paratope-selected ensemble happened to lead every test task; contact/epitope outcomes are exploratory, not successful per-task selection. Two rounds 6+12 arms×3 account for 54, not 75 without additional entries. Newer manuscript claim: matching terminal campaign artifacts were not identified in the 30 inspected HEAD trees; they may exist off-Git. Attach run membership, source/split/selection hashes, final receipts and per-seed/per-complex outputs before promotion to artifact-verified headlines.

## Pinned evidence

1. [paper/neurips_2026_framecontact/REVIEWER_RESPONSE_20261004.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/paper/neurips_2026_framecontact/REVIEWER_RESPONSE_20261004.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `dd878e9b097515b3f6ff51c596d9c8d14287dadd`; SHA-256 `7524e311c438085891ebc856df4fd2afdc322e9bc32862fcf4f6673b5f6f3cce`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
