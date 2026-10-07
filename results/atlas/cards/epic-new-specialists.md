# October 4 report: specialist comparisons and input mismatch

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `report-only` — Newer narrative claim; matching terminal campaign artifacts not identified in the inspected HEAD trees

Reported EpiFormer epitope AP 0.372 exceeds reconstructed EPIC 0.214. Paraplume overlap excludes 69/455; on clean 386, improved EPIC 0.709 versus Paraplume 0.691 has CI crossing zero.

| Field | Value |
|---|---|
| workstream | EPIC newer report |
| cohort | Reported official 455 and overlap-clean 386 |
| inputs | EpiFormer bound experimental backbone versus EPIC predicted monomers; released Paraplume |
| seeds | Reported 3-seed EpiFormer/EPIC and released specialist |
| estimand | Matched labels/axes do not imply matched inputs |

## What this does not establish

Not a modality-matched architecture leaderboard. As-published labels need checking against the unavailable original checkpoint. Newer manuscript claim: matching terminal campaign artifacts were not identified in the 30 inspected HEAD trees; they may exist off-Git. Attach run membership, source/split/selection hashes, final receipts and per-seed/per-complex outputs before promotion to artifact-verified headlines.

## Pinned evidence

1. [paper/neurips_2026_framecontact/REVIEWER_RESPONSE_20261004.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/paper/neurips_2026_framecontact/REVIEWER_RESPONSE_20261004.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `dd878e9b097515b3f6ff51c596d9c8d14287dadd`; SHA-256 `7524e311c438085891ebc856df4fd2afdc322e9bc32862fcf4f6673b5f6f3cce`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
