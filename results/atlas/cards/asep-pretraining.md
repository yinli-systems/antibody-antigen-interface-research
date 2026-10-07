# AsEP transfer: pair gain and epitope trade-off

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `readout-bound` — Committed research readout inspected; not independently recomputed

September 4 summary reports pooled pair AUROC 0.6220→0.6904 for1D versus3D+margin transfer, while epitope pooled AP falls 0.2210→0.1567.

| Field | Value |
|---|---|
| workstream | Historical studies |
| cohort | Official 455 SAbDab2 transfer; separate 172-case AsEP-derived study |
| inputs | AsEP-derived pretraining and1D/3D+margin conditions |
| seeds | Seed replication not fully specified by summary |
| estimand | Pooled/macro AUROC, pooled epitope AP and pruning metrics |

## What this does not establish

Committed summary, not a newly verified matched campaign. Do not infer significance or compare AUROC to another cohort’s AP.

## Pinned evidence

1. [docs/evaluations/2026-09-04_sabdab2_asep_benchmarks_summary.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/b730070e4d3247e26b3f64f5c10f8f4e16bfccae/docs/evaluations/2026-09-04_sabdab2_asep_benchmarks_summary.md)
   - Branch `feature/asep-pretraining-pipeline`; commit `b730070e4d3247e26b3f64f5c10f8f4e16bfccae`.
   - Git blob `dadbd005b134bce94fbf14c483c9f80aa0bcc747`; SHA-256 `e5044d6a4aaa86f8aae3456fb3f26f983f601015f212d8fae5b836b8f49267d6`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
