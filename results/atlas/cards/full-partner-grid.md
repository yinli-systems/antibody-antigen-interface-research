# Full partner grid does not confirm the small-panel anomaly

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

122×119=14,518 distinct pairings. The earlier small-panel MRR improvement after corruption is not robustly reproduced on the complete grid.

| Field | Value |
|---|---|
| workstream | Partner retrieval |
| cohort | 122-antibody /119-antigen development library |
| inputs | Frozen checkpoints and registered corruption conditions |
| seeds | 9 repeated grid tasks; not independent new pairings |
| estimand | Finite-library ranking, separate from within-complex contact AP |

## What this does not establish

Unobserved pairings are unlabelled alternatives, not verified non-binders. Score/length correlations do not prove causal bias.

## Pinned evidence

1. [extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_READOUT.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_READOUT.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `3f73027ac12d153cf3261bf678c56766804acd94`; SHA-256 `cc404e6b6336b53e87ca30fe22060d65e928161a1dc300fa96bf385e9e997fa8`.

## Exact checked registry values

| ID | Value | Estimator |
|---|---|---|
| `ranking_unique_pair_count_per_task` | `14518` | Cartesian candidate pairs and repeated measurements |
| `ranking_total_pair_evaluations` | `130662` | Cartesian candidate pairs and repeated measurements |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
