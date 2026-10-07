# Ten-seed pair head versus its own marginal baseline

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

Paired macro AP margin +0.00311345, CI [0.00196869,0.00429043], positive 10/10. Ratio of means≈16.1%; mean seed ratios≈16.7%; neither is the single-seed 32.7%.

| Field | Value |
|---|---|
| workstream | Objectives & controls |
| cohort | 217 validation complexes /68 clusters; documented split drift |
| inputs | Bilinear head versus outer product of its own marginals |
| seeds | 10 seeds |
| estimand | Absolute paired difference primary; seed bootstrap and exact sign bound |

## What this does not establish

Exceeding a separable baseline does not alone prove true binding-partner identification; own-model marginals are not an independent specialist baseline.

## Pinned evidence

1. [results/baseline_margin_replication_v1/report.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/ed8c33fb3e25666a72cf1f368e03d83d5c7d89a4/results/baseline_margin_replication_v1/report.json)
   - Branch `kevin/evaluate-sensitivity-ci-fix-v1`; commit `ed8c33fb3e25666a72cf1f368e03d83d5c7d89a4`.
   - Git blob `78b0539245dcc8cc7212ccb94e0ce5bea8e65a22`; SHA-256 `d03a2ce41f1a0944e8eefd5cb815cbb642b52fbc31b7c78051232e148b98e447`.
2. [results/CLAIMS_LEDGER.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/CLAIMS_LEDGER.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `de64b7b1bd246f9119b35d9e061407514d14c995`; SHA-256 `11601363a09cd2b61f8557c9a5ed7091e646624a06ab1bac8bbd11d20a925e86`.

## Exact checked JSON fields

| Selector | Value |
|---|---|
| `["mean_margin"]` | `0.0031134461184687` |
| `["bootstrap_ci95"]` | `[0.0019686912510177143, 0.004290434941020674]` |
| `["seed_count"]` | `10` |
| `["positive_seed_count"]` | `10` |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
