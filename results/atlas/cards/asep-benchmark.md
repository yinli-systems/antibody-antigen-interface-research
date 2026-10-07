# AsEP benchmark position without inflation

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

Ten-seed macro MCC≈0.1572 on epitope_group and 0.2584 on epitope_ratio. These constrain rather than establish benchmark leadership.

| Field | Value |
|---|---|
| workstream | External baselines |
| cohort | AsEP group/ratio test protocols with exposure histories |
| inputs | Repository model under documented benchmark conventions |
| seeds | 10 seeds per protocol |
| estimand | Per-complex MCC with validation-selected thresholds |

## What this does not establish

Pooled-versus-macro comparisons and favorable older literature rows cannot establish SOTA. Preserve registered test-access history.

## Pinned evidence

1. [results/asep_benchmark_v1/BENCHMARK.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/asep_benchmark_v1/BENCHMARK.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `f0f3d3b50bdd99236e8ccb374fd0c6a4a479d5d3`; SHA-256 `b06f8b39adbb0f6362c55b9bbe2615105fcc69fb965d413099d569a4e8af1b55`.

## Exact checked registry values

| ID | Value | Estimator |
|---|---|---|
| `asep_group_mcc` | `0.15724713931069248` | mean of per-complex MCC over ten trained seeds |
| `asep_ratio_mcc` | `0.25839540723437454` | mean of per-complex MCC over ten trained seeds |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
