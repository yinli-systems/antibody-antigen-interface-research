# Aligned features versus shuffled and width controls

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

Aligned ESM-IF1 minus ESM-2: +0.005111 AP; shuffled minus ESM-2: −0.003093. The random-width control is undecided, not equivalent.

| Field | Value |
|---|---|
| workstream | Correspondence |
| cohort | SAbDab2 development 217 |
| inputs | Frozen ESM-2 with aligned/shuffled ESM-IF1 and random-width controls |
| seeds | 10 paired seeds |
| estimand | Macro pair AP differences; seed-bootstrap intervals |

## What this does not establish

Training-seed repeatability is not population confirmation or universal alignment necessity.

## Pinned evidence

1. [results/representation_attribution_v1/report.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/representation_attribution_v1/report.json)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `a00b0bd9d589a655902446b10e3755e3232ff6b4`; SHA-256 `23525fb42c38c82579299654b4ec54a68e37483fe7fdd7334b99b829adf4e832`.
2. [results/CLAIMS_LEDGER.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/CLAIMS_LEDGER.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `de64b7b1bd246f9119b35d9e061407514d14c995`; SHA-256 `11601363a09cd2b61f8557c9a5ed7091e646624a06ab1bac8bbd11d20a925e86`.

## Exact checked registry values

| ID | Value | Estimator |
|---|---|---|
| `rep_aligned_delta` | `0.005110687808672911` | paired-seed effect; confidence intervals resample training seeds |
| `rep_aligned_ci` | `[0.0037804980020868864, 0.006439117257188512]` | paired-seed effect; confidence intervals resample training seeds |
| `rep_shuffled_delta` | `-0.003092952877098886` | paired-seed effect; confidence intervals resample training seeds |
| `rep_random_ci` | `[-0.0021509073762782096, 0.0009703236011609193]` | paired-seed effect; confidence intervals resample training seeds |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
