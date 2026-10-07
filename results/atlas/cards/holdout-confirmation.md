# Registered holdout: one encoder confirms

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

ProteinMPNN aligned−shuffled: +0.003594 AP,9/10 favorable, interval above zero. ESM-IF1: +0.002260,6/10, interval includes zero. Both outcomes must remain visible.

| Field | Value |
|---|---|
| workstream | Correspondence |
| cohort | Registered 217-complex holdout, distinct from development 217 |
| inputs | ProteinMPNN and ESM-IF1 correspondence controls |
| seeds | 10 paired seeds per encoder |
| estimand | Per-encoder paired macro AP contrast and registered decision rule |

## What this does not establish

Disclosed 32-case overlap with a different exposed baseline test. The 185-case sensitivity does not replace the registered outcome; do not pool encoder seed counts.

## Pinned evidence

1. [results/holdout_confirmation_v1/confirmation.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/holdout_confirmation_v1/confirmation.json)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `b46ee8b2699744b33c3495ecc0a2e0d420fe5f35`; SHA-256 `e30b34bd3720e9443ccd53975415ecf983671a9715aef63e4324c02526b2fb46`.
2. [results/holdout_confirmation_v1/README.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/holdout_confirmation_v1/README.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `33eb0a3f7cf547358fd541c4949dbe48f0ca4156`; SHA-256 `ddcf7e2f3ef5f5ba8dd8946151578817eaba5a789652dbe718d5bbfc325b06f3`.

## Exact checked registry values

| ID | Value | Estimator |
|---|---|---|
| `holdout_ProteinMPNN_delta` | `0.0035941254577831964` | paired-seed aligned minus shuffled; seed bootstrap |
| `holdout_ProteinMPNN_ci` | `[0.0015589868078879768, 0.005881458517017491]` | paired-seed aligned minus shuffled; seed bootstrap |
| `holdout_ESM-IF1_delta` | `0.002260074267027126` | paired-seed aligned minus shuffled; seed bootstrap |
| `holdout_ESM-IF1_ci` | `[-7.198970681111876e-06, 0.004900886221604488]` | paired-seed aligned minus shuffled; seed bootstrap |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
