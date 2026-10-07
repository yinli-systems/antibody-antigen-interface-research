# Pooling and ESM-C controls retain negative endpoints

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `readout-bound` — Committed research readout inspected; not independently recomputed

TopK1/2/4/8 and log-mean-exp are close; noisy-OR is worse. Published AbCDR-ESMC improves contact/epitope but lowers paratope. Local CDR adaptation trades CDR against full-chain quality.

| Field | Value |
|---|---|
| workstream | Historical studies |
| cohort | Development cohorts in August 27 snapshot |
| inputs | Pooling rules and ESM-C representation controls |
| seeds | Study-specific seeds, not pooled |
| estimand | Readout AP, downstream AP and MLM metrics separate |

## What this does not establish

Local MLM adaptation is not proof of a downstream gain. Preserve negative endpoints beside positive ones.

## Pinned evidence

1. [results/LATEST_RESULTS_20260827.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/LATEST_RESULTS_20260827.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `05c109cf66d0f41d03566a2f09d1c425eaba5f80`; SHA-256 `eadb6c4e747c02da3e260f3a958219a293afe7876bd69aea4c3da96bd2f0bb0b`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
