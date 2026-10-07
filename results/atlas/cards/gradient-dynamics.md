# Gradient alignment changes during training

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `readout-bound` — Committed research readout inspected; not independently recomputed

Recorded sign crossings occur at steps 77–110 on AsEP and 38–50 on SAbDab2. Converged positive cosine does not describe the early trajectory.

| Field | Value |
|---|---|
| workstream | Objectives & controls |
| cohort | AsEP/SAbDab2 training diagnostics |
| inputs | Shared-trunk gradients and two diagnostic toolchains |
| seeds | 3 runs per documented cohort/toolchain |
| estimand | Per-step trajectories, not accuracy |

## What this does not establish

Gradient norm or cosine alone does not establish the mechanism of the loss-ablation performance effect.

## Pinned evidence

1. [results/gradient_alignment_dynamics_v1/step_trace_summary.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/gradient_alignment_dynamics_v1/step_trace_summary.json)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `e0d6c84e7b61ed173201645cd2f4a896c6d7b90b`; SHA-256 `8251cfdf5842d1a70715e5bd970a5769258dd9513cf03eeedc9c075034e8defb`.
2. [results/CLAIMS_LEDGER.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/CLAIMS_LEDGER.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `de64b7b1bd246f9119b35d9e061407514d14c995`; SHA-256 `11601363a09cd2b61f8557c9a5ed7091e646624a06ab1bac8bbd11d20a925e86`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
