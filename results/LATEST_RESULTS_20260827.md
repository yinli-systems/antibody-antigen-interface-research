# Audited result snapshot — 2026-08-27

This index links the latest compact, machine-readable analysis artifacts. It
does not include checkpoints, embeddings, pair-level prediction matrices, or
private wet-lab candidate payloads.

## FrameContact task routing

- **Fresh156 route confirmation:** all three fixed seeds improved epitope
  mean-per-complex average precision by `+0.005702`, `+0.014097`, and
  `+0.008877`; the median seed gain is `+0.008877`. The equal-antigen-cluster,
  seed-averaged paired estimate is `+0.029518` with 95% bootstrap interval
  `[0.012630, 0.047429]` over 49 clusters. Contact and paratope are unchanged by
  deterministic routing.
- **Full development:** routed median contact/epitope/paratope
  mean-per-complex AP is `0.028576 / 0.237464 / 0.704746`; the epitope median
  gain over the joint baseline is `+0.005051`, with contact and paratope again
  unchanged by construction.
- **Evidence boundary:** fresh156 confirms the route chosen before full
  retraining. The later 2,565/549 run uses the 549-complex validation set for
  checkpoint selection and is therefore a development summary, not independent
  confirmation. Neither artifact reads the old exposed test or temporal
  holdout.

Artifacts: [`framecontact_task_routing_v1/`](framecontact_task_routing_v1/)

## Epitope pooling readout control

A paired, fixed-checkpoint control on the 549-complex development validation
set compares TopK1/max, TopK2, TopK4, TopK8, noisy-OR, and log-mean-exp without
parameter updates. Mean-per-complex AP is `0.235524`, `0.235871`, `0.235907`,
`0.235891`, `0.179440`, and `0.235520`, respectively. The TopK1/2/4/8 and
log-mean-exp differences are small and their paired bootstrap intervals overlap
zero; noisy-OR is materially worse. This is a readout sensitivity control, not
the still-separate strict seven-arm retraining result.

Artifacts: [`epitope_pooling_readout_v1/`](epitope_pooling_readout_v1/)

## ESM-C representation controls

- **Published AbCDR-ESMC positive control:** relative to base ESM-C 600M, the
  three-seed development median changes are contact `+0.002540`, epitope
  `+0.005752`, and paratope `-0.006018`. This authorizes the published checkpoint
  as a development positive control, not a test claim.
- **Local paired-only adaptation:** at fixed epoch 4, CDR-focused adaptation
  versus whole-chain adaptation changes CDR50 MLM accuracy by `+0.044513` and
  loss by `-0.130777`, but changes whole-chain15 accuracy by `-0.077141` and
  loss by `+0.252473`. The result supports a CDR/full-chain trade-off only;
  downstream performance-gain claims remain unauthorized.

Artifacts: [`esmc_representation_v1/`](esmc_representation_v1/) and
[`esmc_cdr_adaptation_v1/`](esmc_cdr_adaptation_v1/)

## Structural robustness and prospective gate

- **Bound-to-unbound geometry:** the frozen 52-case measurement completed with
  no model selection, calibration refit, exposed-test access, or temporal
  access. It is a measurement-only robustness analysis and does not authorize
  redistribution of the source structures.
- **Prospective temporal double-cold:** failed closed before label opening. The
  label-free cohort contained 15 eligible complexes but only 9 unique PDBs,
  below the pre-frozen requirement of 12 complexes and 10 unique PDBs. No task
  labels or cross-chain distances were read, and thresholds were not changed.

Artifacts: [`bound_unbound_geometry_v1/`](bound_unbound_geometry_v1/) and
[`temporal_double_cold_v1/`](temporal_double_cold_v1/)
