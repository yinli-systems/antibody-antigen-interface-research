# FrameContact: three outputs on the corrected historical test

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

Corrected test 549 routed medians: contact AP 0.02667, epitope AP 0.22031, paratope AP 0.69859. The old 455 cache was not held out for this checkpoint.

| Field | Value |
|---|---|
| workstream | Interface prediction |
| cohort | Checkpoint-aligned historical 549 test, already exposed |
| inputs | Sequence plus independently predicted intra-chain geometry; no bound complex pose |
| seeds | 17 /29 /43; fixed routed medians |
| estimand | Mean-per-complex AP, not pooled AP or accuracy |

## What this does not establish

This correction applies to the specific checkpoint, not every official 455 evaluation. Some AF3 predictions use homolog MSAs. Not fresh blind or proven sequence-double-cold evidence.

## Pinned evidence

1. [results/aligned_test549_replay_v1/RESULT.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/aligned_test549_replay_v1/RESULT.json)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `a90a0437bf519397b68b4c22d22ef9f6a72a2900`; SHA-256 `26822d1b921f3566112b0f8498e230e41f5d2cbe487f7b5474e77c24d67a79d2`.
2. [paper/neurips_2026_framecontact/CHECKPOINT_TEST_ALIGNMENT_20260905.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/paper/neurips_2026_framecontact/CHECKPOINT_TEST_ALIGNMENT_20260905.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `53f1dbb89cb669a6f06c4cd32cbdfcfd5bfac719`; SHA-256 `e27a47ebb92ed9542a4b58e771b522d3e6cd2310eb94ddd49385d95a2cff4fc7`.

## Exact checked registry values

| ID | Value | Estimator |
|---|---|---|
| `framecontact_contact_ap` | `0.026674939866968273` | median of seeds 17/29/43; routed components, not a probability ensemble |
| `framecontact_epitope_ap` | `0.22031151958597614` | median of seeds 17/29/43; routed components, not a probability ensemble |
| `framecontact_paratope_ap` | `0.6985876493387135` | median of seeds 17/29/43; routed components, not a probability ensemble |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
