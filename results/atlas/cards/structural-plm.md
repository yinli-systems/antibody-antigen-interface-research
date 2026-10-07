# Structural PLMs: 36/36 complete, no promotion

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

Complete SaProt/ProSST sweep: 36 tasks,12 arms; no real-layer crossed interval excludes zero against its matched random-width control. Decision NO-PROMOTION.

| Field | Value |
|---|---|
| workstream | Representation screening |
| cohort | 217 exposed development complexes /68 components |
| inputs | Five real layers and one matched random control per family |
| seeds | 3 seeds per arm: 1/2/3 |
| estimand | Paired seed/component effects; development selection is not promotion |

## What this does not establish

ProSST selects layer 12; SaProt tie-window selects 24, not raw-highest 33. This is specific to this study, not universal structural-PLM failure.

## Pinned evidence

1. [extensions/frontier_multimodel_v1/evidence/structural_plm_final_20260930/ANALYSIS.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/96864049bdb935d14a4439c3f048231f59e78bee/extensions/frontier_multimodel_v1/evidence/structural_plm_final_20260930/ANALYSIS.json)
   - Branch `kevin/structural-plm-final-20260930`; commit `96864049bdb935d14a4439c3f048231f59e78bee`.
   - Git blob `0c515f1457de631731098f0de07d32be5f5482a1`; SHA-256 `d40d8a434349531bc7df9625220b3becef0b42a860da826310d5fa031264ed41`.
2. [extensions/frontier_multimodel_v1/STRUCTURAL_PLM_FINAL_READOUT.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/96864049bdb935d14a4439c3f048231f59e78bee/extensions/frontier_multimodel_v1/STRUCTURAL_PLM_FINAL_READOUT.md)
   - Branch `kevin/structural-plm-final-20260930`; commit `96864049bdb935d14a4439c3f048231f59e78bee`.
   - Git blob `e8d68f8842431bd1c3037c241ebf6f1cb4d8c2d3`; SHA-256 `1c7344aa808b091ace0fbf0400be2b58060762e98a38f54a0888d4640dec9fef`.

## Exact checked JSON fields

| Selector | Value |
|---|---|
| `["task_count"]` | `36` |
| `["arm_count"]` | `12` |
| `["validation_complex_count"]` | `217` |
| `["development_layer_selection", "prosst", "selected_layer"]` | `12` |
| `["development_layer_selection", "saprot", "selected_layer"]` | `24` |
| `["development_layer_selection", "saprot", "raw_highest_layer"]` | `33` |
| `["development_layer_selection", "prosst", "selected_mean_pair_macro_ap"]` | `0.02217555459107116` |
| `["development_layer_selection", "saprot", "selected_mean_pair_macro_ap"]` | `0.024451940544316726` |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
