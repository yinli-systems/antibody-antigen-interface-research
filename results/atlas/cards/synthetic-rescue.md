# Learned rescue: synthetic gain with clean-input cost

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

Same-AA rescued−corrupted AP: +0.0157209,95% CI [0.0112560,0.0211527]. Clean inputs incur about −0.001430 component-weighted AP cost.

| Field | Value |
|---|---|
| workstream | Correspondence |
| cohort | 217 development complexes /68 components; 3 synthetic conditions |
| inputs | Trained residue matcher and fixed downstream models |
| seeds | Paired (101,1),(202,2),(303,3) |
| estimand | Equal-component/equal-paired-seed effect; pointwise crossed seed×component bootstrap |

## What this does not establish

1,953 rows are repeated observations of 217 complexes, not 1,953 biological samples. Not OOD, natural-corruption detection or production improvement.

## Pinned evidence

1. [extensions/rescue_qualification_v2/evidence/20260930/evidence_attempt3/THREE_SEED_AGGREGATE.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc/extensions/rescue_qualification_v2/evidence/20260930/evidence_attempt3/THREE_SEED_AGGREGATE.json)
   - Branch `kevin/rescue-qualification-20260930`; commit `a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc`.
   - Git blob `7598fc0120d54de232f53a5f9dc3a6efd7fac980`; SHA-256 `b5dc2f31682a17502262c60c72cee9d926557f520ffe532da2b070ee421adcdc`.
2. [extensions/rescue_qualification_v2/READOUT.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc/extensions/rescue_qualification_v2/READOUT.md)
   - Branch `kevin/rescue-qualification-20260930`; commit `a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc`.
   - Git blob `576aed0760b5b88077a3f267704257df566ad4d6`; SHA-256 `a659cb4617d5ee26ab4f4008043e5fcde250b7654029500d435389b5e2280043`.

## Exact checked JSON fields

| Selector | Value |
|---|---|
| `["contrasts", {"where": {"condition": "same_aa_shuffle:0", "contrast": "rescued_ap_minus_corrupted_ap"}}, "equal_component_equal_paired_seed_effect"]` | `0.01572086539530153` |
| `["contrasts", {"where": {"condition": "same_aa_shuffle:0", "contrast": "rescued_ap_minus_corrupted_ap"}}, "pointwise_crossed_bootstrap_ci95"]` | `[0.011255990518420451, 0.02115269924594697]` |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
