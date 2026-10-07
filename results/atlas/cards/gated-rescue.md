# Gating reduces clean damage but does not establish safety

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

All registered synthetic corruptions trigger. Clean damage becomes −0.0004598 AP, crossed CI [−0.0011427,+0.0000198]; no clean-safety guarantee.

| Field | Value |
|---|---|
| workstream | Correspondence |
| cohort | 217 development complexes /68 components; train-only threshold calibration |
| inputs | Label-free gate; no contact AP used to choose thresholds |
| seeds | 3 paired rescue/downstream seeds |
| estimand | Equal-component/equal-seed contrasts and crossed intervals |

## What this does not establish

Train-clean trigger≈0.39% becomes 15.67% /17.97% /0.46% on validation. No post-hoc noninferiority margin or deployment guarantee.

## Pinned evidence

1. [extensions/rescue_qualification_v2/evidence/20260930/gated_rescue_v1/GATED_RESCUE_CROSSED_ANALYSIS.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc/extensions/rescue_qualification_v2/evidence/20260930/gated_rescue_v1/GATED_RESCUE_CROSSED_ANALYSIS.json)
   - Branch `kevin/rescue-qualification-20260930`; commit `a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc`.
   - Git blob `4eb8e91002cd83958cacd40332641f1fbfbb4f4a`; SHA-256 `f40f84837f3484e490513f5729b273c017620934725c50aa26a1b1bf2368e987`.
2. [extensions/rescue_qualification_v2/GATED_RESCUE_READOUT.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc/extensions/rescue_qualification_v2/GATED_RESCUE_READOUT.md)
   - Branch `kevin/rescue-qualification-20260930`; commit `a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc`.
   - Git blob `97b3521b78d5f813a32a34584451267721739f07`; SHA-256 `73d7b1db445d9bff650772698618873e44556f9fcaec9a26c8561f7aaf045366`.

## Exact checked JSON fields

| Selector | Value |
|---|---|
| `["unique_complexes"]` | `217` |
| `["independent_components"]` | `68` |
| `["record_count"]` | `1953` |
| `["clean", "gated_minus_aligned", "equal_component_equal_seed"]` | `-0.00045979402276384953` |
| `["clean", "gated_minus_aligned", "crossed_seed_component_ci95"]` | `[-0.0011426844725987797, 1.9782609094639334e-05]` |
| `["clean", "unconditional_minus_aligned", "equal_component_equal_seed"]` | `-0.001430459907154479` |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
