# Cheap selectors also detect synthetic corruption

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `readout-bound` — Committed research readout inspected; not independently recomputed

Moved-fraction and related inexpensive statistics also detect same-AA/global permutations at or near 100%. The synthetic construction moves about 91% of same-AA rows and nearly all globally shuffled rows.

| Field | Value |
|---|---|
| workstream | Correspondence |
| cohort | Same 217 development complexes; training-only calibration |
| inputs | Label-free heuristic controls |
| seeds | 3 fixed seeds |
| estimand | Detection and clean-trigger rates, not downstream AP |

## What this does not establish

Counterevidence to a uniquely capable natural-error detector. Preserve the synthetic-line stopping boundary; no repeated threshold tuning on exposed outcomes.

## Pinned evidence

1. [extensions/rescue_qualification_v2/GATED_RESCUE_READOUT.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc/extensions/rescue_qualification_v2/GATED_RESCUE_READOUT.md)
   - Branch `kevin/rescue-qualification-20260930`; commit `a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc`.
   - Git blob `97b3521b78d5f813a32a34584451267721739f07`; SHA-256 `73d7b1db445d9bff650772698618873e44556f9fcaec9a26c8561f7aaf045366`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
