# Contact-gated interaction successor audit

## Scope

This is **single-seed development evidence from a later FrameContact successor campaign**, not an ablation of the anonymous submitted EPIC checkpoint. Slurm array `1351883` used the frozen ICG exploratory protocol. Three learned-gate variants completed at seed 17; the intended ordinary/no-gate control (`1351883_5`) timed out after 3 days and produced neither a final result nor a run receipt. Therefore no causal learned-gate effect is authorized.

## Completed successor variants

| Variant | Contact macro AP | Contact pooled AP | Epitope macro AP | Paratope macro AP | Contact ECE |
|---|---:|---:|---:|---:|---:|
| framecontact_icg_m3_full_gate | 0.025956 | 0.012446 | 0.226989 | 0.726048 | 0.057747 |
| framecontact_icg_m4_interaction_gate | 0.026632 | 0.012076 | 0.208928 | 0.737060 | 0.045125 |
| framecontact_icg_m6_joint_consistent | 0.020295 | 0.009174 | 0.195934 | 0.709847 | 0.131206 |

The measurements are heterogeneous: M4 has the highest contact/paratope macro AP among these three, whereas M3 has higher contact recall and epitope macro AP; M6 is lower on contact and epitope ranking despite additional consistency/distogram objectives. These are not tensor-identical interventions, and validation was also used for calibration/threshold selection.

## Missing control and claim boundary

The no-gate M2 task timed out and left no completed evaluation. The original 59 MB stdout is represented by its SHA-256/byte count and a compact truncated tail in `M2_TIMEOUT_TAIL.txt`; the failure is not hidden. Without M2, and without the exact submitted OpenDDE checkpoint/source authority, the camera-ready must not write “contact gating improves performance.” The supported statement is only that EPIC uses contact-conditioned communication, while later successor experiments show the mechanism is trainable and its behavior is endpoint-dependent.

Canonical machine-readable artifact: `ICG_SUCCESSOR_AUDIT.json`.
