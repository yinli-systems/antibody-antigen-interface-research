# October 4 report: reconstructed ablation campaigns

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `report-only` — Newer narrative claim; matching terminal campaign artifacts not identified in the inspected HEAD trees

Manuscript reports 36-model CampaignA and 39-model CampaignB; predicted-geometry epitope gains about+0.031 /+0.025 AP. Gate/cross-attention findings are mixed.

| Field | Value |
|---|---|
| workstream | EPIC newer report |
| cohort | Reported official 455 test; 2,887 train /321 carved validation |
| inputs | Reconstructed AF3-monomer EPIC, not the original submitted checkpoint |
| seeds | Reported 17/29/43 |
| estimand | Reported simultaneous PDB-block contrasts; analysis revision needs authority |

## What this does not establish

Newer manuscript claim: matching terminal campaign artifacts were not identified in the 30 inspected HEAD trees; they may exist off-Git. Attach run membership, source/split/selection hashes, final receipts and per-seed/per-complex outputs before promotion to artifact-verified headlines.

## Pinned evidence

1. [paper/neurips_2026_framecontact/REVIEWER_RESPONSE_20261004.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/paper/neurips_2026_framecontact/REVIEWER_RESPONSE_20261004.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `dd878e9b097515b3f6ff51c596d9c8d14287dadd`; SHA-256 `7524e311c438085891ebc856df4fd2afdc322e9bc32862fcf4f6673b5f6f3cce`.
2. [paper/epic_camera_ready_20260930/campaign_v1/ANALYSIS_PROTOCOL.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/a7bb1a0ca76fd7edca603842425bfc1ec2e1523c/paper/epic_camera_ready_20260930/campaign_v1/ANALYSIS_PROTOCOL.json)
   - Branch `kevin/epic-reviewer-response-20260930`; commit `a7bb1a0ca76fd7edca603842425bfc1ec2e1523c`.
   - Git blob `63c57b08b7d31550bb6cecc4945f32bff0f0e8dd`; SHA-256 `163245a45aee020b71c9f995a997ba7e08913286a95d8b10d7cf8a2a7c581012`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
