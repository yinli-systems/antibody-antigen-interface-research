# Current execution status — 2026-09-30

This file is a timestamped operational snapshot, not a substitute for terminal result receipts.

Observed at approximately **2026-09-30 06:52 UTC**.

## Structural PLM layer sweep

- Formal array: `1640503`, 36 tasks, maximum six concurrent RTX 4090 jobs.
- Tasks `0–5`: **COMPLETED 0:0**.
- Tasks `6–11`: RUNNING at the snapshot.
- Tasks `12–35`: PENDING behind the `%6` concurrency limit.
- Complete-campaign analysis: `1640514`, PENDING on `afterok:1640503`.
- No formal `FAILURE.json` existed at the snapshot.

Completed primary validation pair-AP receipts:

| Task | Arm | Seed | best AP | last AP |
|---:|---|---:|---:|---:|
| 0 | SaProt layer 0 | 1 | 0.0179876 | 0.0158388 |
| 1 | SaProt layer 0 | 2 | 0.0216760 | 0.0188744 |
| 2 | SaProt layer 0 | 3 | 0.0220817 | 0.0184690 |
| 3 | SaProt layer 8 | 1 | 0.0207800 | 0.0183922 |
| 4 | SaProt layer 8 | 2 | 0.0204811 | 0.0183127 |
| 5 | SaProt layer 8 | 3 | 0.0218583 | 0.0187704 |

No layer winner or representation-superiority claim is authorised before the full 36-task analysis and matched controls complete.

## RePaRank

- `1640813`: official sequence checkpoint full 217-case same-grid evaluation — **COMPLETED 0:0**.
- `1640815`: sequence+structure three-case engineering canary — **COMPLETED 0:0**.
- `1640868`: first full sequence+structure run — **FAILED 1:0** after discovering missing predicted-chain structures. This is an authority coverage failure, not a performance result.
- Coverage audit: 187/217 validation complexes have complete predicted-chain authority; 30 are incomplete, comprising 40 unique missing chains (18 heavy, 3 light, 19 antigen; 7,985 residues).

## AF3 structure-authority completion

A 40-chain, no-MSA/no-template AlphaFold 3 completion plan is frozen at [`../extensions/frontier_multimodel_v1/evidence/af3_completion_v1/PLAN.json`](../extensions/frontier_multimodel_v1/evidence/af3_completion_v1/PLAN.json).

The latest task-0 scheduler observation is `1640914_0`, **FAILED 2:0 in 1 second**, before AF3 inference. The completion runner expects the legacy `$CAMPAIGN_ROOT/batches/batch-000/inputs/` layout while the new frozen builder stores task input under `$CAMPAIGN_ROOT/inputs/batch-000/`; this is treated as an orchestration-contract failure, not as an AF3 or scientific failure. No task 1–39 result is promoted from this state.

## Git / provenance

- Pre-consolidation `main`: `910b752ecf3e1493b80a7605151048bf2cd3e69b`.
- Active multimodel branch snapshot imported into this consolidation: `28dffada7aeaf9e5969f87414e91529aef60f717`.
- Every pre-consolidation branch head is pinned by `archive/2026-09-30/...` tags.
