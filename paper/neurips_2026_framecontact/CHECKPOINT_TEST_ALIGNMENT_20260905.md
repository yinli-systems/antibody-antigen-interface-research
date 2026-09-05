# Checkpoint / test-cache alignment correction — 2026-09-05

## Confirmed issue

The original test455 cache and the selected no-privileged-distance checkpoint use
different split authorities. Recounting exact case-sensitive instance IDs gives:

| Original 455 instances, classified by checkpoint split | Count |
|---|---:|
| Training | 272 |
| Validation | 73 |
| Held-out test | 110 |

The checkpoint's actual held-out test has 549 instances, of which 439 are absent
from the original 455 list. The sets are **not** related by simply dropping 94
test cases. Averaging three seeds or changing the structure predictor cannot
repair training/test overlap.

The embedding-only `sabdab2_esm2_frozen_test455_v1` package has no geometry. The
separate `sabdab2_original_test455_model_cache_v1` package uses experimental
intra-chain geometry and is explicitly restricted to case-study use. The
colleague's actual Chai-1/configuration/threshold usage still needs their run
manifest; it cannot be inferred from their summary table alone.

## Correction being implemented

New remote bundle (verified ready only when `READY/RESULT.json` has PASS):

```text
/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/sabdab2_antigen_fold_test549_predicted_geometry_v2/
```

The bundle reuses the existing 549-instance AF3-predicted geometry and original
ESM-2 bytes. No structure prediction, labels, checkpoints, split, routing,
training or calibration are regenerated. Experimental contact labels are
measurement targets, not model inputs. The old 455 packages are preserved.

Use three fixed seed-matched routes: Contact and Paratope from `baseline`,
Epitope from `edge_topk2_supervised`, seeds 17/29/43. The previously supplied
seed43 epoch6 checkpoint is the baseline/contact case-study selection; it is
not the complete routed system.

The new entrypoint hashes the inputs, copied source and checkpoints before
inference; rejects overlap with the exact development IDs; strips labels before
`model.forward`; and checks unchanged model parameters afterward. Reports keep
pooled AP and MPC-AP separate, and distinguish frozen validation thresholds,
fixed raw-sigmoid 0.5, and fixed calibrated-probability 0.5.

This remains an **exposed historical antigen-fold test**, not a fresh blind test
and not a newly proven antibody-and-antigen sequence-double-cold benchmark.
Some AF3 predictions used homolog MSAs; do not call these query-only predictions.

## What can be said to collaborators now

这次核对发现，之前交付的 original 455 split 与 seed43 epoch6 checkpoint 的
训练划分不一致：其中 272 条属于它的训练集、73 条属于验证集、110 条才属于
它的 held-out test。因此现在的数值不能直接作为这份 checkpoint 的独立测试
结果，也不能把差异解释为“少了近 100 个样本”。我正在把 checkpoint 对应的
549 条历史测试集、预测几何、ESM-2 embedding、六个 checkpoint 及固定路由
统一打包并做复现核对。等整包和三种子结果通过校验后，再给大家对应路径和
同口径指标；暂时不要用这张 455 条结果表替换 manuscript 主结果。

## Execution notes

- Packaging/inference code and ten guard tests are isolated under
  `remote_tools/aligned_test549_handoff_v1` on the teacher-handoff worktree.
- An initial submission overlapped an unfinished file transfer: `sbatch` could
  not find `prepare.sbatch`; no job was started. The failed submission receipt
  is preserved. The submitter now requires an externally hashed transfer
  manifest before submitting, and any recovery uses a separate receipt name.
- Scheduling is not completion. `READY` records loader/forward verification;
  `FULL_SUMMARY` separately records the six complete component evaluations.
- Preparation `1557211` failed closed before model inference because the builder
  expected `checkpoint_sha256` instead of the actual v2 receipt field
  `selected_checkpoint_sha256`. Dependent `1557212/1557213` were cancelled.
  The incomplete `_v1` bundle is preserved; the corrected `_v2` build adds
  regression coverage for the real receipt schema and does not relax its hashes.
- Recovery chain: preparation `1557224`, evaluation array `1557225_[0-5]`,
  analyzer `1557226`. The preparation was observed RUNNING on `gpu_4090`;
  evaluation and summary remain success-dependent. See the separate submission
  receipt `SUBMISSION_RECOVERY_RECEIPT_SCHEMA_V2.json` for exact tool digests.
