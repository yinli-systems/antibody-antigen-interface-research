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

## Verified handoff correction

New remote bundle: preparation `1557224` completed `0:0`, with
`PASS_MODEL_READY_ALIGNED_TEST549_BUNDLE`. All 549 production-loader items and
all six model forward checks passed. The six complete evaluations are separate.

```text
/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/sabdab2_antigen_fold_test549_predicted_geometry_v2/
```

Externally pinned `BUNDLE.json` SHA-256:
`3e370ff3ebae1f6261865ace9b71666385ff5722d07f08388c5a490365400958`.
The sealed package has 1,956 inventoried files and occupied approximately 657 MiB
before retained full-evaluation predictions. `run_bundle.py` is the entrypoint;
`README.md` contains portable commands and task routing.

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
统一打包并做复现核对。整包 loader、六个模型前向及六路完整评测均已通过校验。
正确匹配的 549 条历史测试集上，固定三种子路由的 MPC-AP 中位数为 Contact
0.02667、Epitope 0.22031、Paratope 0.69859。请先不要用原 455 条结果表替换
manuscript 主结果；549 条结果也需要与相同测试划分、输入设定和指标口径下的
baselines 比较，不能直接套用原 455 条 benchmark 的对照行。

## Completed historical test replay

`FULL_SUMMARY/RESULT.json` status: `PASS_ROUTED_HISTORICAL_TEST549`.
Six components each evaluated all 549 instances with frozen checkpoints,
validation temperatures and thresholds. No retraining or selection was performed.

| Task | AUROC | Pooled AP | MPC-AP | Precision | Recall | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Contact | 0.92580 | 0.01001 | 0.02667 | 0.01502 | 0.09249 | 0.03461 |
| Epitope | 0.67184 | 0.12599 | 0.22031 | 0.10767 | 0.64497 | 0.12475 |
| Paratope | 0.95399 | 0.63403 | 0.69859 | 0.56869 | 0.74347 | 0.61144 |

Each cell is the median of that metric across seeds 17/29/43, not a new
single-checkpoint estimate and not a probability ensemble. Precision/Recall/MCC
above use the existing validation-selected threshold for each component;
fixed raw-sigmoid 0.5 and calibrated-probability 0.5 remain separate diagnostics in
the component outputs. MPC-AP excludes zero-positive complexes with explicit counts;
pooled metrics retain every evaluated candidate across all 549 instances.

Complete per-seed values and TP/FP/TN/FN are in
`results/aligned_test549_replay_v1/RESULT.json` and the remote
`FULL_SUMMARY/ROUTED_METRICS.csv`. Do not select seed17 for contact or seed29 for
paratope after seeing these test results. The preselected seed43 case-study
checkpoint and the three seed-matched routes are unchanged.

Final result SHA-256:
`26822d1b921f3566112b0f8498e230e41f5d2cbe487f7b5474e77c24d67a79d2`.
Final receipt SHA-256:
`e906ec9f91f1ede54611d65851eb5c641f0c870be2685ee1b31aa3bc2dd40f04`.

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
