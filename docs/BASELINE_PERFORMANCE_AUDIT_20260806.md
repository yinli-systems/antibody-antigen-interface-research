# Baseline performance audit and fair-comparison plan

更新时间：2026-08-06  
范围：EpiFormer、MIPE、SPPIDER-seq 的可审计 SAbDab2 结果；以及与 AsEP 上最新 LF3DRoPE 结果的协议对照。

## 先给结论

1. EpiFormer 的 SAbDab2 formal test 已经有本地冻结结果，不是 smoke：455/455 个 test complex 完成映射，mapping failure 为 0，prediction verification 通过。它是 **epitope-only projection**，没有 native residue-pair/contact 输出，也没有 paratope head。
2. 在与 SPPIDER-seq **完全相同的 455 个 SAbDab2 test instance** 上，EpiFormer 的 epitope micro AUPRC/F1/MCC 分别为 0.2775/0.3350/0.2715，SPPIDER-seq 为 0.1454/0.2104/0.1269。两者的逐 complex paired bootstrap 差值均为正，说明这不是只由 pooled 长度分布造成的表面差异。
3. MIPE 的本地结果是官方五折 held-out 评测（560 个 held-out complexes），不是 SPPIDER 的 455-complex frozen test；它应单独标为 `official five-fold held-out`，不能把两个 split 直接叫成同一个 test。
4. 当前 FrameContact 主表仍是严格 antigen-fold **development validation**，没有打开 AsEP official test。因此不能把 FrameContact 的当前 validation 数字和 EpiFormer/LF3DRoPE 的 AsEP test 数字直接做 SOTA 排名。

## SAbDab2 可审计结果

所有阈值依照各自冻结 receipt：EpiFormer 为 0.3；SPPIDER 为 validation-selected threshold；MIPE 为每折 validation-fitted threshold。Precision/Recall/F1/MCC 是 residue-level micro 指标；AUPRC/AUROC 是 ranking 指标。

| 方法 / 目标 | 数据范围 | AUROC | AUPRC | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| **EpiFormer / epitope** | SAbDab2 frozen test, 455 complexes | 0.8008 | 0.2775 | 0.2577 | 0.4785 | 0.3350 | 0.2715 |
| MIPE / epitope | official 5-fold held-out, epoch-800 | 0.7919 | 0.2514 | 0.2111 | 0.5152 | 0.2995 | 0.2556 |
| MIPE / paratope | official 5-fold held-out, epoch-800 | 0.8433 | 0.6215 | 0.5190 | 0.7281 | 0.6060 | 0.4621 |
| MIPE / epitope | official 5-fold held-out, validation-best | 0.8074 | 0.2804 | 0.2246 | 0.5523 | 0.3194 | 0.2808 |
| MIPE / paratope | official 5-fold held-out, validation-best | 0.8811 | 0.6892 | 0.5586 | 0.7954 | 0.6563 | 0.5346 |
| **SPPIDER-seq / epitope** | SAbDab2 frozen test, 455 complexes | 0.6718 | 0.1454 | 0.1309 | 0.5359 | 0.2104 | 0.1269 |
| **SPPIDER-seq / paratope** | SAbDab2 frozen test, 455 complexes | 0.9486 | 0.6511 | 0.5693 | 0.7313 | 0.6402 | 0.6023 |

### EpiFormer 的 P/R 是如何得到的

EpiFormer 的 frozen score receipt 存储了 micro F1、MCC、阳性数和总 residue 数，但没有单独存储 P/R 或四格表。由这四个已冻结量反解得到唯一整数 confusion matrix：

```text
TP = 3813, FP = 10981, FN = 4155, TN = 78303
Precision = TP / (TP + FP) = 0.2577396242
Recall    = TP / (TP + FN) = 0.4785391566
F1        = 0.3350320700
MCC       = 0.2715204002
```

这不是用 test label 重新选阈值；阈值 0.3 和 checkpoint 已经在 formal provenance 中冻结。`results/epiformer/score.json` 还明确说明该输出是 epitope residue projection，不是 native pair-interaction score，所以不能把 EpiFormer 的这张表误写成 contact-map 结果，也不能补写 paratope 数字。

### EpiFormer 与 SPPIDER 的同 test paired audit

两份 epitope prediction parquet 的 `instance_id` 集合完全相同：455 个 complex、97,252 个 residue rows。用各自冻结阈值做逐 complex 指标，再进行 10,000 次 complex-level paired bootstrap（seed 20260806），得到：

| 指标（macro per complex） | EpiFormer | SPPIDER-seq | Δ(EpiFormer−SPPIDER) | 95% paired bootstrap CI | EpiFormer 更好比例 |
|---|---:|---:|---:|---:|---:|
| AUPRC | 0.3480 | 0.2034 | +0.1446 | [0.1264, 0.1632] | 76.9% |
| AUROC | 0.7677 | 0.6337 | +0.1340 | [0.1172, 0.1505] | 79.1% |
| F1 | 0.3117 | 0.2151 | +0.0966 | [0.0797, 0.1132] | 69.9% |
| MCC | 0.2351 | 0.0988 | +0.1362 | [0.1144, 0.1580] | 70.5% |

该表是 frozen-test 的描述性 paired comparison；它没有用于选择模型、checkpoint 或阈值。由于 EpiFormer 是 epitope-only，而 SPPIDER-seq 是 sequence-only 的双角色 residue predictor，两者的 epitope 行可以直接对照，paratope 行不能对照 EpiFormer。

## 论文公开的 EpiFormer / LF3DRoPE 数字：不要与 SAbDab2 混称

### EpiFormer 论文的 AsEP 结果

EpiFormer 论文在 AsEP 上报告：

| AsEP split | AUROC | AUPRC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| epitope-ratio（3 seeds，mean±std） | 0.924±0.003 | 0.493±0.012 | 0.363±0.014 | 0.720±0.012 | 0.482±0.011 | 0.464±0.009 |
| epitope-group | 0.826 | 0.290 | 0.196 | 0.698 | 0.305 | 0.290 |

论文另有一个外部 SAbDab（494 complexes，非本地 SAbDab2 split）的 cross-dataset 表，只公开 AUC/F1：

```text
zero-shot: AUC 0.782, F1 0.297
LODO fine-tuning: AUC 0.858, F1 0.364
```

所以老师问的“EpiFormer 在 SAbDab2 test 的 Precision/Recall/F1/MCC”应使用本地 `results/epiformer/score.json` 的 formal SAbDab2 receipt，而不是把论文的 SAbDab(494) AUC/F1 当成 SAbDab2，也不能从论文外部表虚构 P/R/MCC。

### LF3DRoPE（Inter-Residue Geometry Attention）

LF3DRoPE 的论文使用 AsEP 官方 PEPNet split，每个 split 为 1,383 train / 170 validation / 170 test，3 seeds，threshold 在 validation 上按 MCC 选择后固定到 test。公开数字为：

| AsEP split | MCC | Precision | Recall | AUROC | F1 |
|---|---:|---:|---:|---:|---:|
| epitope-ratio | 0.410±0.008 | 0.469±0.022 | 0.428±0.012 | 0.846±0.003 | 0.447±0.006 |
| epitope-group | 0.171±0.010 | 0.248±0.015 | 0.189±0.003 | 0.690±0.011 | 0.215±0.008 |

LF3DRoPE 的核心是把 N/Cα/C 构造的 residue-local frame 中的相对位移直接作为 rotary attention phase；它的 antibody/antigen encoder 参数不共享，cross-attention、global antibody context 和 pair-MIL head 在编码之后加入。这个结构与我们当前的 geometry self-attention 方向高度相关，但它仍然是 **AsEP test**，不是 SAbDab2 test。

## 对老师提出的“公平比较”应采用的协议

当前 FrameContact 只有 antigen-fold validation，因此下一步不能直接拿它和上面的 AsEP test 表并排宣布提升。正式比较应按下面的顺序冻结：

1. **同一 benchmark**：使用 AsEP 官方 ratio/group split 和官方 surface/CDR 预处理；记录 complex 数、unique antigen/antibody、surface/CDR 长度分布。
2. **同一标签定义**：4.5 Å residue-level epitope/contact 定义；显式记录缺失坐标和 residue mapping mask。
3. **同一选择规则**：checkpoint、threshold 只在 validation 上选；test 只在协议冻结后一次性打开。
4. **同一报告口径**：MCC 作为主指标，同时报告 AUROC、AUPRC、Precision、Recall、F1；另报 per-complex macro、pooled micro 和 calibration。
5. **同一随机性口径**：至少 3 个 seed（建议沿用 FrameContact 的 5 seeds），报告 mean±std 或 complex-level bootstrap CI。
6. **同一输入审计**：EpiFormer/LF3DRoPE 用的 PLM、surface/CDR 子集、N/Cα/C frame、坐标扰动和 cross-attention 位置全部登记；native inter-chain geometry 不能偷偷混入只允许单体坐标的条件。
7. **FrameContact 仍保留 SAbDab2 antigen-fold**：它作为结构新颖性/外部泛化分析，不与 AsEP official test 混成一个 leaderboard 数字。

## 当前 FrameContact 的边界

`framecontact/reports/decision-v1-20260802/DECISION_REPORT.md` 明确写的是 development validation only；当前 matched epoch 21 的代表性行是：

| Arm | Contact AUPRC | Epitope AUPRC | Paratope AUPRC | Partner margin |
|---|---:|---:|---:|---:|
| framecontact_distogram | 0.0107 | 0.1255 | 0.6556 | +0.0037 |
| framecontact_unary_only | 0.0107 | 0.1224 | 0.6848 | −0.0039 |

这些数字没有读取 SAbDab2 frozen test 或 AsEP official test，不能与上面论文 test 数字直接比较。下一次 AsEP 开启前，应先冻结 split、metrics、threshold 和 checkpoint policy，再统一运行 FrameContact、EpiFormer-compatible baseline 与 LF3DRoPE-compatible control。

## 可复核入口

在仓库根目录执行：

```bash
python3 scripts/audit_baseline_metrics.py
python3 scripts/compare_epiformer_sppider_sabdab2.py
python3 -m pytest -q tests/test_audit_baseline_metrics.py
```

核心 receipts：

```text
results/epiformer/score.json
results/epiformer/provenance.json
results/epiformer/preflight.json
results/epiformer/verification.json
mipe-sppider-handoff/.../MIPE_HELDOUT_EVALUATION.json
mipe-sppider-handoff/.../SPPIDER/results/final_v3_ensemble_seed42/{epitope,paratope}/final/score.json
framecontact/reports/decision-v1-20260802/DECISION_REPORT.md
```

## 可直接发给老师的汇报

> 老师您好，我把 EpiFormer 和另外两个 baseline 的 test/held-out 指标按来源重新核对并整理好了。EpiFormer 在我们 SAbDab2 frozen test 上已经有 formal 结果：455/455 个 complex 完成映射，threshold=0.3，epitope micro AUROC/AUPRC/F1/MCC = 0.8008/0.2775/0.3350/0.2715；由 frozen receipt 中的 F1、MCC、阳性数和总 residue 数反解得到 Precision=0.2577、Recall=0.4785（TP=3813、FP=10981、FN=4155、TN=78303）。EpiFormer 当前只有 epitope projection，没有 paratope head，也没有 native pair-contact 输出，所以不会补写 paratope 或 contact 指标。
>
> 在相同 455 个 SAbDab2 test complexes 上，SPPIDER-seq 的 epitope AUROC/AUPRC/Precision/Recall/F1/MCC = 0.6718/0.1454/0.1309/0.5359/0.2104/0.1269，paratope = 0.9486/0.6511/0.5693/0.7313/0.6402/0.6023。逐 complex paired bootstrap 也支持 EpiFormer epitope 的提升：AUPRC +0.1446（95% CI 0.1264–0.1632），F1 +0.0966（0.0797–0.1132），MCC +0.1362（0.1144–0.1580）。MIPE 的结果是官方五折 held-out（560 complexes），epoch-800 的 epitope/paratope AUROC/AUPRC/Precision/Recall/F1/MCC 分别为 0.7919/0.2514/0.2111/0.5152/0.2995/0.2556 和 0.8433/0.6215/0.5190/0.7281/0.6060/0.4621；validation-best 为 0.8074/0.2804/0.2246/0.5523/0.3194/0.2808 和 0.8811/0.6892/0.5586/0.7954/0.6563/0.5346。
>
> 关于 Inter-Residue Geometry Attention（LF3DRoPE），论文是在 AsEP 官方 ratio/group test 上评测，不是 SAbDab2；公开 ratio/group 的 MCC 分别为 0.410±0.008/0.171±0.010，F1 为 0.447±0.006/0.215±0.008，并且报告了对应的 Precision、Recall 和 AUROC。我们当前 FrameContact 的结果还是 antigen-fold validation，所以我会先按 AsEP 官方 split、相同 label/mapping、validation-only checkpoint/threshold 和至少 3 个 seed 做一次公平比较，再把 SAbDab2 antigen-fold 作为结构新颖性泛化结果单独汇报。
