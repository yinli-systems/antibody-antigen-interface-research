# SPPIDER-seq × SAbDab2 最终研究报告

状态日期：2026-07-23
最终 run：`final_v3_ensemble_seed42`
结论：训练、冻结测试、统计检验、partner control、相似性审计和交付核验均已完成。

## 1. 结论摘要

这次工作已经从“把官方代码跑起来”提升到一套可复核的 benchmark study。最终模型在 frozen SAbDab2 的 paratope 预测上表现强，micro AUROC/AUPRC 为 **0.9486/0.6511**；epitope 为 **0.6718/0.1454**，明显优于最初版本，但仍属于中等偏弱的序列级结果。

最重要的科学边界是：SPPIDER-seq 原生输出的是 paratope/epitope 的**边缘 residue probability**，不是 antibody-residue × antigen-residue 的 interaction matrix。partner permutation 表明模型只利用了较弱的配对信息，因此不能把当前输出描述成 residue-pair interaction prediction。

## 2. 最终测试结果

决策阈值只用 strict validation set 选择，然后锁定后一次性用于 frozen test。AUROC/AUPRC 与阈值无关；F1/MCC 使用 validation MCC-optimal threshold。

| Role | Residues | Threshold | AUROC (95% cluster CI) | AUPRC (95% cluster CI) | F1 (95% cluster CI) | MCC (95% cluster CI) |
|---|---:|---:|---:|---:|---:|---:|
| Paratope | 89,938 | 0.32534 | 0.9486 (0.9417–0.9537) | 0.6511 (0.6269–0.6783) | 0.6402 (0.6242–0.6567) | 0.6023 (0.5836–0.6202) |
| Epitope | 97,252 | 0.08231 | 0.6718 (0.6258–0.7081) | 0.1454 (0.1213–0.1668) | 0.2104 (0.1785–0.2381) | 0.1269 (0.0847–0.1620) |

区间来自对 84 个 canonical test clusters 做 1,000 次 cluster bootstrap，而不是把高度相关的 residues 当成独立样本。paratope 的 Brier/ECE 为 0.04955/0.00418；epitope 为 0.07399/0.02350。

## 3. 相对旧版本的真实提升

最终模型与旧 `trained_seed42` 在完全相同的 frozen test residues 上做 paired cluster bootstrap。下表是 final − previous；95% CI 全部高于 0。

| Role | ΔAUROC | ΔAUPRC | ΔF1 | ΔMCC |
|---|---:|---:|---:|---:|
| Paratope | +0.00573 (0.00290–0.00851) | +0.03767 (0.01702–0.05882) | +0.07490 (0.03587–0.11933) | +0.07357 (0.03861–0.10719) |
| Epitope | +0.06213 (0.02736–0.08937) | +0.03036 (0.01295–0.04389) | +0.18043 (0.14540–0.21449) | +0.09079 (0.04991–0.13197) |

epitope AUPRC 的绝对提升为 0.03036，相对旧值 0.11509 约提升 **26.4%**。这里最可信的是阈值无关的 AUROC/AUPRC；F1/MCC 的提升同时包含了更合理的 validation-only threshold policy。

## 4. 严格 split 与泄漏审计

- Frozen membership：3,208 train、455 test；train/test 有 274/84 个 canonical clusters，cluster overlap 为 0。
- Final development split：2,727 train、481 validation、455 test；cluster 数为 233/41/84。
- 两个 role 的 train/validation/test 之间，query sequence overlap 和 canonical cluster overlap 均为 0。
- 抗体和抗原都没有 train/test exact sequence overlap；完整 antibody–antigen sequence pair overlap 为 0。
- MMseqs2 审计发现抗体同源性仍高：test antibody 中 97.58% 在 train 中有 identity ≥0.7 且双向 coverage ≥0.8 的序列；56.48% ≥0.8；2.86% ≥0.9；0% ≥0.95。
- 抗原侧更严格：在 identity ≥0.5 且双向 coverage ≥0.8 时为 0%。这说明 epitope test 更接近 remote generalization，也解释了它明显更难。
- 在 455 个 paratope test instances 中，nearest-train identity 与 per-instance AUPRC 的 Spearman ρ = −0.0613，p = 0.192；没有证据表明最终 paratope 分数由最近邻 identity 单调驱动。

## 5. 训练选择与 seed 稳定性

严格 validation 上比较了 scratch、两个官方 checkpoint initialization、学习率和 weight decay。官方 checkpoint 并未在 SAbDab2 strict validation 上胜过 scratch，因此最终采用 validation 预先选定的两个 scratch models 做固定 50:50 ensemble：

- Paratope：`lr=1e-4` + `lr=1e-5`。
- Epitope：`lr=1e-4` + `lr=3e-5`。

同一固定 recipe 在 seeds 17/42/73 上的 strict validation 结果为：

| Role | AUROC mean ± sample SD | AUPRC mean ± sample SD |
|---|---:|---:|
| Paratope | 0.95874 ± 0.00059 | 0.67940 ± 0.00598 |
| Epitope | 0.73928 ± 0.00776 | 0.15415 ± 0.00683 |

seed 稳定性支持最终 recipe 不是只对 seed 42 偶然有效。最终 frozen test 只使用预先锁定的 seed 42 ensemble 和 thresholds，不在 test 上重新挑模型。

## 6. Partner-specificity control

测试时保留每条 focal sequence 和 label 不变，只对 partner sequence 做确定性无固定点置换，并使用同一模型与阈值：

| Role | Matched AUPRC | Permuted AUPRC | Matched − permuted (paired 95% CI) |
|---|---:|---:|---:|
| Paratope | 0.65109 | 0.63924 | +0.01185 (0.00021–0.02615) |
| Epitope | 0.14544 | 0.14253 | +0.00292 (−0.00049–0.00632) |

paratope AUPRC 存在小幅 partner benefit；epitope 的区间跨 0。AUROC、F1、MCC 的多数配对区间也跨 0。这说明最终模型主要捕捉 intrinsic interface propensity，只带有有限的 partner identity dependence。这是一个需要如实汇报的模型局限。

## 7. 与现有方法的合理比较

官方 SPPIDER-seq 论文使用自己的 receptor–peptide 数据与 split，报告的 blind-test AUROC/AUPRC 约为 receptor 0.790/0.212、peptide 0.797/0.205。论文明确排除了 antibody–epitope complexes，因此这些数字只能做外部背景，不能与本次 SAbDab2 指标直接排名。[SPPIDER-seq paper](https://academic.oup.com/bioinformatics/article/42/7/btag388/8707840)；[official repository](https://github.com/aporollo-lab/SPPIDER-seq)

在同一个 frozen SAbDab2 benchmark 上，结构辅助 EpiFormer 的 epitope marginal AUROC/AUPRC 为 **0.8008/0.2775**，显著高于 sequence-only SPPIDER-seq 的 0.6718/0.1454。这是目前最有意义的内部对照，也表明 antigen epitope prediction 对结构信息非常敏感。抗体专用方法如 ParaAntiProt 的论文结果来自不同数据和 split，同样不应与这里直接排序。[ParaAntiProt](https://www.nature.com/articles/s41598-024-80940-y)

## 8. 交付完整性

- 455/455 test instances 均完成。
- 每个 role 的 candidate projection 都有 18,473,412 行、455 个 row groups。
- 与 frozen candidate keys 逐行完全一致。
- null score = 0，non-finite score = 0。
- 两个官方 checkpoints 均 strict-load 成功。
- 11 项 contract/model tests 全部通过。
- 所有新代码 comments 和 docstrings 均为 English。

本地核心文件：

- `outputs/sppider_seq/final_research_summary.json`
- `outputs/sppider_seq/final_v3_ensemble_seed42/{paratope,epitope}/final/{score,verification,cluster_bootstrap}.json`
- `outputs/sppider_seq/research/paired_final_vs_old/*/paired_cluster_bootstrap.json`
- `outputs/sppider_seq/research/paired_matched_vs_permuted/*/paired_cluster_bootstrap.json`
- `outputs/sppider_seq/final_v3_ensemble_seed42/figures/final_test_curves.{png,pdf}`

并行云完整产物根目录：

```text
/cluster/project/model_deployment/results/sabdab2/sppider_seq/final_v3_ensemble_seed42
```

## 9. 最终评价

这份工作的含金量主要来自：官方架构复现、严格无 cluster/query 泄漏的验证设计、frozen test discipline、multi-seed robustness、cluster-level uncertainty、MMseqs2 相似性审计、partner permutation control，以及完整的预测键和 provenance 核验。它已经达到一份扎实 benchmark/reproduction study 的标准。

它还不是一个“顶级 pair-interaction 模型”结果。要回答老师图中黄色方框的 residue-pair matrix 问题，应继续使用 MIPE/EpiFormer 的 native interaction head，或对 SPPIDER-seq 增加显式 pairwise decoder，并采用 partner-aware hard negatives 和 antibody-aware clustering。
