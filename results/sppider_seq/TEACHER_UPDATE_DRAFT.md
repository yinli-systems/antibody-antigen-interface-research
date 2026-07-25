# 给老师的进度回复（可直接发送）

老师您好，SPPIDER-seq 在 SAbDab2 上的训练和冻结测试已经完成。我这次不只是按默认参数跑了一遍，还补齐了比较严格的实验设计和核验：

1. 重新做了 cluster-aware 的 train/validation/test 划分，train、validation、test 之间 canonical cluster 和 focal query sequence overlap 都为 0；最终为 2,727/481/455 个 instances。
2. 比较了 scratch、两个官方 checkpoint initialization、多个学习率和 weight decay，并在 strict validation 上固定了 50:50 ensemble 和 decision threshold；又用 seeds 17/42/73 检查了 recipe 的稳定性。
3. 最终 frozen test 上，paratope 的 residue-level micro AUROC/AUPRC 为 **0.9486/0.6511**，F1/MCC 为 **0.6402/0.6023**；epitope 为 **0.6718/0.1454**，F1/MCC 为 **0.2104/0.1269**。
4. 对 84 个 test clusters 做了 1,000 次 cluster bootstrap。相对之前版本，paratope AUPRC 提升 **0.0377**（95% CI 0.0170–0.0588），epitope AUPRC 提升 **0.0304**（95% CI 0.0130–0.0439），提升在配对重采样下仍然稳定。
5. 做了 MMseqs2 train–test similarity audit：抗体侧同源性较高，但 test AUPRC 与 nearest-train identity 没有显著相关；抗原侧在 identity ≥0.5、双向 coverage ≥0.8 时没有 train neighbor，因此 epitope 更接近 remote generalization，也更难。
6. 也按您的要求检查了 partner-specificity：打乱 partner 后，paratope AUPRC 从 0.6511 降到 0.6392，epitope 从 0.1454 降到 0.1425。说明模型有少量 partner 信号，但整体仍主要是 marginal interface propensity。

最终 recipe 和 threshold 都是在 strict validation 上锁定后才运行 `final_v3`，评估后没有再按 test 指标选模型。不过该 benchmark 更早已经用于 zero-shot 和旧版本评估，因此我把它定位为固定 benchmark 上的严格适配研究，不表述为首次开启的 one-shot blind test。

需要特别说明：SPPIDER-seq 原生只输出 paratope/epitope residue probabilities，并不输出 epitope–paratope residue-pair interaction matrix，所以我没有把它的 projection 当作 native pair prediction。相同 frozen SAbDab2 上，结构辅助 EpiFormer 的 epitope AUROC/AUPRC 是 0.8008/0.2775，仍明显高于 sequence-only SPPIDER-seq；如果后续重点是黄色方框里的 pair-interaction matrix，我建议主线继续放在 MIPE/EpiFormer 或增加显式 pairwise decoder。

所有结果都已经保存到：

```text
/cluster/project/model_deployment/results/sabdab2/sppider_seq/final_v3_ensemble_seed42
```

其中包括 score、完整 predictions、provenance、逐行 key verification、cluster-bootstrap JSON，以及 ROC/PR 的 PNG/PDF 图。两个官方 checkpoint 的严格加载和 11 项自动测试也全部通过。
