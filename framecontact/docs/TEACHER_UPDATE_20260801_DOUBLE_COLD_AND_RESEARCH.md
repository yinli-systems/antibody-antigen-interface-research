老师您好，我又根据最新研究和 SAbDab2 官方数据说明补做了一轮
split/partner-specificity 审计，先同步一个不含测试结果的进展。

当前正在跑的 antigen-fold v3 已经保证 PDB assembly、抗原/抗体精确序列
以及 Foldseek reciprocal-min TM-score >=0.50 的 hard edge 跨 train/validation/test
均为 0；但我进一步核对 SAbDab2 官方 cluster metadata 后发现，v3 仍有 8 个
antigen sequence cluster（涉及 240 个样本）和 11 个 antibody cluster（涉及
64 个样本）跨 role。因此它适合继续完成当前架构 development，但不足以单独
支撑最严格的 double-cold 泛化结论。

我已经在不读取 task label 或 prediction 的前提下重新构建了一个 confirmatory
split，把以下关系全部并入 connected component：

1. Foldseek reciprocal-min TM-score >=0.50；
2. exact PDB assembly、antigen sequence 和 antibody sequence；
3. SAbDab2 官方 `agclusters` 中每一条 antigen chain 的 cluster；
4. SAbDab2 官方 antibody-only final cluster。

这里还修正了一个容易忽略的细节：官方 `agclusters` 是用 `/` 分隔的多链
cluster 列表，不能把整串当成一个 ID。最终 token-level audit 中，上述所有
authority 的跨 role overlap 都为 0，role 数量为 2584/540/539。随后冻结的
最终冻结的 `double-cold-source-v13` 把代码、协议、测试以及测试所需的
evaluation/receipt/fixture authority 一并纳入清单，共认证 498 个文件。远端隔离
依赖环境在禁写 bytecode、禁 pytest cache 的条件下完整回归为 284 passed、
5 个仅因 pinned Boltz source 未提供而 skip、0 failed；测试后再次验证 manifest
也通过；本地含 pinned Boltz source 的完整回归为 289 passed、0 failed。源码树 SHA-256 为
`8ced33fbf6cdd7747fec264faeedf88bb0af27e913c23d4a924ff4d735b12de1`。

新 split 的 test role 目前没有生成 partner panel，也没有读取 label 或模型输出。
我只冻结了 train 和 validation 的双向 reciprocal-31 wrong-partner panels。等当前
五 seed 架构消融完成后，我会先用 validation-only 固定 winning architecture、
contact marginal/head、calibration 和 checkpoint rule，再在新 split 上从头训练，
最后一次性开启 539 个 test complexes，避免用更严格 split 的 test 反向选模型。

这轮最新 research 对实验设计有三个直接影响：

- 2026 年 7 月的 germline-shortcut 工作显示，random split 可使免疫受体 top-1
  retrieval 虚高 15--28 个百分点，因此完整序列 PLM 的提升不能自动解释成
  antigen specificity；
- 另一项对 106 个 cognate complexes 和 11,342 个 shuffled pair 的研究显示，
  AF3/Boltz-2/Chai-1 的结构 confidence 经常不能区分真实和错误 partner，因此
  wrong-partner ranking 需要成为核心指标，而不是附加 control；
- ICLR 2026 RaftPPI 表明，confidence-weighted hard negatives 和 retrieval Recall@K
  很适合大规模筛选。我会把这个思路用于 K=1 vs K=4 的 validation-only listwise
  specificity 消融，同时设置 contact AUPRC、precision/recall 和 calibration 的
  non-degradation gate，防止模型只会做 partner ranking 却损失 contact localization。

需要特别说明的是，目前这些新增结果证明的是 split、provenance 和评估协议更严格，
还不是模型在 frozen test 上提升。当前五 seed 训练保持原 source/protocol 继续运行，
我没有中途修改或重启。最终我会分别汇报 development、double-cold frozen test、
temporal 和官方 AsEP，且不会把不同 cutoff/split 的数字混成一个 SOTA 比较。
