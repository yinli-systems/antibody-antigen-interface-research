# 给老师的阶段性汇报（2026-08-01）

老师您好，我补充检查了 precision / recall 随 epoch 的变化，并把
unary-only 与 distogram 做了同 epoch、同五个 seed 的配对比较。目前
unary-only 已同步到 epoch 14，distogram 已同步到 epoch 11；其他
contact-marginal、hybrid 和 triangle 条件尚未开始，因此以下都只是
development 中途诊断，不是最终模型选择，也没有读取 frozen test。

在双方都完成的 epoch 11 上，distogram 相对 unary-only：

- pair contact pooled AUPRC 从 0.01186 提升到 0.01262；固定阈值
  precision 从 0.01213 提升到 0.01324，但 recall 从 0.4216 降到
  0.4120；
- epitope precision 从 0.1266 提升到 0.1386，但 recall 从 0.5716
  降到 0.4727，pooled AUPRC 从 0.12848 降到 0.12639；
- paratope pooled AUPRC 从 0.69024 降到 0.67756，precision 和 recall
  也分别下降约 0.0119；
- partner retrieval 仍然是混合结果：固定 antibody、检索 antigen 的
  R@1 提升约 0.0040；固定 antigen、检索 antibody 的 R@1 下降约
  0.0091；worst-direction margin 提升约 0.0067。

因此现在更严格的结论是：distogram 暂时让 contact/epitope 输出更
稀疏、precision 更高，但牺牲了 recall；同时它改善了注册的最差方向
margin，却没有同步改善两个方向的 R@1，所以目前还不能判定它优于
unary-only，更不能做 frozen-test 结论。

我进一步审计后发现一个训练—评估不一致：当前训练每个方向、每个
epoch 只采一个 wrong partner，用 pairwise softplus；而 validation 是在
31 个候选 partner 中做双向排名，并关注较差方向。因此 margin 与 R@1
不同步是有可能的。

为此我实现了一个不改变现有 running arrays 的 post-freeze 候选：每个
方向同时采 K=4 个确定性、无重复 wrong partners，用 listwise ranking
loss，并以 smooth-max 强调较差方向。温度固定为 T=1；这样 K=1 时它与
旧 pairwise softplus 在数值和梯度上完全等价，避免因为温度改变而把
specificity loss 的梯度偷偷放大。

这条路径目前完成了 269 项回归测试，并在真实 SAbDab2 train-role 的
最长 antigen 样本（2363 aa）上完成 RTX 4090 BF16 前向/反向。通过只对
wrong-partner branches 做 activation checkpointing，K=4 的 Slurm
MaxRSS 约 1.95 GiB、peak allocated CUDA memory 约 10.46 GB；没有做
optimizer step，也没有读取 validation/test，因此这只是可训练性证据，
不是性能提升。

下一步我会先等当前五条件 × 五 seed 全部结束，并只用 development
冻结 base architecture；随后在相同架构下做旧 K=1 pairwise 与预先
冻结的 K=4 listwise 五-seed 对照。主检验会同时看两个方向的 R@1/MRR
和 worst-direction margin，并把 contact AUPRC、P@K、epitope/paratope
AUPRC/F1/MCC、precision/recall 和 calibration 设为 non-degradation
gate。只有这些选择全部冻结后，才会一次性开启 frozen/external test。
