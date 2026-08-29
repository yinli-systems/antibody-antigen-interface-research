# 给老师的更新：contact bottleneck 与 matched epoch 17

## 建议直接发送的版本

老师您好，我又完成了一次五个 seed 的只读同步审计。目前 unary-only 已同步到
epoch 20，distogram 已同步到 epoch 17；其他 contact-marginal、contact-hybrid
和 distogram-triangle 条件还在队列中，现有任务没有被重启或改动。

在 matched epoch 17，distogram 相对 unary-only 的结果是混合的：pooled contact
AUPRC 约 `+0.00030`，固定 raw-logit 阈值下 contact precision 约 `+0.00113`，
pooled epitope AUPRC 约 `+0.00488`；但 worst-direction partner margin 变为
`-0.00535`，contact recall 约 `-0.03478`，epitope recall 约 `-0.08758`，
paratope AUPRC 约 `-0.03530`，并且 paratope AUPRC 在 `5/5` 个 seed 都下降。
所以之前 epoch 15 的正向 partner-margin 信号没有保持，我不会据此提前选择
distogram。当前比较同时改变了 pair head 和 residue aggregation，必须等
contact-hybrid 跑起来后再做可解释的主效应比较。

针对 partner specificity，我把一个新的 contact-bottleneck 候选实现并完成了
工程回归。它不是直接对全链 embedding 做检索，而是从 pre-cross contact head
计算 counterfactual contact lift：总 contact probability 减去移除 pair interaction
后的 probability，只对 partner-specific 的正向接触增量做 noisy-OR，再池化两个
单体 self-encoder 的 residue representation。这样纯 unary interface propensity
会得到零 lift，retrieval 必须经过 pair-specific contact attribution，不能直接绕过
localization。

我把 stop-gradient 和 joint-gradient 两条路径显式分开：前者检验现有 contact
map 是否已经包含 specificity，后者检验 retrieval loss 会改善还是扭曲 contact
map。该模块默认关闭，只允许 late fusion；如果用于 interleaved cross-attention
会直接报错，因为那时的特征已经看过 partner，不能再称为独立 monomer self
feature。当前 v17 本地全量测试为 `298/298`，远端隔离环境为
`293 passed、5 skipped、0 failed`，5 项 skip 仅来自未部署的可选 Boltz
baseline；同时已加入 zero-lift、weight entropy、effective support 和 top-1
mass 诊断，用于识别 dead 或近似全链均匀 pooling。

这部分目前只有实现和测试，还没有启动新 GPU 训练，也没有 frozen/external test
结果，因此我不会汇报任何性能提升。下一步我会先等现有队列按冻结协议完成，
用 contact-hybrid 拆清楚 distogram 与 residue aggregation 的贡献；只有 base contact
模型通过 contact AUPRC、precision/recall、双向 partner retrieval 和 calibration
门槛后，才在 validation-only 上比较 contact-bottleneck 的 stop-gradient 与 joint
版本。当前任务实际运行在 RTX 4090，不会表述成 A100 结果。

## 更短的飞书版本

老师您好，我做了新的五 seed 同步审计：unary-only 已到 matched epoch 20，
distogram 到 matched epoch 17。distogram 的 contact precision 和 epitope AUPRC
有小幅提升，但 epoch 15 的 partner-margin 增益没有保持；到 epoch 17，
worst-direction margin 约下降 `0.00535`，epitope recall 下降 `0.0876`，paratope
AUPRC 下降 `0.0353`，且后者在 `5/5` seeds 都下降。因此我不会提前把 distogram
定为胜出，仍需等 contact-hybrid 启动后拆分 pair head 与 residue aggregation 的
贡献。

另外我已把 contact-bottleneck specificity 候选实现并完成回归：用 pre-cross
contact probability 的 pair-interaction counterfactual lift 池化两个单体 encoder
的 residue features，再做双向 wrong-partner ranking，并显式比较 stop-gradient 与
joint-gradient。模块默认关闭、只允许 late fusion；新的 source authority 会与未训练
边界一起单独冻结。
目前没有启动该候选训练，也没有访问 frozen/external test，所以现在只汇报方法和
工程证据，不汇报性能提升。

## 证据边界

- matched-epoch 数字来自旧 v3.1 development split 的在途日志，不是 final model
  selection，也不代表 double-cold 或 external generalization；
- 精确五 seed sign-flip 检验的最小双侧 p 值为 `0.0625`；多数差值并不显著；
- contact-bottleneck 尚未注册实验条件，尚无 GPU 训练或质量指标；
- frozen SAbDab2 test 与 official AsEP test 均未因本次工作开启；
- 实际运行硬件为 RTX 4090，不是 A100。
