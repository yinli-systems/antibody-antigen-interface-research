# 给老师的阶段汇报（2026-08-01，开发集运行中）

老师您好，目前 residue-head 和 distogram 两组五种子开发实验都在正常运行，
没有读取 frozen test 或 external test。最新只读快照中，`unary-only` 五个种子
已同步到 epoch 18；`distogram + gated hybrid` 五个种子已同步到 epoch 15，
其中一个种子到 epoch 16。每个种子仍严格按照 validation 上较差方向的
partner-retrieval margin 选择 checkpoint，contact mean per-complex AUPRC 只作
tie-break，没有根据当前展示的某一个 residue metric 提前选模型。

在五个相同种子的同步 epoch 15，把 `distogram + gated hybrid` 暂时与
`binary contact + unary-only` 对照时，开发集上的平均变化为：

- contact mean per-complex AUPRC：`+0.00006`，基本持平；
- contact pooled AUPRC：`+0.00031`；
- 固定 raw-logit=0 的 contact precision：`+0.00184`，5/5 个种子为正；
- 固定阈值 contact recall：`-0.08294`；
- epitope pooled AUPRC：`+0.00050`，但 precision `+0.00853`、recall
  `-0.08106`；
- paratope pooled AUPRC：`-0.03519`，precision `-0.05464`、recall
  `+0.02704`；
- 双方向中较差的 partner-specific margin：`+0.02141`。

其中 partner margin 在 4/5 个种子上为正，leave-one-seed-out 的平均增益范围
为 `+0.01142` 到 `+0.02896`；但 exact sign-flip `p=0.1875`，五个种子下证据
仍不足。contact precision 在 5/5 个种子上上升、paratope AUPRC 及 precision
在 5/5 个种子上下降、epitope recall 也在 5/5 个种子上下降；这些项目的双侧
exact sign-flip `p=0.0625`，已经是五个种子能够达到的最小 p 值。整体上更像是
distogram 改变了 precision/recall 与 pair/residue 之间的权衡，而不是已经形成
稳定的全面提升。

这里我暂时不会把结果解释成“distogram 已经优于 binary contact”，因为上述
对照同时改变了 pair head 和 residue aggregation，存在混杂。真正对应您问题的
无混杂比较需要等排队中的 `contact-marginal` 和 `contact-hybrid` 五种子完成，
再将 distogram 与相同 gated-hybrid residue head 的 binary-contact 条件比较。
目前的现象反而支持您提出的判断：contact 表示可能改善 pair-level reasoning，
但如果 residue marginalization/gating 处理不好，会牺牲尤其是 paratope 的
预测质量和 recall。

我还审计了 2026 年最新的 Handshake 工作。它使用 ProstT5/LoRA、双向
cross-chain attention 和 contact supervision，是很有价值的 partner-aware
sequence-only 强基线。不过官方代码当前没有许可证和公开 checkpoint，默认
随机切分也不满足我们的 endpoint-disjoint/antigen-fold 要求；其 contact matrix
导出还会临时切到 `train()`，可能引入 dropout 随机性。因此我不会直接复制代码
或横向比较论文数字，而会独立实现一个 deterministic 的
`ProstT5-cross-contact` control，在完全相同的 SAbDab2 double-cold split、五个
种子、contact 定义和 reciprocal partner panel 上重训。

下一步顺序保持不变：

1. 等当前所有开发条件完成，只用 validation 冻结 residue aggregation、pair
   factorization 和 checkpoint；
2. 在已完成审计的 SAbDab2 double-cold split 上，从头重训选中模型以及
   partner-aware sequence-only 强基线；
3. 同时报告 pair contact、epitope/paratope、precision/recall、calibration 和
   双向 partner retrieval；
4. 只有模型与阈值全部冻结后，才一次性运行 frozen test 和 external benchmark。

当前结果是有机制信号、但尚未完成模型选择的开发证据；还不能表述为 external
generalization 或 benchmark improvement。

补充工程证据：为下一阶段新增的 partner-aware geometry-ablated binary/distogram
controls 和最新强基线审计已冻结到 `double-cold-source-v13`。远端独立依赖目录
补齐 `pyarrow==17.0.0` 后，在禁写 bytecode、禁 pytest cache 并执行测试后
manifest 复验的条件下为 `284 passed, 5 skipped, 0 failed`；5 项 skip 均为未部署
pinned Boltz source。本地含 pinned Boltz source 的完整回归为 `289 passed, 0 failed`。
源码及测试 authority 清单含 498 个文件，tree SHA-256 为
`8ced33fbf6cdd7747fec264faeedf88bb0af27e913c23d4a924ff4d735b12de1`。
这只证明实现、来源和数据协议可复现，不代表生物学性能。
