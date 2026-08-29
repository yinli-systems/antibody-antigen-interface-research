# 给老师的阶段汇报（development-only，2026-08-01）

老师您好，我已经把您建议重点关注的 precision、recall、F1 和 MCC
补成了逐 epoch、五 seed 同步的监控，同时保留 validation-selected threshold
和固定 raw-logit-zero 两套口径。目前两条冻结训练阵列仍在运行，下面只是
development train/validation 快照，不是最终 checkpoint 比较，也没有打开 frozen
test 或 external test。

为了避免用不同 epoch 比较，我取两边五个 seed 都存在的最新共同 epoch 10
做了 paired comparison。unary-only 的 contact
precision/recall/F1/MCC 为
0.01927/0.07445/0.03051/0.03572；distogram 为
0.02163/0.08980/0.03391/0.04108。对应 paired delta 为
+0.00236/+0.01536/+0.00340/+0.00536，其中 precision 和 F1 是 5/5 seed
同方向，recall 和 MCC 是 3/5 seed 同方向。与此同时，distogram 的 P@10
下降 0.00270、P@Lmin/5 下降 0.00200，paratope pooled AUPRC 下降
0.01304，所以目前还不能判断 distogram 整体获胜。

逐 epoch 看，两种模型的 precision 和 recall 并没有同时持续提高。以固定
raw-logit-zero 阈值为例，unary 的 precision 从 0.00708 提高到 0.01132，
但 recall 从 0.90873 降到 0.43462；distogram 的 precision 从 0.00867
提高到 0.01407，recall 从 0.83818 降到 0.31615。说明训练过程中主要发生了
score sharpening 和 precision--recall operating-point trade-off，不能只看某个
epoch 的 precision 上升就认为 contact ranking 整体改善。

我也修正了 partner-specificity 的汇报口径。冻结协议真正用于选择的是每个
seed 的“两个检索方向中较差的 margin”，再对五 seed 求均值。在共同 epoch
10 上，unary 为 -0.01781，distogram 为 +0.00152，paired delta 为
+0.01933，4/5 seed 为正，但其 seed 间 SD 为 0.03636；而两个检索方向的
R@1 平均分别变化 -0.00109 和 -0.01931。因此 margin 有积极信号但方差较大，
而 retrieval accuracy 尚未改善，不能据此声称模型已经学到稳定的 partner
specificity。之前看到的较大数字是两方向 margin 的平均值，不能代替更严格的
worst-direction criterion。两个方向的 R@1 和 MRR 现在也分别记录，后续不会
只汇报平均 margin。

根据最新文献，我又补充了三个直接相关的对照方向：Cell 2026 的 MSA
Pairformer、2026 年 7 月的 LC-SEPLM，以及 ABConformer。MSA Pairformer
说明单链预训练的 sequence--pair refinement 和 triangle operations 可以迁移到
PPI contact，但它使用 MSA，信息量和预处理成本都高于 frozen ESM，不能当作
同预算 drop-in baseline。LC-SEPLM 支持用 long-range contact supervision
做 ESM LoRA，不过目前没有直接的 partner-specific PPI 证据，而且外部
AlphaFold/Swiss-Prot 预训练必须先做时间和重叠审计。ABConformer 的实验也
显示局部带宽会提高 precision、降低 recall；它使用的是学习到的序列伪位置，
不是未对齐两个单体坐标系下的真实三维距离。

所以当前阵列完成并冻结选择后，我准备做一个 2 x 2 因果消融：frozen ESM
和 training-only contact-adapted ESM，分别配 unary head 和同一个
FrameContact pair trunk。这样可以区分收益究竟来自更好的单链结构先验，还是
来自显式 partner interaction。MSA Pairformer 会作为信息更丰富的上界单独
报告，并先完成官方 checkpoint、MSA 构建、许可和去污染审计。

下一步仍按原协议执行：等待 unary/marginal/hybrid/distogram/triangle 五个条件
全部完成五 seed，使用 validation-only 规则冻结模型；只有冻结后才一次性运行
允许的 held-out/external evaluation。当前没有使用 test 结果选择模型。
