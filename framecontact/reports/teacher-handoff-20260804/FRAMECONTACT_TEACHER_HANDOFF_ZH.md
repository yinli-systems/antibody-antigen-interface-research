# FrameContact 阶段性材料：公平基线、antigen-fold 统计与架构说明

更新时间：2026-08-04

## 0. 先给结论

1. 当前 FrameContact 数字来自 **SAbDab2 v3.1 antigen-fold validation**，不是 test，也不是 MIPE 官方数据集；因此不能把 MIPE 论文表格或此前 SPPIDER-seq test 数字直接放在同一列并宣称优劣。
2. 可以公平比较，但必须把 MIPE 或其他模型在当前 2,565/549 的 train/validation 角色上重新训练，并用 FrameContact 的同一 residue key、label、mask 和 evaluator 评分。
3. 当前最直接、已经同 split 同 evaluator 的强对照是 `sequence_dense`、`framecontact_unary_only`、late `framecontact`/`framecontact_contact_hybrid` 和 interleaved ICG 系列。
4. 当前 v3.1 validation 有 **549** 个 complex，包含 **314 个 exact antigen sequences、334 个 exact antibody sequences**；长度统计见下表。
5. 仓库中没有名为 `FrameContact_Contact` 的唯一条件。为避免概念漂移，本文用代码中的精确 condition name；如果口头上的 `FrameContact_Contact` 指 contact-derived residue head，则对应的是 `framecontact_contact_marginal` 或 `framecontact_contact_hybrid`，两者必须分开写。
6. 现有五 seed 开发集矩阵已经给出一个明确负结论：`distogram` 和局部 `triangle` 都没有通过预先冻结的 pair-contact promotion gate；它们会改变 precision/recall 或 residue 指标，但尚未改善主要 contact ranking 与 partner specificity。

## 1. SAbDab2 v3.1 antigen-fold 数据组成

统计权威：

- split：`CURRENT_V3_1_ANTIGEN_FOLD.csv`
- split SHA-256：`fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6`
- metadata SHA-256：`d7668d20a898a10569dc5fa0a4f0a888bd0f8636236a4c6cecc35d469307bd59`
- 完整机器可读统计：[`ANTIGEN_FOLD_V3_1_COHORT_SUMMARY.json`](ANTIGEN_FOLD_V3_1_COHORT_SUMMARY.json)

这里把“unique partner”拆成两个定义：

- exact sequence：序列 SHA-256 不同；
- metadata cluster：SAbDab2 元数据中的聚类 ID 不同。

不能只写“unique antigen=314”而不说明定义。

| role | complex | exact antigen seq. | exact antibody seq. | antigen cluster | antibody cluster | PDB assembly |
|---|---:|---:|---:|---:|---:|---:|
| train | 2,565 | 857 | 1,007 | 262 | 770 | 1,915 |
| validation | **549** | **314** | **334** | **108** | **250** | **305** |
| sealed test（仅组成统计，不做模型评测） | 549 | 236 | 267 | 66 | 184 | 328 |

validation 长度分布：

| partner | min | P05 | P25 | median | mean | P75 | P95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| antigen | 51 | 91 | 113 | **153** | 193.39 | 221 | 413.4 | 1,390 |
| antibody | 110 | 116 | 223 | **227** | 208.17 | 230 | 235 | 246 |

validation antigen 长度分桶：50–99 为 72 个，100–199 为 313 个，200–499 为 146 个，500–999 为 14 个，>999 为 4 个。549 个 antigen 都是 protein。antibody 中 445 个 conventional Fv，104 个 single-domain heavy chain。

重要边界：v3.1 保证 exact antigen sequence、exact antibody sequence 和 PDB assembly 跨角色为零重叠，但 metadata antigen/antibody cluster 仍分别有 8/11 个跨角色值。因此准确名称是当前 `antigen-fold v3.1`，不能误写成完全 metadata double-cold split。

## 2. 什么才叫“与 MIPE 公平比较”

### 2.1 论文数字只能作为背景

MIPE 使用 sequence、structure、多模态对比学习和两向 interaction matrix；官方仓库提供自己的 pickle 数据与 `cvdata.pkl`/`testdata.pkl` 入口，并使用 CC BY-NC 4.0 许可。[MIPE official repository](https://github.com/WangZhiwei9/MIPE)；[IJCAI 2024 paper](https://www.ijcai.org/proceedings/2024/0669.pdf)。后续研究描述的 MIPE benchmark 为 626 个 complex，并沿用 90% cross-validation + 10% independent test，而本项目当前 development split 是 3,114 个 SAbDab2 instances（2,565 train + 549 validation）的结构隔离 v3.1 split；另外 549 个 sealed-test instances 不参与本阶段特征物化或比较。[ParaSurf data description](https://pmc.ncbi.nlm.nih.gov/articles/PMC11855283/)

所以 MIPE 论文或此前复现数字与当前 FrameContact validation 至少在以下方面不相同：

- 样本集合；
- train/validation/test 切分；
- 去重与结构隔离规则；
- residue candidate mask；
- label/contact cutoff；
- pooled/macro 聚合方式；
- threshold 和 checkpoint selection。

这些数字可以列在“历史背景”表，不能列在“同协议 leaderboard”。

### 2.2 本项目的公平性六项门槛

一个 baseline 只有同时满足以下条件才进入主比较表：

1. 同一个 v3.1 split SHA-256；
2. 同一 2,565 train / 549 validation instance ID 集合；
3. 同一 residue ordering、observed/missing mask 和 label authority；
4. checkpoint、temperature、threshold 只使用 validation；
5. 同一 evaluator 同时输出 pooled 与 complex-macro AUPRC、AUROC、precision、recall、F1、MCC；
6. 报告失败、缺失样本和 seed，不允许只保留成功 run。

### 2.3 当前 baseline 的可比状态

| baseline | 当前是否可进同协议主表 | 原因/下一步 |
|---|---|---|
| `sequence_dense` | 是 | 与 FrameContact 同 loader、split、seed、evaluator；它是最重要的容量/ESM 对照 |
| `framecontact_unary_only` | 是 | 同 pair trunk，但 residue 输出不使用 contact marginal |
| `framecontact_contact_marginal` | 是 | residue 输出严格来自 contact map；必须与 hybrid 分开 |
| `framecontact_contact_hybrid` | 是 | unary 与 contact-derived score 用 zero-initialized gate 融合 |
| MIPE 官方/此前复现 | **否，尚需重训** | 官方 626-complex 数据与当前 3,114-instance development cohort 不同；现有 MIPE pickle 也不覆盖 v3.1 development IDs |
| SPPIDER-seq 旧结果 | **否，背景** | 旧 frozen split 为 2,727/481/455，不是当前 antigen-fold |
| ParaSurf/Paragraph/其他论文表 | **否，背景** | 数据、candidate region 或阈值不同 |

现有历史结果仍有价值，但只能回答“实现能否在它自己的协议下工作”，不能回答“谁在当前 antigen-fold 上更好”。

### 2.4 MIPE 公平复现的最小执行方案

1. 只为 v3.1 的 3,114 development instances 重新生成 MIPE 所需 antigen ESM、antibody AbLang、GearNet node/edge 与 pair label；不物化 sealed test，也不复用 MIPE 自带的随机 split。
2. 固定官方模型结构和 objective；只添加数据 adapter、mask adapter 和标准化预测导出，不把 FrameContact 模块塞进 MIPE。
3. 用 train 训练、validation 选择 checkpoint；本阶段不读取 test predictions。
4. 把 antigen/paratope logits 按 `instance_id + residue_index` 导出，再调用 FrameContact evaluator。
5. 同时报告官方 MIPE evaluator 和统一 evaluator；两者不一致时以统一 evaluator 作为公平表，官方 evaluator作为复现诊断。

### 2.5 已完成五 seed 开发集消融：能回答什么

机器可读汇总：[`FRAMECONTACT_HEAD_DISTOGRAM_MULTISeed_V1.json`](FRAMECONTACT_HEAD_DISTOGRAM_MULTISeed_V1.json)，SHA-256 `a6da6615fee8dac0b9a7aba83f0934458d0bca6ac161e37147e94e05b274a14a`。汇总器逐一验证了 15 个 run 的 result、checkpoint、receipt、protocol 与 data authority；所有 run 均为 validation-only selection，`test_loaded=false`、`external_test_loaded=false`。

下面是每个 arm 各自 validation-selected checkpoint 的五 seed 描述性差值。由于各 arm 的最佳 epoch 不同，这张表用于回答“最终开发集模型表现如何”，不能替代 matched-epoch 机制归因。

| 单变量比较 | contact pooled AUPRC | contact macro AUPRC | epitope pooled AUPRC | paratope pooled AUPRC | 结论 |
|---|---:|---:|---:|---:|---|
| distogram − binary-hybrid | −0.00046 | −0.00123 | −0.00455 | −0.01947 | distogram promotion gate 未通过 |
| triangle − distogram | −0.00059 | −0.00008 | +0.00049 | +0.01338 | triangle promotion gate 未通过 |

triangle 的主要变化是 operating point，而不是 pair ranking：在各 arm 自己的 validation-selected threshold 下，epitope recall 平均增加约 `+0.1505`（5/5 seeds），但 epitope precision 平均下降约 `−0.0137`；在固定 raw-logit=0 阈值下，contact precision 仅增加约 `+0.00102`，contact recall 反而平均下降约 `−0.0830`。同时 minimum-direction partner selection margin 平均下降 `−0.00683`。因此不能把 triangle 表述成“contact prediction 已提高”；更准确的结论是它改变了稀疏度与 residue-level PR trade-off，但没有通过 pair-contact 和 partner-specificity 门槛。

residue aggregation 的五 seed/四 seed 描述性结果也支持同一判断：

- gated-hybrid 相对 unary-only：contact AUPRC `+0.00118`、epitope AUPRC `+0.00875`，但 paratope AUPRC `−0.00983`，两个方向的 partner margin 均下降；
- pure contact-marginal 当前只有 4 个完整 seed：epitope AUPRC `+0.01552`，但 paratope AUPRC `−0.07794`，4/4 seeds 都更差。

所以当前最稳妥的工程选择仍是保留 hybrid 作为开发 baseline，但论文层面的核心问题没有解决：contact-derived score 帮助的是一般 interface propensity，尚未证明更强的 cognate partner compatibility。

### 2.6 specificity-gauge v3：完整五 seed matched epoch 12

开发作业 `1349969` 的 15 个 tasks 均已至少完成 epoch 12，因此可以在不选 checkpoint 的情况下比较三种 residue/specificity formulation。机器可读 live snapshot：[`FRAMECONTACT_GAUGE_V3_JOB1349969_LIVE_EPOCH12_20260804.json`](FRAMECONTACT_GAUGE_V3_JOB1349969_LIVE_EPOCH12_20260804.json)，SHA-256 `c15f20f4b9bb0e04929812deba3978a13d3cda9a1487f2a21126fb8cc518dee6`。这是 in-flight antigen-fold validation 描述，不是最终选模结果。

| matched epoch 12 | contact AUPRC | epitope AUPRC | paratope AUPRC | worst-direction margin |
|---|---:|---:|---:|---:|
| pure gauge marginal | 0.01129 | 0.12564 | 0.62799 | +0.01275 |
| gauge hybrid | 0.01144 | 0.12791 | 0.71055 | −0.00307 |
| hybrid + residue contrastive | 0.01092 | 0.12534 | 0.64506 | −0.00309 |

单变量差值：

- hybrid − marginal：paratope AUPRC `+0.08256`（5/5 seeds），但 worst-direction partner margin `−0.01582`；
- contrastive − hybrid：两个方向的平均 margin 分别 `+0.01276/+0.00396`，但 worst-direction margin `−0.00002`，contact AUPRC `−0.00052`，paratope AUPRC `−0.06549`（0/5 seeds 改善）。

这说明 hybrid 的优势主要是恢复容易预测的 paratope propensity，而不是提高 partner specificity；现有 residue contrastive loss 又以明显 paratope 退化为代价，且没有改善最差方向。因此两者目前都不能作为核心方法晋级，必须继续以 interaction-only routing 与 retrieval gate 作为主判断。

## 3. 架构：精确到计算图

### 3.1 共同输入

每个 residue 输入：

- frozen ESM residue embedding；
-真实 `N/Cα/C` 坐标；
- chain ID；
- residue mask；
- valid-frame / missing-frame mask。

抗体 encoder 和抗原 encoder **参数完全不共享**。坐标只在各自单体内部构造局部 frame 和相对几何；未对齐单体之间从不直接计算 `x_ab - x_ag`。

### 3.2 单链 geometry self-attention

```text
for partner in {antibody, antigen}:              # 两套独立参数
    h0 = Linear(LayerNorm(frozen_ESM))
    h_dense = DenseSelfAttention(h0, residue_mask)
    h_geo = SpatialSelfAttention(
        h0,
        neighbor_graph = kNN/radius/sequence-window,
        attention_bias = MLP(distance_RBF,
                             local_direction,
                             relative_rotation)
    )
    h = h_dense + tanh(geometry_gate) * (h_geo - h0)
```

`geometry_gate` 从 0 初始化，因此模型初始时精确退化为 dense sequence path；geometry 分支仍可从梯度逐步开启。

### 3.3 late FrameContact（代码 condition：`framecontact`）

```text
h_ab, h_ag = separate_intra_chain_encoders(ab, ag)

# 显式 pair contact head
interaction_ij = dot(Wq h_ab[i], Wk h_ag[j]) / sqrt(d)
contact_logit_ij = interaction_ij + unary_ab[i] + unary_ag[j] + bias

# 两个方向参数不共享
h_ab2 = CrossAttention_ab_from_ag(h_ab, h_ag,
                                  bias = contact_logit)
h_ag2 = CrossAttention_ag_from_ab(h_ag, h_ab,
                                  bias = contact_logit.T)

final_contact_logit = PairHead(h_ab2, h_ag2)
paratope_contact = Marginalize_j(final_contact_logit)
epitope_contact  = Marginalize_i(final_contact_logit)

paratope = unary_paratope + tanh(g_para) * paratope_contact
epitope  = unary_epitope  + tanh(g_epi)  * epitope_contact
```

这是 **late fusion**：每条链先完成全部 self-attention，再做一次 bidirectional cross-attention。

### 3.4 三种 residue head，不应合称一个 `FrameContact_Contact`

| exact condition | residue 输出 | 科学问题 |
|---|---|---|
| `framecontact_unary_only` | 单链 encoder 的 unary logits | 不使用 contact map 时的 interface propensity |
| `framecontact_contact_marginal` | 仅由 contact probability 沿 partner 维度 marginalize | contact-first factorization 能否独立支撑 residue prediction |
| `framecontact_contact_hybrid` | unary + 可学习 zero-gated contact marginal | contact 错误是否可以被 unary path 缓冲 |

所以若老师口头说的 `FrameContact_Contact` 是“加入 contact-derived residue score”，建议在飞书里直接改写成 `contact-marginal` 或 `gated-hybrid`，不要保留一个无法对应代码的别名。

### 3.5 interleaved ICG

```text
for layer in 1..L:
    h_ab = IntraChainLayer_ab(h_ab)
    h_ag = IntraChainLayer_ag(h_ag)
    contact_components = PairHead(h_ab, h_ag)
    routing_bias = choose(
        ordinary        -> 0,
        full_contact    -> interaction + unary_ab + unary_ag,
        interaction_only-> interaction
    )
    h_ab, h_ag = BidirectionalCrossAttention(h_ab, h_ag, routing_bias)
```

普通 cross-attention、full-contact routing 和 interaction-only routing 只改变 routing bias 的来源；M2/M3/M4 其余模块保持相同。前 5% steps gate 为 0，后 10% 线性 ramp，避免随机 pair head 在初期破坏单链表示。

### 3.6 新 Plan B：soft coarse-to-fine，而不是 hard crop

老师提出第一轮 paratope/epitope 只用于 cropping、不计算 loss。实现采用更安全的 soft prior：

```text
coarse_para = stop_gradient(sigmoid(unary_para / T))
coarse_epi  = stop_gradient(sigmoid(unary_epi  / T))
pair_prior  = floor + (1-floor) * coarse_para[:,None] * coarse_epi[None,:]
routing_bias = interaction_only_logit + strength * log(pair_prior)
```

- coarse logits 不直接加入 loss；
- 默认 stop-gradient；
- `floor=0.05`，所有真实 residue pair 仍保留，不会因早期 epitope 错误被永久删除；
- final contact 和 final residue predictions 仍使用完整 pair candidate mask。

### 3.7 新 far-patch exclusion

原 reverse triangle 只约束距离差，不能显式“拉近一个 patch、推远另一个 patch”。新项使用普通 triangle inequality：

```text
对 CDR residue i、当前 antigen patch residue j、远端 surface residue k：

D(i,j) + D(i,k) >= d_antigen(j,k) - slack
```

当模型把 `D(i,j)` 拉小，而 `j,k` 在 antigen 单体上相距很远时，约束会要求 `D(i,k)` 变大。它只使用预测 inter-chain distance 和已知 antigen intra-chain distance，不需要对齐两个单体坐标系。

四条件单变量矩阵：

1. P0 interaction-only，无 crop；
2. P1 + soft crop；
3. P2 + joint contact/distogram；
4. P3 + far-patch exclusion。

P3 只与 P2 比，避免把 distogram 和 far-patch 两项改变混成一个结论。

## 4. 名词自包含检查

| 名词 | 本项目中的精确定义 | 容易误解的地方 |
|---|---|---|
| contact logit | 一个 residue pair 接触的未归一化二分类分数 | 不是距离，也不是概率 |
| contact probability | `sigmoid(contact_logit)`，或 distogram 近距离 bins 概率之和 | 概率需校准；不能跨 complex 随意比较 |
| distogram | residue pair 距离区间的 categorical distribution | 不是一个单值 distance matrix；期望距离只是其投影 |
| unary propensity | 只由目标链本身得到的 interface 倾向 | 能找到表面/CDR，但不能证明 partner specificity |
| interaction-only | pair head 中同时依赖两个 residue embedding 的双线性项 | 去掉 unary shortcut 不保证自然获得 specificity，仍需监督与检验 |
| marginalization | 把 pair contact 沿 partner residue 轴聚合成 residue score | noisy-OR、mean probability、log-mean-exp 含义不同 |
| macro AUPRC | 先按 complex 算 AP，再平均 | 对每个 complex 权重相同；零正例样本需明确处理 |
| pooled AUPRC | 把所有 residue/pair 合并后算 AP | 长 complex 权重更大，且受跨-complex calibration 影响 |
| antigen-fold validation | 结构隔离规则下的 development role | 不是 external benchmark，也不是 frozen test |
| partner margin | cognate pair compatibility 减 wrong-partner compatibility | 正值只说明排序方向；仍需 CI、Recall@1/MRR 和定向 epitope 变化 |

## 5. 对此前表述的修正

1. “最终 epitope/paratope 严格从 contact marginal 得到”只适用于 `contact_marginal_only`；hybrid 模型仍包含 unary logits。
2. “full-contact gate 改善了模型”目前证据只是单 seed、早期 matched epoch；准确表述应是“出现早期正向信号”。
3. “interaction-only 更 partner-specific”是架构假设，不是当前实验结论；epoch 7 的 margin 反而没有支持它。
4. “antigen-fold 有 549 个 unique antigen”不准确；549 是 complex 数，exact antigen sequence 是 314。
5. published MIPE、旧 SPPIDER-seq 和当前 FrameContact 的数字不能直接排名。

### 5.1 老师问的 train-set 拟合，目前的严格答案

旧 run 每个 epoch 记录了完整 train objective（contact、epitope、paratope、specificity、distogram/triangle loss），但没有同步记录 train contact/epitope/paratope AUPRC。因此仅凭“train loss 持续下降”不能断言模型已经在 train set 上充分拟合。

仓库已有 hash-bound 的 post-hoc train evaluator：它只加载 validation-selected checkpoint，复用 validation 冻结的 temperature 与 threshold，对完整 2,565 个 train instances 评分；train 指标不能重新选 checkpoint，且入口显式禁止 test/external role。完整 3 条件 × 5 seeds 作业 `1350158` 已提交，依赖 `afterok:1349969_*`；上游 15 个 gauge-v3 tasks 全部成功完成后会自动启动，不需要重复提交。它会补算：

- train/validation contact pooled 与 macro AUPRC gap；
- epitope/paratope AUPRC、precision、recall、F1、MCC gap；
- 两个方向的 train partner margin 与 validation partner margin。

判读规则提前固定：train 高、validation 低说明主要是跨 antigen-fold 泛化/正则化瓶颈；train 和 validation 都低说明主要是优化、监督或表示能力瓶颈；train partner margin 高但 validation 低说明记忆了训练 partner；两边 margin 都低说明 partner specificity 连训练集都没学到。

## 6. 当前可交付状态

- cohort 统计脚本与 hash-bound JSON：完成；
- FrameContact/ICG/Plan B 架构伪代码与名词检查：完成；
- Plan B 与 far-patch 实现：定向回归通过；
- distogram/triangle 五 seed：15-run 权威汇总完成，两个 promotion gate 均未通过；
- 4 条件 × 5 seeds coarse-to-fine 协议与 Slurm 入口：已冻结；P0 五 seed 已在 ParaCloud 运行，其余条件按 GPU 名额顺序提交；
- complete-train AUPRC audit：实现和权限测试完成；15-task 作业 `1350158` 已排队，等待 gauge-v3 `1349969_*` 全部成功后自动启动；
- MIPE 同 split baseline：公平性协议已定义；远端 readiness audit 确认官方源码/checkpoint、TorchDrug、ESM 与 SAbDab2 cache 存在，但 AbLang runtime、官方 pickle 和 v3.1 MIPE-format feature cache 不存在，需只为 3,114 个 development instances 重新物化后重训；详见 [`MIPE_V3_1_READINESS_AUDIT.json`](MIPE_V3_1_READINESS_AUDIT.json)；
- frozen/external test：本阶段不用于选择或比较。

## 6.1 Plan A：更广义 PPI 预训练的可执行边界

通用 PPI 预训练不能直接与 Plan B 混在同一轮消融。推荐把它注册为下一轮、只改变初始化的实验：

```text
A0: current FrameContact initialization
A1: monomer split-and-merge proxy pretraining -> same FrameContact fine-tuning
A2: real general-PPI contact pretraining       -> same FrameContact fine-tuning
```

三个条件必须保持 SAbDab2 下游架构、optimizer、训练步数、seed 和 evaluator 相同。预训练数据还必须按 SAbDab2 v3.1 的 antigen/antibody exact sequence 与结构 cluster 做去污染；否则 unseen-antigen-fold 结论不成立。

优先做 A1，因为它不需要把一般 PPI 的链类型和抗体链直接等同，并且预训练目标与 inter-chain contact prediction 一致。Split-and-Merge Proxy 的核心就是把单体切成两个 pseudo partners，再预测被切断的 contact；这可以学习 interface/contact 几何，但不能自动证明 antibody specificity。A2 只有在 DIPS/DB5 等 general-PPI 数据的许可、结构质量、链拆分和去污染审计完成后再运行。

一手依据：[Split and Merge Proxy (ICLR 2024 submission)](https://openreview.net/forum?id=FZfWQFrdBT)；AsEP 也把 antibody-specific epitope prediction 明确写成跨图 bipartite link prediction，并报告模型在 unseen epitope 设置中的显著退化，说明“通用 contact 预训练 + 严格结构隔离下游验证”比只扩大当前 MLP 更符合问题本身。[AsEP (NeurIPS 2024 Datasets and Benchmarks)](https://openreview.net/forum?id=Q7xKdEMrrZ&noteId=ZTGL9UCNRL)

Plan A 的 promotion gate 不是预训练 loss 下降，而是同一 antigen-fold validation 上：

1. pair contact pooled 与 macro AUPRC 同时不退化；
2. partner retrieval MRR/Recall@1 和 cognate-minus-wrong margin 提升；
3. epitope AUPRC 改善且 paratope 不出现系统性下降；
4. geometry/partner shuffle 的定向下降变强，而不是只提高 unary propensity。

## 7. 给老师的短版回复

老师您好，我把您提的三部分先整理成了可审计版本。当前实际使用的 SAbDab2 v3.1 antigen-fold 是 2,565/549/549；validation 的 549 个 complex 包含 314 个 exact antigen sequences 和 334 个 exact antibody sequences。antigen 长度中位数 153，P25–P75 为 113–221，最大 1,390；antibody 中位数 227，P25–P75 为 223–230。完整统计会连同 split/metadata SHA-256 放进飞书文档。

公平比较方面，MIPE 论文和我们此前复现结果使用的是另一套 626-complex/random-CV 协议，SPPIDER-seq 旧结果也是 2,727/481/455 的另一套 split，因此目前不能和 FrameContact 的 antigen-fold validation 直接排名。我会保留它们作为历史背景；主比较只接受在当前 2,565/549 instance IDs 上重训、用同一 residue mask/label/evaluator 得到的结果。目前 `sequence_dense`、unary-only、contact-marginal、contact-hybrid 和 interleaved 条件已经满足同协议要求，MIPE 还需要为当前 v3.1 cohort 重新生成官方所需的 AbLang/ESM/GearNet features 后重训。

架构上我也把命名拆清了：仓库中没有唯一的 `FrameContact_Contact` 条件，而是 unary-only、contact-marginal 和 gated-hybrid 三种 residue head。FrameContact 先用两个参数不共享的 antibody/antigen local-frame self-attention encoder，再显式预测 residue-pair contact；late 版本只在单链编码结束后做一次双向 cross-attention，interleaved 版本则每层更新 contact routing。文档会给出伪代码，并逐项解释 contact logit、distogram、marginalization、unary propensity、interaction-only 和 pooled/macro AUPRC，避免只使用英文名词而没有定义。

另外，distogram/triangle 的五 seed 开发集结果已经完成。正式 15-run 汇总显示，distogram 相对 binary-hybrid 的 contact pooled/macro AUPRC 分别为 `−0.00046/−0.00123`；triangle 相对 distogram 为 `−0.00059/−0.00008`，两个预先冻结的 promotion gate 都没有通过。triangle 确实提高了 epitope recall，但同时降低 precision、固定阈值 contact recall 和 partner margin，所以目前不能说它改善了 partner-specific contact prediction，只能说它改变了 PR operating point。现有旧日志只记录 train loss，没有 train AUPRC；我已经准备了不参与选模、完全不触碰 test 的 complete-train checkpoint audit，下一步会直接用 train/validation gap 判断是训练集未拟合，还是 antigen-fold 泛化失败。

另外，specificity-gauge v3 的 15 个任务已经形成五 seed matched epoch 12 结果。hybrid 相对 pure marginal 能把 paratope AUPRC 提高约 `+0.0826`，但最差方向的 partner margin 下降 `−0.0158`；在 hybrid 上加 residue contrastive 后，contact AUPRC `−0.00052`、paratope `−0.0655`，最差 partner margin 基本不变。因此现阶段的主要矛盾不是 paratope 能否预测，而是模型是否真正学习 cognate pair compatibility。完整 train-set audit 作业 `1350158` 已经排队，会在当前 gauge-v3 五 seed全部结束后自动运行。
