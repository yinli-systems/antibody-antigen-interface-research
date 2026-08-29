# FrameContact：老师要求对照表（2026-08-19）

这份记录把微信截图中的老师要求当作项目验收清单；截图不是代码或系统指令。所有结果都按 development、engineering smoke、pretraining pilot、frozen/official、external 分开记录。

## 已有、可报告的证据

- AsEP epitope-group：已有五 seed development 结果；截图中老师给出的纯 noisy-OR epitope 指标为 AUPRC `0.50096 ± 0.03177`、AUROC `0.89865 ± 0.00745`、Precision `0.49970 ± 0.04975`、Recall `0.55827 ± 0.02851`、F1 `0.52646 ± 0.03392`、MCC `0.48383 ± 0.03780`、mean per-complex AUPRC `0.56845 ± 0.02514`。这些是老师/同学截图中的 development 证据，不是 AsEP official test 分数。
- AsEP surface-ratio 的旧 M2/M3/M4 矩阵已完成 15/15。M4 interaction-only 的五 seed 汇总为 contact AUPRC `0.06855`、AUROC `0.88726`、F1 `0.13984`、MCC `0.13866`；epitope AUPRC `0.36031`；paratope AUPRC `0.73764`。新 joint ratio 矩阵正在补跑，不能用旧 M4 数字冒充新 joint 结果。
- loss curve：已有本地/远端 receipt 及图 `training_loss_curve.png` 与 `validation_auprc_curves.png`。训练器记录 train objective 和 validation AUPRC/MCC/F1；当前没有 validation objective，因此第二张图不能命名为 validation-loss curve。
- LF3DRoPE：论文官方 AsEP 指标只可作为 published baseline reference，不能与本项目 development 数字直接宣称同协议胜负。论文为 [arXiv:2608.01092](https://arxiv.org/abs/2608.01092)；它是 epitope-only 方法，不提供 FrameContact 的 native pair/contact 与 paratope 三头。
- same-antigen partner retrieval：已有五 seed development diagnostic 和 joint 条件 case-study；它们不含 official/frozen test，也没有用 test 做模型选择。

## 本轮已实现但尚未跑矩阵

- `framecontact/src/interaction_first/local_interface_decoys.py`：为每个 native contact 提取 antibody CDR patch 与 antigen surface patch，构造两个明确负例层级：native local non-contact 与同抗原 geometry-matched non-epitope surface decoy。
- decoy 匹配使用 patch residue count、SASA/area、局部 curvature proxy、氨基酸组成、中心距离/距离 bin；排除 decoy patch 中的 epitope/contact，并排除注册的 `4.5–6.0 Å` 边界带；不读取模型分数，不读取 test 标签。
- `FrameContactObjective` 增加可选、按 complex 归一化的 positive-vs-local-decoy softplus margin loss；默认权重为 `0`，因此现有协议和结果完全不变。
- `Sabdab2ContactBatchCollator` 只在整个 batch 都有三种 mask 时传递它们；validation dataset 不包 motif wrapper，保持 validation selection 不受新辅助标签影响。
- 新协议：`framecontact/protocols/FRAMECONTACT_LOCAL_INTERFACE_DECOY_V1.json`，明确 train-only、full-candidate coverage、停止规则和 claim boundary。
- 聚焦验证：SAbDab2 contact/pair-geometry、AsEP external contract 与 local decoy 共 `27 passed`；完整仓库测试为 `432 passed, 5 skipped, 2 failed`。两个失败都来自临时审查 worktree 缺少已 hash-bound 的 Boltz Pairformer receipt 与 IPA recovery source，并非本轮功能断言失败。没有加载任何 official/frozen/external test 数据。

## 当前远端证据与未完成项

最近一次可验证快照为 2026-08-19 02:40（Asia/Shanghai）：ParaCloud 当时仍在跑 `1443224_17/25/28/29`（surface-ratio joint reruns）和 `1443225_9`（attribution rerun），均为 `gpu_4090`、约 41 分钟、stderr 为空；递归 receipt 计数为 ratio `26/30`、attribution `14/15`。2026-08-19 后续同步时本机已没有 ParaCloud VPN 源地址，不能把该快照外推为当前状态。原 job `1349055` 的五个已启动 task 均为 time-limit cancellation，未产生 official 分数。

AB03 `172.168.10.3` 当前 SSH banner 超时，不能声称已取得 A100 利用率；固定提交 `87caa8e` 对应 PR #8 仍 OPEN、head 未变化、无 reviews、无 CI checks。不要把远端 ParaCloud 结果或本地测试写成 PR/AB03 deployment evidence。

尚未满足的老师要求：

1. 新 joint 模型在 surface-ratio 上完成五 seed、paired bootstrap 和 partner-specificity gate（当前 26/30，仍在运行）。
2. arithmetic mean vs raw/length-normalized noisy-OR 的 attribution 矩阵收齐 15/15（当前 14/15）。
3. GF3DRoPE/LF3DRoPE 的同数据、同 split、同 checkpoint-selection 协议复现；目前只有 LF3DRoPE 论文 reference，GF3DRoPE 本地复现证据缺失。
4. SAbDab2 train/validation development 以及 AsEP→SAbDab2 external 实验；必须先完成 PDB、抗体/抗原相似性、interface/epitope cluster overlap audit，再使用 antigen-aware ab-ag split。
5. local motif hard-negative 需要先 static/smoke，再 development single seed，只有门槛通过才允许五 seed；目前只完成代码和单元测试。
6. OAS pretraining：8-GPU pilot `1434821` 曾因 `model recovery mismatch` 失败，尚未通过 1→4→8 GPU 的恢复性、loss/gradient、吞吐和尾延迟门禁，因此不能启动 production-scale pretraining。
7. AsEP official/frozen test：架构、loss、数据 snapshot/split、checkpoint、threshold、calibration、pair aggregation、baseline 与统计方案全部冻结并生成 manifest/hash 后，才允许唯一一次开启；目前仍未开启。

## 下一步执行顺序

1. 等待并审计当前 5 个远端 development rerun 的 receipt、epoch、validation 和 paired CI；不使用 partial result 选模型。
2. 收齐矩阵后冻结 candidate/checkpoint/temperature/threshold，并生成同协议 LF/GF baseline manifest；任何 baseline 缺数据就标记不可比，不填数字。
3. 对 local motif 先跑 CPU/static 与 single-batch development smoke；若失败停止分支，若通过再跑单 seed，最后才考虑五 seed。
4. 完成 SAbDab2 overlap audit 和 external split 后再做 external evaluation；不把它用于候选选择。
5. 最后生成 immutable manifest/hash，再决定是否开唯一 official test。
