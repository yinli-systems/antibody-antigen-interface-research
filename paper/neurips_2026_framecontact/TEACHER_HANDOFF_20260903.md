# Teacher handoff: latest three-task and bilateral Top-K results

## Message ready to send

老师好，最新版结果和双侧 Top-K 消融已经整理完成。

在只使用序列及序列独立预测的链内几何作为模型输入、且不使用 bound-distance
auxiliary supervision 的完整 development 实验中，三个固定 seed 路由后的 contact、
epitope 和 paratope mean-per-complex AP 中位数分别为 `0.02607`、`0.23268`
和 `0.69618`。Contact 和 paratope 来自 seed-matched baseline checkpoint，epitope
来自 seed-matched TopK2 expert checkpoint。这组结果不使用实验复合物坐标作为模型
输入，也不使用实验跨链距离作为辅助训练目标；实验结构仅用于构造监督标签。

三种子各项指标如下（均为各自 selected checkpoint 在 development validation 上的
结果；MPC-AP 为 mean-per-complex AP）：

| Task | Seed | AUROC | MPC-AP | MCC | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Contact | 17 | 0.9319 | 0.0254 | 0.0488 | 0.1337 | 0.0198 |
| Contact | 29 | 0.9350 | 0.0261 | 0.0420 | 0.0961 | 0.0204 |
| Contact | 43 | 0.9228 | 0.0267 | 0.0412 | 0.0999 | 0.0190 |
| **Contact median** | -- | **0.9319** | **0.0261** | **0.0420** | **0.0999** | **0.0198** |
| Epitope | 17 | 0.6495 | 0.2375 | 0.1257 | 0.6097 | 0.1355 |
| Epitope | 29 | 0.6580 | 0.2318 | 0.1349 | 0.6477 | 0.1366 |
| Epitope | 43 | 0.6572 | 0.2327 | 0.1306 | 0.6071 | 0.1381 |
| **Epitope median** | -- | **0.6572** | **0.2327** | **0.1306** | **0.6097** | **0.1366** |
| Paratope | 17 | 0.9540 | 0.6973 | 0.6222 | 0.7464 | 0.5871 |
| Paratope | 29 | 0.9541 | 0.6962 | 0.6266 | 0.7737 | 0.5745 |
| Paratope | 43 | 0.9525 | 0.6924 | 0.6068 | 0.6953 | 0.6013 |
| **Paratope median** | -- | **0.9540** | **0.6962** | **0.6222** | **0.7464** | **0.5871** |

昨天新跑的 pLDDT>=70 confidence-masked candidate 的三任务中位 MPC-AP 虽为
`0.02716 / 0.24259 / 0.71033`，但没有通过预先冻结的晋级规则：seed 29 的
paratope MPC-AP 相对 control 下降 `0.03571`，超过每个 seed 最多下降 `0.005`
的限制。因此该组应保留为 development ablation，不能替换正文采用的三种子
checkpoint。昨天的 bilateral Top-K 训练消融也只有 seed 17，同样不应替代三种子主结果。

另外，我按照预先固定的协议完成了 epitope/paratope 两侧
`K in {1,2,4,8}` 的 from-scratch 消融。Epitope 侧，K=2 的综合表现最好：
mean-per-complex AP、pooled AP、AUROC、F1 和 MCC 分别为
`0.23753 / 0.14405 / 0.64951 / 0.22177 / 0.12568`。K=8 的
mean-per-complex AP 与 K=2 接近，但其余指标均更低，因此当前保留 epitope
TopK2 是合理的。

Paratope 侧，原生 readout 的 mean-per-complex AP 为 `0.67973`；TopK1 和
TopK2 分别达到 `0.70251` 和 `0.70139`。TopK1 的 mean-per-complex AP、F1
和 MCC 最好，TopK2 的 pooled AP 和 AUROC 最好。因此 paratope Top-K
在重新训练后显示出明确的正向信号，但目前只有 seed 17，并且此前的
fixed-checkpoint readout 替换没有改善。为了不夸大结果，我建议正文暂时保留已经冻结的
native-paratope/TopK2-epitope 路由，把完整 K 消融放入 Appendix；如果希望把
paratope Top-K 写成最终模型组成，还需要补三个 seed 的确认实验后再冻结选择。

所以，现有结果能够支持“epitope TopK2 的选择具有消融依据，以及 paratope
Top-K 值得进一步确认”，但不能仅凭这组单 seed development 消融宣称双侧
Top-K 普遍最优或综合达到 SOTA。与 baseline/SOTA 的 comparable 结论仍应基于同一
benchmark、同一 split 和同一 metric 的最终 main performance table。

## Frozen development numbers

### Epitope Top-K from-scratch, seed 17

| K | Mean-per-complex AP | Pooled AP | AUROC | F1 | MCC |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.228217 | 0.138299 | 0.645819 | 0.217486 | 0.113597 |
| 2 | **0.237529** | **0.144046** | **0.649511** | **0.221773** | **0.125680** |
| 4 | 0.234279 | 0.139897 | 0.645966 | 0.220547 | 0.119503 |
| 8 | 0.237326 | 0.134207 | 0.622493 | 0.204677 | 0.104092 |

### Paratope Top-K from-scratch, seed 17

| Readout | Mean-per-complex AP | Pooled AP | AUROC | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Native reference | 0.679731 | 0.629400 | 0.951133 | 0.643753 | 0.607037 |
| TopK1 | **0.702509** | 0.629018 | 0.953033 | **0.656401** | **0.626935** |
| TopK2 | 0.701388 | **0.657073** | **0.954372** | 0.654526 | 0.618771 |
| TopK4 | 0.673524 | 0.585477 | 0.945263 | 0.622226 | 0.583647 |
| TopK8 | 0.680196 | 0.618129 | 0.950771 | 0.648468 | 0.611886 |

## Evidence boundary

- All eight runs completed successfully under the frozen campaign.
- All arms use seed 17 and checkpoint-selection validation data.
- `distance_loss_weight=0`; no privileged distance-label cache was used.
- No old SAbDab2 test or temporal holdout was loaded.
- Fixed-checkpoint readout sensitivity and from-scratch training ablation are
  retained as separate evidence.
- The result supports an Appendix ablation, not a universal optimal-K claim.
