# 给老师和杨吴玙的交付说明（2026-09-05）

## 455 条 ESM-2 cache

455 个样本的 ESM-2 embedding 与 split/index 已整理好，原路径保持：

```text
/ssd/scxi253/model_deployment/handoffs/framecontact_case_study_20260904/sabdab2_esm2_frozen_test455_v1/
```

455 条 ID 与已确认的清单一致；514 个唯一 embedding 全部通过哈希校验，
并完成了 910 次 Ab/Ag 的实际 `EmbeddingStore` 加载。入口见
`LOADER_PATHS.json` 和 `CONSUMPTION_NOTES.md`。

需要更正此前“完整 cache 可直接跑 inference”的说法：目前完成的是
embedding 与 split/index 部分。对方 README 仍在 ParaCloud 不可见的
`/data/ab02/.../test_split_request_v1/`，完整 cache 的字段、几何和标签打包
要求尚未核对。请把 `README_test_split_request_v1.md` 复制到上述共享目录旁；
收到规范后可复用现成 embeddings 完成剩余打包，无需重新计算 ESM。

## MIPE：已找回与原论文表格对应的 checkpoint 输出

重新追溯后，找到了原表 MIPE validation-best 的完整预测及 checkpoint
哈希绑定，所选 epochs 为 `572 / 605 / 704 / 522 / 771`。两组 checkpoint
（validation-best 与 epoch-800）共 1,120 份预测全部通过原始哈希校验。

以下为 **validation-best、固定阈值 0.5、micro 汇总**：

| Task | AUROC | Pooled AP | MCC | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Contact（双向 attention 均值） | 0.498119 | 0.001596 | -0.000172 | 0.000000 | 0.000000 |
| Epitope | 0.807419 | 0.280407 | 0.232395 | 0.144572 | 0.464835 |
| Paratope | 0.881067 | 0.689221 | 0.508601 | 0.565147 | 0.675023 |

Contact 的 TP / FP / TN / FN 为 `0 / 210 / 11,412,892 / 18,316`。
上述 micro 是先对所有样本累计 TP/FP/TN/FN，再计算比例。AUROC 和 AP 沿用
同一批原始预测的冻结评测；AP 是 pooled AP，不是 MPC-AP。

前次发出的 Contact TP=1、FP=161 属于较早的 checkpoint 导出，不能与
原论文 MIPE 行拼接使用。原表 Epitope/Paratope Recall `0.55235 / 0.79544`
使用的是每折 validation-fitted 阈值，若全文统一写 fixed 0.5，必须改用
本次恢复出的 `0.14457 / 0.56515`。

Contact 可以明确标注为有 pair supervision 的 MIPE attention-readout
adaptation。这组评测包含 **560 个五折 held-out complexes**，与 455 条
SAbDab2 test 不是同一 cohort，正文表 caption 必须交代，不能写成所有
方法都是同一 455 条样本的直接比较。

结果、CSV 和校验 receipt：

```text
/ssd/scxi253/model_deployment/results/mipe/paper_track_fixed05_recovery_v1/
```

仓库中的完整说明：
[MIPE paper-track recovery](../../results/mipe_paper_track_fixed05_recovery_v1/README.md)。
