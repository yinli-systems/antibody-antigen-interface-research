# FrameContact 阶段性汇报（2026-08-07）

老师您好，我把目前可以在不打开 frozen/external test 的前提下完成的工作收口了，
并把不同方法的评估范围、输出类型和剩余实验分开记录。

## 已完成

1. **公平基线审计**：整理了 EpiFormer、MIPE 和 SPPIDER-seq 的可追溯结果，保留
   每个结果的 split、样本数、输出类型和阈值角色；没有把不同 split 的数字拼成一个
   虚假的 leaderboard。
2. **公平比较表**：新增
   `docs/FAIR_COMPARISON_TABLE_20260807.md`。EpiFormer 只有 epitope 输出，
   MIPE/SPPIDER 没有 native residue-pair contact 输出，因此 paratope/contact 的
   不适用项明确写成 N/A，不把 residue score 转成 contact score。
3. **FrameContact 架构和实验边界**：保留现有 FrameContact/FrameContact_Contact
   baseline，并记录 interleaved routing、full-contact gate、interaction-only gate、
   binary contact/distogram、triangle/EDM consistency 以及 partner-shuffle 等协议。
4. **可复现性**：增加表格渲染脚本和回归测试；审计脚本、paired bootstrap 和表格测试
   均可在本地重跑。针对新增部分的测试结果为 **4 passed**。
5. **证据边界**：新增
   `docs/FRAMECONTACT_COMPLETION_MATRIX_20260807.md`，逐项区分已经完成的本地审计、
   antigen-fold development evidence、尚未完成的 GPU 实验，以及尚未打开的 frozen/
   official test。

## 当前开发集结果（不能当作最终 test 结果）

最近一次 ParaCloud 快照中，recovery 和 antigen-fold 开发任务的 epoch 分别为 25、
41、25、24；对应 contact AUPRC 为 0.00849、0.01095、0.01123、0.01267。另一个
完成到 epoch 54 的开发任务 contact AUPRC 为 0.01320。上述均为 antigen-fold
validation，仅用于开发选择，不能写入 frozen-test baseline 表。

## 还没有声称完成的部分

- 五 seed 全条件矩阵以及 validation-only winner 冻结；
- train-role 的 contact/epitope/paratope 拟合审计；
- same-antigen/different-antibody partner-specificity challenge 和 retrieval；
- late cross-attention 与 every-layer interleaved 的 matched comparison；
- binary contact 与 distogram 在同一 residue aggregation 下的比较；
- triangle/EDM 五 seed 消融、GVP/EGNN/IPA/Pairformer 强基线；
- CHIMERA co-design case study；
- SAbDab2 frozen test、AsEP official test 和 AB03 A100 utilization。

这些项目不能用其他 split 的公开结果替代。当前 job 1349055 已取消，AsEP official
test opening gate 仍未打开，AB03 SSH 在 banner 阶段超时，因此我没有伪造对应结果。

## 可复现命令

```bash
python3 scripts/audit_baseline_metrics.py
python3 scripts/compare_epiformer_sppider_sabdab2.py
python3 scripts/render_fair_comparison_table.py \
  --output docs/FAIR_COMPARISON_TABLE_20260807.md
python3 -m pytest -q \
  tests/test_audit_baseline_metrics.py \
  tests/test_render_fair_comparison_table.py
```

完整仓库测试在当前机器上仍受缺少 `torch`、`Bio`、`ortools`、`highspy` 依赖和测试
模块路径冲突影响；这不应被表述为完整测试通过。
