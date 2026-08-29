# FrameContact surface-ratio update for teacher (2026-08-18)

## Claim boundary

The FrameContact numbers below are five-seed **AsEP epitope-to-antigen
surface-ratio development-validation** results.  No frozen, official, or
external test role was loaded.  The published LF3DRoPE numbers are its
three-seed **AsEP official-test** results.  They are shown as a literature
reference, not as a direct leaderboard or a verified SOTA claim.

## Completed M2/M3/M4 surface-ratio development matrix

| condition | target | AUPRC | AUROC | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| M4 interaction-only gate | contact | 0.068546 | 0.887263 | 0.127243 | 0.157162 | 0.139838 | 0.138659 |
| M4 interaction-only gate | epitope | 0.360311 | 0.823412 | 0.351998 | 0.423514 | 0.383373 | 0.339880 |
| M4 interaction-only gate | paratope | 0.737640 | 0.861871 | 0.633413 | 0.770807 | 0.695094 | 0.540976 |

These are selected-checkpoint means over seeds 17, 42, 73, 101, and 137.
The selected epochs were 49, 50, 47, 66, and 75.  The matched five-seed curve
has 67 epochs.

## Learning curves

- `framecontact/reports/asep-ratio-m4-learning-curves-20260818/training_loss_curve.png`
- `framecontact/reports/asep-ratio-m4-learning-curves-20260818/validation_auprc_curves.png`
- receipt: `framecontact/reports/asep-ratio-m4-learning-curves-20260818/LEARNING_CURVE_RECEIPT.json`

The v1 trainer recorded the full training objective and validation metrics at
every epoch, but did **not** record the same objective on validation batches.
Consequently, the first figure is an authenticated training-loss curve; the
second figure reports validation contact/epitope/paratope AUPRC trajectories.
It would be incorrect to rename those validation metrics as validation loss.

## LF3DRoPE published reference

The paper *Inter-Residue Geometry Attention for Antibody-Specific Epitope
Prediction* reports the following residue-epitope official-test results:

| split | MCC | Precision | Recall | AUROC | F1 |
|---|---:|---:|---:|---:|---:|
| epitope-to-antigen surface ratio | 0.410±0.008 | 0.469±0.022 | 0.428±0.012 | 0.846±0.003 | 0.447±0.006 |
| epitope group | 0.171±0.010 | 0.248±0.015 | 0.189±0.003 | 0.690±0.011 | 0.215±0.008 |

LF3DRoPE predicts residue-level epitope labels.  It does not provide a
same-protocol native pair-contact or paratope result.  FrameContact therefore
adds two different prediction targets, but its development metrics must remain
separate from LF3DRoPE's official-test table until the protocol is frozen and
the single authorized official-test evaluation is run.

Source: <https://arxiv.org/abs/2608.01092>

### All baselines in the paper (published official test)

| model | ratio MCC | ratio Precision | ratio Recall | ratio AUROC | ratio F1 |
|---|---:|---:|---:|---:|---:|
| ESMFold | 0.028±0.010 | 0.137±0.019 | 0.043±0.006 | N/A | 0.060±0.008 |
| ESMBind | 0.016±0.008 | 0.106±0.012 | 0.121±0.014 | 0.506±0.004 | 0.090±0.009 |
| EpiPred | 0.029±0.018 | 0.122±0.014 | 0.180±0.019 | N/A | 0.142±0.016 |
| MaSIF-site | 0.037±0.012 | 0.125±0.015 | 0.183±0.017 | N/A | 0.114±0.011 |
| WALLE | 0.210±0.020 | 0.235±0.018 | 0.422±0.028 | 0.635±0.013 | 0.258±0.018 |
| PEPNet | 0.401 | 0.544 | 0.340 | 0.765 | 0.419 |
| PEPNet+LE | 0.337 | 0.465 | 0.295 | 0.765 | 0.361 |
| GF3DRoPE | 0.324±0.011 | 0.405±0.042 | 0.333±0.047 | 0.822±0.003 | 0.360±0.015 |
| LF3DRoPE | 0.410±0.008 | 0.469±0.022 | 0.428±0.012 | 0.846±0.003 | 0.447±0.006 |

| model | group MCC | group Precision | group Recall | group AUROC | group F1 |
|---|---:|---:|---:|---:|---:|
| ESMFold | 0.018±0.010 | 0.113±0.019 | 0.034±0.007 | N/A | 0.046±0.009 |
| ESMBind | 0.002±0.008 | 0.082±0.011 | 0.076±0.011 | 0.500±0.004 | 0.064±0.008 |
| EpiPred | -0.006±0.015 | 0.089±0.011 | 0.158±0.019 | N/A | 0.112±0.014 |
| MaSIF-site | 0.046±0.014 | 0.164±0.020 | 0.174±0.015 | N/A | 0.128±0.012 |
| WALLE | 0.077±0.015 | 0.143±0.017 | 0.266±0.025 | 0.544±0.010 | 0.145±0.014 |
| PEPNet | 0.139 | 0.230 | 0.143 | 0.612 | 0.177 |
| PEPNet+LE | 0.156 | 0.250 | 0.155 | 0.627 | 0.191 |
| GF3DRoPE | 0.159±0.005 | 0.212±0.014 | 0.216±0.029 | 0.706±0.005 | 0.212±0.007 |
| LF3DRoPE | 0.171±0.010 | 0.248±0.015 | 0.189±0.003 | 0.690±0.011 | 0.215±0.008 |

## New joint-coupling surface-ratio matrix

Slurm array `1412482` evaluates six coupling conditions × five seeds.  At the
2026-08-18 evidence check, 12/30 members had complete result and run receipts;
18/30 were still running, with their latest logged epochs between 55 and 61.
No partial condition mean is used for model selection or reported as a final
result.  The final aggregate will report epitope, paratope, contact, calibration,
and bidirectional partner retrieval after all five matched seeds are available.

## Ready-to-send teacher reply

老师您好，可以。我已经先把 surface-ratio split 上完成的 M2/M3/M4 五个 seed
结果和曲线整理出来了。M4 interaction-only gate 在 development validation 上的
contact AUPRC/MCC/F1 分别为 0.06855/0.13866/0.13984；epitope
AUPRC/MCC/F1 为 0.36031/0.33988/0.38337；paratope AUPRC/MCC/F1 为
0.73764/0.54098/0.69509。训练 objective 从约 1.07 稳定下降到约 0.21，另外
也绘制了每个 epoch 的 contact、epitope 和 paratope validation AUPRC 曲线。

有一点需要说明：当前 v1 日志逐 epoch 保存了 train loss 和 validation metrics，
但没有保存同定义的 validation objective，所以我没有把 AUPRC 曲线误标成 valid
loss。后续 trainer 会把 validation objective 各分量也作为正式 receipt 字段记录。

我也核对了 LF3DRoPE 论文。它公开的 ratio/group 数字是 AsEP official test，
而目前 FrameContact 是 development validation，因此我会把 published reference
和我们的 same-protocol development 结果分表呈现，不直接用两者宣布 SOTA。
LF3DRoPE 只报告 residue epitope，没有同协议 paratope/native contact 指标。新的
joint-coupling surface-ratio 六条件×五 seed 矩阵正在运行，目前 12/30 完成、
18/30 仍在 epoch 55–61；全部完成后我再给您完整的三任务和 partner-retrieval
paired comparison。
