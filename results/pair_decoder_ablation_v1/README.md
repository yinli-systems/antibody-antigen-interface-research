# Pair-decoder ablation v1

Status: development-only single-seed result. It establishes nothing on its own
and authorizes no model selection, no test access, and no benchmark claim. The
frozen test split was not named or read.

Date: 2026-09-09

Machine-readable: [`report_compact.json`](report_compact.json) — the full report
including per-complex records stays on the cluster at
`/ssd/scxi253/decoder_ablation/out/report.json` (714 KB), since per-complex
prediction payloads are not tracked here.

## What was run

Two arms, identical in every respect except the pair decoder: same split, same
seed (42), same frozen ESM-2 embeddings, same schedule, same losses.

| | |
|---|---|
| Training cohort | 3,229 authority-train instances |
| Evaluation | 217 validation complexes, 9,638,880 candidate pairs, 9,036 contacts |
| Positive rate | 0.094% (micro), 0.135% (macro) |
| Selection | validation `partner_rank`, best checkpoint at **epoch 1** for both arms |

## Result: both decoders beat the partner-agnostic baseline

The predicted-marginal outer product scores a pair by
`P(epitope_i) * P(paratope_j)`, so it carries no information about *which*
antibody residue pairs with *which* antigen residue.

| macro AUPRC | value | over baseline |
|---|---:|---:|
| bilinear | **0.02397** | **+32.7%** |
| hadamard_mlp | 0.02268 | +25.5% |
| outer-product baseline | 0.01806 | — |
| prevalence floor | 0.00135 | — |

This is the question the workstream exists to answer, and the answer here is
positive: the pair head carries partner-specific information beyond the
marginals. It also does **not** reproduce the inversion recorded for the MIPE
replay, where the native pair head scored 0.00342 against 0.04045 for the same
baseline. That inversion is a property of that reconstruction, not of the task.

## Result: the decoder choice does not matter

| macro | bilinear | hadamard_mlp |
|---|---:|---:|
| auprc | 0.023971 | 0.022680 |
| auroc | 0.922338 | 0.922499 |
| bedroc_alpha20 | 0.451355 | 0.450926 |
| mcc | 0.073072 | 0.073590 |
| best_mcc | 0.102101 | 0.097913 |
| f1 | 0.018523 | 0.019689 |
| best_f1 | 0.063181 | 0.061576 |
| balanced_accuracy | 0.819645 | 0.806149 |
| precision_at_true_count | 0.029941 | 0.030810 |
| ef_at_1pct | 19.31 | 18.49 |
| cohens_d | 2.452 | 2.473 |

The differences sit in the third decimal and **disagree in direction**: bilinear
leads on AUPRC, best F1, best MCC, enrichment and balanced accuracy; hadamard
leads on AUROC, MCC, F1, precision at the true count and Cohen's d. With one
seed this is noise, and the preregistered power analysis for this project puts
the one-sided 95% lower confidence limit on a three-seed effect size at
`-0.254`, so a single comparison has no discriminating power at all.

The hypothesis that motivated the Hadamard decoder — that a low-rank bilinear
form cannot express the interaction and falls back on the marginals — is
therefore not supported, on real data as it was not on synthetic data.

## The fixed 0.5 threshold understates the model

| macro | at 0.5 | best attainable | ratio |
|---|---:|---:|---:|
| F1 | 0.0185 | **0.0632** | **3.4x** |
| MCC | 0.0731 | **0.1021** | 1.4x |

The optimum sits at 0.85 (F1) and 0.66 (MCC), far above 0.5. Historical F1 and
MCC figures computed at a fixed 0.5 are therefore partly a measurement artefact
rather than a statement about the model.

## AUROC is misleading at this prevalence

AUROC 0.922 against AUPRC 0.024 on the same predictions, at a 0.094% positive
rate. The AUPRC is 17.8x the prevalence floor, which is a real signal, and it is
still 0.024 in absolute terms: roughly 97.6% of the top-ranked pairs are not
contacts. Reporting AUROC alone would give the opposite impression.

Enrichment is the friendlier framing of the same predictions and is equally
true: the top 1% of the ranking is about 19x enriched for contacts, which is
useful for reducing a candidate set before docking or experiment.

## Limits

1. **One seed.** Neither the margin over the baseline nor the gap between arms
   is statistically established.
2. **Both arms peaked at epoch 1** and then overfitted for ten more epochs.
   With 3,229 training instances the binding constraint is data, not
   architecture.
3. The validation split comes from a split file that has drifted from the digest
   the S2 assignment registered. Train membership and the sealed pool both still
   reproduce and train and sealed remain disjoint, but one instance moved
   between test and validation.
4. Both training jobs exited non-zero. The failure is in
   `cluster_bootstrap_mean_ci` during post-hoc confidence intervals, after the
   prediction parquets were written atomically, and is identical in both arms.
   It is a pre-existing defect in the evaluation path, unrelated to the decoder.

## What would make this a result

A multi-seed replication of the *baseline margin*, not of the arm comparison.
"Does the pair head beat the outer product" is the question worth powering; on
this variance seven or more seeds would settle it, and one training run takes
about two hours, so ten seeds in parallel costs one wall-clock afternoon.
