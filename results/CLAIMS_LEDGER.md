# Every number that may appear in the paper, and what stands behind it

Built 2026-09-11 by reading the artifacts, not by recalling them. It exists
because a headline figure in this project's own summary turned out to belong to
no artifact at all.

Each row binds: the claim, the experiment and its control, the cohort and split,
node or pair, micro or macro, the threshold rule, the seed set, the absolute and
relative effect, what the confidence interval resamples, and the file.

## The defect that motivated this

`results/EVIDENCE_INVENTORY.md` carried **"+30.5% pair AUPRC, 20 seeds, 20/20,
sign bound 0.8609"** as the pair head's margin over the predicted-marginal outer
product. No artifact contains that number. What exists:

| artifact | arm | seeds | absolute | relative (mean of per-seed ratios) | sign bound |
|---|---|---:|---:|---:|---:|
| `baseline_margin_replication_v1.json` | replication arm | 10 | +0.003113 | +16.7% | 0.7411 |
| `margin_all_esm2.json` | ESM-2 | 20 | +0.002792 | +15.1% | 0.8609 |
| `margin_all_esm2_antiberty_esmif1.json` | ESM-2 + AntiBERTy + ESM-IF1 | 20 | +0.006106 | **+27.8%** | 0.8609 |
| `margin_all_esm_if1_only.json` | ESM-2 + ESM-IF1 | 10 | +0.005073 | +22.8% | 0.7411 |
| `margin_all_antiberty_only.json` | ESM-2 + AntiBERTy | 10 | +0.002443 | +13.6% | 0.7411 |

The quoted figure took the seed count and sign bound from a **twenty-seed arm**
and a relative margin resembling **a different arm's**, and presented them as one
result. Withdrawn. Where a margin over the outer product is quoted, it is one of
the rows above and it names its arm.

**Two relative margins, both correct, not interchangeable.** The replication
README reports +16.1%; the table above reports +16.7% for the same run. The first
is the ratio of means, `mean(margin) / mean(baseline)`; the second is the mean of
per-seed ratios. Any relative figure must say which, and this project uses the
**mean of per-seed ratios** because that is the quantity the paired design
supports.

## The ledger

Absolute effects are given because relative ones on a small base exaggerate. All
intervals are percentile bootstrap at 95%.

| # | claim | comparison | cohort / split | node/pair | micro/macro | threshold | seeds | absolute | relative | CI resamples | artifact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | The pair head beats a predicted-marginal outer product | model vs `outer_product` baseline | sabdab2 validation | pair | macro | n/a (AUPRC) | 10 | +0.003113 | +16.7% | seeds | `baseline_margin_replication_v1/report.json` |
| 2 | Aligned ESM-IF1 beats ESM-2 | `esm_if1_only` vs `esm2` | sabdab2 validation | pair | macro | n/a | 10, paired | +0.005111 | +22.8% | seeds | `representation_attribution_v1/report.json` |
| 3 | Aligned ESM-IF1 beats a width-matched random projection | vs `random_control` | sabdab2 validation | pair | macro | n/a | 10, paired | +0.005706 | +26.1% | seeds | same |
| 4 | The random projection is **not** distinguishable from ESM-2 | `random_control` vs `esm2` | sabdab2 validation | pair | macro | n/a | 10, paired | −0.000595 | −2.6% | seeds | same, **undecided** (bound 0.150) |
| 5 | Shuffled ESM-IF1 is worse than ESM-2 | `esm_if1_shuffled` vs `esm2` | sabdab2 validation | pair | macro | n/a | 10, paired | −0.003093 | −13.8% | seeds | same, decided (bound 0.005 on the *worse* direction) |
| 6 | Aligned beats shuffled | `esm_if1_only` vs `esm_if1_shuffled` | sabdab2 validation | pair | macro | n/a | 10, paired | +0.008204 | +42.4% | seeds | same |
| 7 | Displacement damage saturates by ~8 **rows** | k-ladder paired to k=0 | sabdab2 validation | pair | macro | n/a | 10, paired | k=8 −0.0175, global −0.0187 | 94% of global at k=8 | seeds | `correspondence_corruption_curve_v1/curve_esm_if1.json` |
| 8 | Displacement harms a **sequence** representation too | AntiBERTy k-ladder | sabdab2 validation | pair | macro | n/a | 10, paired | k=1 −0.00273, global −0.00614 | 44% of global at k=1 | seeds | `curve_antiberty_sabdab2.json` |
| 9 | Node and edge gradients oppose early, agree later | per-step cosine | AsEP + sabdab2, training | shared trunk | per step | n/a | 3 per cohort × 2 toolchains | crossings 77–110 (AsEP), 38–50 (sabdab2) | n/a | not resampled; every run classified | `gradient_alignment_dynamics_v1/step_trace_summary.json` |
| 10 | Benchmark position, group split | vs published | AsEP `epitope_group` **test** | node (epitope) | macro | selected on validation, max per-complex MCC | 10 | MCC 0.157 | vs 0.171 main-table / 0.188 best configuration | n/a, single reading | `asep_benchmark_v1/` |
| 11 | Benchmark position, ratio split | vs published | AsEP `epitope_ratio` **test** | node (epitope) | macro | same | 10 | MCC 0.258 | vs 0.410 best, and below WALLE's paper 0.305 | n/a, single reading | `asep_benchmark_v1/ratio_test_measurement.json` |
| 12 | Width effect under `total` selection | d256 vs d128 | AsEP `epitope_group` **validation** | node (epitope) | macro | **best-on-split**, not comparable to published | 10, paired | +0.0278 | n/a | seeds | `checkpoint_selection_confirmation_v1/stage_two_group_validation.json` |
| 13 | That configuration did not confirm | vs prior frozen configuration | AsEP `epitope_ratio` **test** | node (epitope) | macro | validation-selected | 10 | 0.2551 vs 0.2584 | −0.0033 | seeds | `ratio_test_confirmation.json` |

## Rules this ledger enforces

**A row may not borrow another row's seed count, sign bound, or cohort.** That is
the failure it was built to catch.

**Relative figures name their definition.** Mean of per-seed ratios throughout.

**Best-on-split numbers are marked.** Row 12's thresholds are chosen on the split
they are scored on. It selects a hyperparameter; it is not comparable to a
published row and may not be placed beside one.

**Undecided is a value.** Row 4 is undecided, not equal. It is the row that makes
row 3 meaningful, and reporting it as "no difference" would misstate both.

**Test rows are single readings.** Rows 10, 11 and 13 come from splits with a
registered access protocol. They have no seed-level interval over repeated test
access because there was none, and both splits are now spent.

## Open

Row 9 has no interval because the classification is per run and every run agrees;
a resampling unit for it has not been argued for and it is reported as a count.

The complex-level variance decomposition for rows 2–6 is being computed. Until
it lands, those rows describe repeatability under training randomness only, and
nothing in them says whether the advantage is spread over complexes or carried
by a few.
