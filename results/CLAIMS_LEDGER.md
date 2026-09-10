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

**Relative figures are bound per number, not unified.** The replication README
reports +16.1% and the table above +16.7% for the same run: the first is
`mean(margin) / mean(baseline)`, the second the mean of per-seed ratios. Both are
correct and they are different estimands.

An earlier version of this file declared that the project "uses the mean of
per-seed ratios". It does not — `aggregate_representation_ablation.py` computes
`mean_difference / control_mean`, a ratio of means, and renaming a number in a
table would not have changed what the code produced. The aggregators now record
`relative_estimand` in their own receipts, and where the per-seed data exists
both estimands are emitted under separate names. Nothing was renamed.

**The absolute paired difference is the primary report.** Percentages are an aid
to the reader and nothing more. These baselines are small — macro pair AUPRC
around 0.02 — so a relative figure magnifies an absolute difference of a few
thousandths, and a paper that leads with the percentage misleads.

**"Decided" is an internal protocol state, not a reported statistic.** Three
different rules are registered across this project's protocols and they are not
reconciled after the fact:

| rule id | where | criterion |
|---|---|---|
| `bootstrap_ci_excludes_zero` | representation ablation | the percentile bootstrap 95% interval of the mean paired difference excludes zero |
| `sign_bound_exceeds_half` | corruption ladder | the exact Clopper-Pearson lower bound on the fraction of seeds in the stated direction exceeds 0.5 |
| `sign_bound_exceeds_half_and_ci_above_sesoi` | baseline margin | both of the above, against a pre-set smallest effect worth caring about |

Each number carries its own `decision_rule_id`. The paper reports the mean
difference, the interval, the count of favourable seeds and the registered
criterion, and does not use the word "decided" as if it were a statistical term.

**A small bound in one direction is not proof of the other.** An improvement
probability whose lower bound is far below 0.5 does not establish worsening; the
bound for the opposite direction has to be computed, and the aggregators do
compute it.

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
| 14 | Deleting the ranking term improves **both** endpoints | `node_edge` vs `node_edge_rank` | AsEP `epitope_group` **validation** | both | macro (pair AUPRC; node MCC at fixed 0.5) | fixed 0.5 for the node metric | 10, paired | pair +0.0214, node MCC +0.0369 | pair +31% | seeds | `objective_component_localisation_v1/PART_TWO.md` |
| 15 | Contact supervision costs nothing detectable on the node side | `node_edge` vs `node_only` | same | node | macro | fixed 0.5 | 10, paired | MCC −0.0026 (3/10), AUPRC −0.0055 (5/10) | n/a | seeds | same, **both undecided** |
| 16 | Contact supervision transforms pair prediction | `node_edge` vs `node_only` | same | pair | macro | n/a | 10, paired | +0.0833 | 0.0905 against 0.0072 | seeds | same |
| 17 | Ranking alone damages node learning | `node_rank` vs `node_only` | same | node | macro | fixed 0.5 | 10, paired | −0.0748 | n/a | seeds | same |
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

Rows 14-17 use the node MCC at a **fixed 0.5** threshold, not a validation-selected
one, because the arms are compared to each other rather than to a published row
and a per-arm selected threshold would let calibration differences masquerade as
learning differences. The threshold-free node AUPRC is reported beside it and
agrees in every case.

Row 9 has no interval because the classification is per run and every run agrees;
a resampling unit for it has not been argued for and it is reported as a count.

The complex-level variance decomposition for rows 2–6 is being computed. Until
it lands, those rows describe repeatability under training randomness only, and
nothing in them says whether the advantage is spread over complexes or carried
by a few.
