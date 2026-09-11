# Cold reproduction, first pass: FAIL

Run 2026-09-11 from a `git archive` of commit `80a220d`, on a login node, in a
process with no access to this project's prose. Receipt: `receipt_pass1.json`.

**Verdict: FAIL.** The pass rule was fixed before the run — every numeric leaf
within 1e-10, every bootstrap interval exact, every script exiting zero — and
five of eight steps did not verify on the first attempt. That is recorded as a
failure and not softened.

## What happened, step by step

| step | seconds | result |
|---|---|---|
| `selection_stage_two` | 11.5 | **MATCH** |
| `ratio_test_confirmation` | 12.3 | **MATCH** |
| `complex_heterogeneity` | 615.5 | **MATCH** |
| `attribution_aligned_vs_shuffled` | 10563.7 | no comparison target was wired |
| `corruption_esm_if1` | 9.2 | 12 mismatches, **0 numeric** |
| `corruption_antiberty` | 5.6 | 12 mismatches, **0 numeric** |
| `gradient_step_trace` | 0.4 | 2 mismatches, **0 numeric** |
| `baseline_margin` | 4590.5 | 35 numeric mismatches |

## Diagnosis

**No step produced different numbers from the same command on the same inputs.**
Every failure is one of three things, and all three are defects in the driver or
in bookkeeping rather than in the science.

**The driver read the wrong arm.** `baseline_margin` was pointed at
`repr_ablation/out/esm_if1_only`; the frozen receipt comes from
`margin_replication/out`. Every per-seed model, baseline and margin differed,
which is the signature of a different dataset rather than a different
computation. Confirmed by reading the correct root: its `mean_margin` matches
the frozen receipt to 1e-10.

This is the most useful thing the exercise found. The claims ledger named the
artifact file and not the run root that produces it, and two similarly named arms
with ten seeds each exist in this project. `--margin-runs` is now a separate
argument whose help text says so.

**New fields.** The corruption receipts were frozen before `decision_rule` and
`decision_rule_id` were added to the aggregators. All 24 mismatches across the
two ladders are field-presence, none numeric. The frozen receipts were
regenerated from the cold outputs, whose numbers are identical, so the frozen
artifacts now match what the current code emits.

**A narrower invocation.** `gradient_step_trace` was given two cohorts where the
frozen receipt has four; the 5090 replication cohorts were missing from the plan.

**No comparison target.** The attribution step ran for 2.9 hours and was compared
against nothing, because the plan passed `None`. The two scripts record the same
quantity under different keys — `mean_difference` against `paired_difference` —
so a whole-file comparison would have reported a total mismatch and said nothing
about the numbers. A written-out field map was added instead of inferring one.

## The headline number, verified

With the map applied to the first pass's own output:

| field | cold | frozen | difference |
|---|---|---|---|
| mean difference | 0.008203640685771799 | 0.008203640685771797 | 2e−18 |
| bootstrap CI low | 0.006927522500949097 | 0.006927522500949101 | 4e−18 |
| bootstrap CI high | 0.009451685065017045 | 0.009451685065017040 | 5e−18 |
| sign lower bound | 0.741134449106947701 | identical | 0 |
| favourable seeds | 10 | 10 | — |

**0 mismatches.** Aligned minus shuffled — the centrepiece of RQ1 — recomputes
from the prediction parquets in a clean process to within 5e−18.

## What this first pass establishes and does not

It establishes that four results reproduce exactly from frozen inputs, including
one that takes ten minutes of recomputation and one that takes three hours.

It does not establish that the suite passes. Three steps are re-running under the
corrected plan, and until they return the verdict stands at FAIL.
