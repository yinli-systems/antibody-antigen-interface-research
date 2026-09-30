# Cold reproduction: all eight results reproduce

Run from a `git archive` of the analysis tree, in processes with no access to
this project's prose — no README, no inventory, no ledger, no hand-written table,
no generated figure. Comparison targets are the frozen JSON receipts, which are
outputs of the same scripts.

Pass rule, fixed before the first run: every numeric leaf within **1e-10**, every
bootstrap interval exact, every script exiting zero.

## Result

| step | seconds | numeric mismatches | note |
|---|---|---|---|
| `baseline_margin` | 2915 | **0** | 37 values bit-identical, max delta 0.00e+00 |
| `attribution_aligned_vs_shuffled` | 10564 | **0** | max delta 5e−18, via a written-out field map |
| `corruption_esm_if1` | 7 | **0** | |
| `corruption_antiberty` | 6 | **0** | |
| `gradient_step_trace` | 0.5 | **0** | four cohorts |
| `selection_stage_two` | 12 | **0** | |
| `ratio_test_confirmation` | 12 | **0** | |
| `complex_heterogeneity` | 616 | **0** | 20 runs recomputed from parquets |

**Every number in this project recomputes from frozen inputs.** The two heaviest
steps — three hours and ten minutes of recomputation from prediction parquets —
land on their frozen values to 5e−18 and exactly.

## It took three passes, and the failures are recorded

The first pass returned FAIL with five of eight steps unverified. The causes were
one bookkeeping gap, three defects in the driver, and one schema change. **None
was a case of the same command on the same inputs producing different numbers.**

| pass | what failed | why |
|---|---|---|
| 1 | `baseline_margin`, 35 numeric | the driver read `repr_ablation/out/esm_if1_only`; the receipt comes from `margin_replication/out` |
| 1 | `corruption` ×2, 24 field-presence | receipts frozen before `decision_rule` fields existed |
| 1 | `gradient_step_trace`, 2 field-presence | the plan passed two cohorts where the receipt has four |
| 1 | `attribution`, not compared | the plan passed no comparison target |
| 2 | `baseline_margin`, killed | SIGKILL on the login node at 69 minutes, after completing there once at 76 |
| 3 | `baseline_margin`, 3 field-presence | receipt frozen before `decision_rule` and `relative_estimand` existed |

The first of these is the one worth keeping. The claims ledger named the artifact
file and not the run root that produces it, and this project contains two
similarly named ten-seed arms whose margins differ by a factor of 1.6. The
driver now takes the margin runs as a separate argument.

The SIGKILL is recorded as unexplained. The same script completed on that node
once and was killed there once; the node carries 273 users, and either a watchdog
or transient memory pressure would look identical from outside. It was moved to a
compute node rather than diagnosed by guesswork, and the receipt says so.

## How the final PASS was reached, stated plainly

Four receipts were frozen before three metadata fields were added to the
aggregators today. They were regenerated from the cold outputs, whose numeric
content is identical — verified field by field before overwriting, 37 values at
delta 0.00e+00 for `baseline_margin`.

**That is a refresh of stale receipts, not a second verification.** The numeric
claim rests on the comparisons in the table above, each made against a receipt
produced before the run that checked it.

## Receipts

`receipt_pass1.json`, `receipt_pass2.json`, `receipt_pass3.json`. Each records
the analysis commit, the digests of `scripts/` and `src/`, the digest of every
output, the comparison target and its digest, the Python version, the platform,
and the tolerance.
