# Geometry superiority gate: power and decision-rule analysis v1

Status: non-authorizing design analysis. It revises no performance number, does
not re-run the gate, does not reverse its recorded outcome, and authorizes no
model selection, outer-fold inference, or test access. **The preregistered gate
remains failed.**

Date: 2026-09-08

Machine-readable receipt:
[`GEOMETRY_GATE_POWER_ANALYSIS_V1.json`](GEOMETRY_GATE_POWER_ANALYSIS_V1.json).
Regenerate with `python3 scripts/analyze_geometry_gate_power.py --json`.

## Why this exists

[`FINAL_GEOMETRY_RESEARCH_REPORT.md`](FINAL_GEOMETRY_RESEARCH_REPORT.md) records
that distance+direction beat topology-only on every reported endpoint, that the
joint bootstrap interval excluded zero, and that the gate failed anyway. The
tracked evidence answers whether the effect was *detected*. It does not answer
whether the experiment could have detected it.

No tracked artifact in this repository reports statistical power, a minimum
detectable effect, or a decision rule other than a difference-of-means test on a
single fit. Without those quantities a failed gate is ambiguous between "there
is no effect" and "this design cannot see an effect of this size", and the two
call for opposite next actions. This analysis supplies the missing quantities
from the already-audited summary statistics.

Inputs, bound by SHA-256 in the receipt:

| Artifact | Role |
|---|---|
| `ab03_a100_replay_verification.json` | per-seed differences, three A100 seeds |
| `a100_checkpoint_paired_statistics_summary.json` | A100 single-checkpoint paired statistics |
| `paired_statistics_summary.json` | RTX 5090 single-checkpoint paired statistics |

## 1. Three seeds cannot clear the sign criterion at any effect size

Across the three A100 seeds, distance+direction minus topology-only:

| Endpoint | Mean | Sample SD | Effect size `d` | Positive seeds | Exact P(A>B) lower bound | Significant |
|---|---:|---:|---:|---:|---:|:--:|
| joint macro AUPRC | +0.009165 | 0.008422 | 1.088 | 3/3 | 0.3684 | no |
| paratope AUPRC | +0.015614 | 0.010297 | 1.516 | 3/3 | 0.3684 | no |
| epitope AUPRC | +0.002716 | 0.006965 | 0.390 | 2/3 | 0.1354 | no |

The bound is the exact Clopper-Pearson one-sided limit. For a unanimous sample
it reduces to `alpha ** (1 / n)`, so at `alpha = 0.05`:

```
n = 3  ->  0.05^(1/3) = 0.368     n = 4  ->  0.473     n = 5  ->  0.549
```

**A unanimous result needs at least five replicates before the criterion
`P(A>B) > 0.5` is reachable at all.** Three seeds could not have passed however
large the true effect was. The three historical seeds are therefore not "almost
enough"; they are structurally incapable of supporting the claim, which is an
argument for the preregistered five-seed design rather than against it.

A percentile bootstrap is not an alternative here: bootstrapping an all-positive
sample of three yields a degenerate point mass at 1.0.

## 2. The five-seed design is itself underpowered for the observed effect

Minimum detectable standardised effect, one-sided, `alpha = 0.05`, power 0.80:

| Replicates | Detectable `d` |
|---:|---:|
| 3 | 2.297 |
| 5 | 1.359 |
| 8 | 0.978 |
| 10 | 0.853 |
| 20 | 0.577 |

The observed joint effect is `d = 1.088`, **below the 1.359 that five seeds can
detect**. Power for the joint endpoint is 0.357 at three seeds and 0.640 at
five; reaching 0.80 needs **seven**. Paratope (`d = 1.516`) is the only endpoint
five seeds can carry. Epitope (`d = 0.390`) would need 43.

So running the five preregistered seeds as specified is expected to fail the
joint endpoint roughly a third of the time even if the effect is exactly as
observed. That is a property of the design, not of the model.

## 3. The multiplicity family contains a variable that cannot vary

`joint macro AUPRC` is the arithmetic mean of `paratope AUPRC` and
`epitope AUPRC`. Checked against the recorded mode values, the largest residual
is `5.55e-17` — machine precision. The composite is determined by its two
components and carries no independent information.

| Endpoint | Unadjusted | Holm over all three | Holm over the two components |
|---|---:|---:|---:|
| joint macro AUPRC | 0.027497 | 0.074393 | — |
| paratope AUPRC | 0.024798 | 0.074393 | **0.049595** |
| epitope AUPRC | 0.195180 | 0.195180 | 0.195180 |

Whether the family should hold two or three members is a preregistration
question, not something this analysis can settle after the fact. It is recorded
only so the choice is visible: the family that was used charges for a third
hypothesis that is a deterministic function of the other two, and under the
two-component family paratope falls below 0.05.

## 4. The field named `holm_adjusted_one_sided_p` holds a Bonferroni value

Both checkpoint receipts record a field named for Holm whose value is `3 x p`:

| Receipt | Recorded | Holm step-down | Bonferroni | Value is |
|---|---:|---:|---:|:--|
| `a100_checkpoint_paired_statistics_summary.json` | 0.0824918 | 0.0743926 | 0.0824918 | Bonferroni |
| `paired_statistics_summary.json` | 0.0821918 | 0.0547945 | 0.0821918 | Bonferroni |

Holm is uniformly at least as powerful as Bonferroni, so the receipts understate
the evidence they report. **Neither correction changes the recorded outcome** —
both exceed 0.05 on the joint endpoint — so the gate's failure stands. The
defect is that a stated method was not the method applied, which matters wherever
these receipts are cited as the statistical record.

## 5. The two checkpoint audits disagree materially

The same comparison was audited on two checkpoint sets:

| Endpoint | A100 mean | A100 p | A100 win | 5090 mean | 5090 p | 5090 win |
|---|---:|---:|---:|---:|---:|---:|
| joint macro AUPRC | +0.009876 | 0.027497 | 0.7222 | +0.014215 | 0.027397 | 0.9444 |
| paratope AUPRC | +0.011514 | 0.024798 | 0.6111 | +0.024839 | **0.000100** | 0.9722 |
| epitope AUPRC | +0.008237 | 0.195180 | 0.6389 | +0.003592 | 0.357464 | 0.6667 |

The RTX 5090 paratope result reaches the permutation floor at 10,000 iterations
with 97.2% of complexes positive. The published headline uses the A100 numbers,
which are the weaker of the two. Both are development evidence; nothing here
selects between them, but a report that cites only one should say why.

## 6. Win fractions are reported over units that are not independent

`positive_complex_fraction` is computed over the 36 validation instances, while
the registered resampling unit is the 16 independent endpoint-connected
components. Exact lower bounds differ accordingly:

| Endpoint (A100) | Win fraction | Lower bound over 36 instances | Lower bound over 16 components |
|---|---:|---:|---:|
| joint macro AUPRC | 0.7222 | 0.5745 | 0.5156 |
| paratope AUPRC | 0.6111 | 0.4605 | 0.3910 |
| epitope AUPRC | 0.6389 | 0.4883 | 0.3910 |

The instance-level bound overstates precision by treating correlated complexes
as independent. The component-level bound is the one consistent with the
resampling unit the protocol already registered.

## What follows

1. The recorded failure is uninformative about the effect. At three seeds the
   design had power 0.357 on the joint endpoint; a failure was the likely
   outcome whether or not the effect is real.
2. Re-testing the same artifacts with a different statistic would not fix this
   and is not proposed. The binding constraint is replicate count.
3. A confirmatory run should be powered before it is launched. On the observed
   variance that is **seven seeds** for the joint endpoint at power 0.80, not
   five. If seven fits are not affordable, the alternative supported by the
   benchmarking literature is to randomise more sources of variation per fit —
   data split, initialisation, data order — rather than to add seeds alone,
   which lowers the variance of the estimator at equal cost.
4. Whatever is chosen, the power calculation belongs in the protocol before the
   run, alongside the threshold. A preregistered threshold without a
   preregistered minimum detectable effect can only produce ambiguous failures.

## References

- Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*,
  MLSys 2021 — the probability-of-outperforming decision rule and the case for
  randomising multiple sources of variation.
- Kapoor et al., *REFORMS: Consensus-based Recommendations for
  Machine-learning-based Science*, Science Advances 2024 — items 7b and 7c
  require uncertainty estimates and a justification of the statistical test.
