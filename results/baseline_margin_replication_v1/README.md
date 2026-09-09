# Baseline margin replication v1

Status: development-only replication on the validation split. It establishes a
margin over a baseline, not a benchmark result, and authorizes no test access
and no model selection. The frozen test split was not named and not read.

Date: 2026-09-10

Machine-readable: [`report.json`](report.json) (the aggregate and its decision
rule), [`seed_receipts.json`](seed_receipts.json) (each seed's own evaluation
summary). The prediction parquets stay on the cluster.

## The question

The single-seed ablation put the pair head 32.7% above the predicted-marginal
outer product in macro AUPRC. One observation cannot establish that. Ten seeds
of the *same arm* were trained and evaluated to give the margin an interval.

The baseline scores a pair by `P(epitope_i) * P(paratope_j)`, from the model's
own marginals, so it carries no information about which antibody residue pairs
with which antigen residue. The margin over it is the partner-specific signal.

## Result: the margin is established

| | |
|---|---:|
| seeds with a positive margin | **10 / 10** |
| mean margin (macro AUPRC) | **+0.003113** |
| sd | 0.002012 |
| effect size *d* | 1.547 |
| bootstrap 95% CI | **[+0.001969, +0.004290]** |
| Clopper-Pearson sign lower bound | **0.7411** (needs > 0.5) |
| verdict | **established** |

Both criteria were fixed before the runs. The sign bound is exact and could not
have been met by fewer than five seeds however large the margin; the bootstrap
interval on the mean excludes zero.

## Result: one seed overstated the effect by roughly double

| relative margin over baseline | |
|---|---:|
| single-seed ablation (seed 42) | +32.7% |
| **ten-seed mean** | **+16.1%** |
| ten-seed median | +13.3% |
| range across seeds | +2.8% to +34.6% |

| seed | model | baseline | margin | rel. |
|---:|---:|---:|---:|---:|
| 1 | 0.021606 | 0.018799 | +0.002808 | +14.9% |
| 2 | 0.022107 | 0.020985 | +0.001122 | +5.3% |
| 3 | 0.025881 | 0.020185 | +0.005695 | +28.2% |
| 4 | 0.020490 | 0.018338 | +0.002151 | +11.7% |
| 5 | 0.022239 | 0.020684 | +0.001555 | +7.5% |
| 6 | 0.021659 | 0.019997 | +0.001662 | +8.3% |
| 7 | 0.020893 | 0.016618 | +0.004274 | +25.7% |
| 8 | 0.022293 | 0.021676 | +0.000617 | +2.8% |
| 9 | 0.022572 | 0.016774 | +0.005798 | +34.6% |
| 10 | 0.024861 | 0.019410 | +0.005451 | +28.1% |

The direction replicates unanimously; the magnitude does not. The ablation's
+32.7% sits in the top fifth of the observed range. Both the model and the
baseline vary across seeds — model sd 0.001680 (cv 7.5%), baseline sd 0.001713
(cv 8.9%) — and because the baseline is built from the same model's marginals,
the two move partly together, which is why the paired margin has a tighter
interval than either arm alone.

The number to carry forward is **+16%**, not +33%.

## Result: the epitope side is not partner-specific

Each evaluation reports the difference between a complex's epitope AUPRC given
its true antibody and the mean over wrong antibodies, with a cluster bootstrap
over 68 clusters and 217 instances.

| | seeds |
|---|---:|
| 95% CI contains zero | **9 / 10** |
| 95% CI excludes zero | 1 / 10 |

Per-seed means run from −0.00006 to +0.00197 against interval half-widths of
about 0.002. One seed in ten excluding zero at the 5% level is what a true
effect of zero produces.

This does not contradict the margin above, and the two are measured at different
levels: the pair head beats the outer product **at the pair level**, while the
antigen's own residue-level epitope prediction barely moves when the antibody is
swapped. The partner-specific signal this model carries lives in the pairing,
not in the marginals.

This result was invisible until now — see below.

## How this evidence was recovered

All ten training jobs exited non-zero, and none of them wrote an
`evaluation.json`. Two defects in post-prediction aggregation raised after the
parquets were written:

1. `cluster_bootstrap_mean_ci` was passed a cluster mapping covering the whole
   split while the deltas covered only the evaluated instances.
2. `macro_metric_summary` averaged keys chosen by a hardcoded exclusion list
   rather than by type, so `ranking_at_k`'s string `tie_policy` raised.

Both are fixed. Re-running `evaluate` on the ten stored checkpoints with the
fixes takes four minutes per seed and completes with exit code 0, which is where
`seed_receipts.json` and the partner-sensitivity result come from. The pair
predictions themselves were never affected.

## Limits

1. **Validation split, one architecture, one dataset.** Nothing here transfers
   to the sealed test split, and no test access is claimed or authorized.
2. **Both arms of every seed peaked at epoch 1.** With 3,229 training instances
   against 5.5M parameters the binding constraint is data, not architecture.
3. The split file has drifted from the digest the S2 assignment registered.
   Train membership (3,229) and the sealed pool (434) both still reproduce and
   remain disjoint, but one instance moved between test and validation.
4. The baseline is derived from the same model's marginals, so this measures
   what the pair head adds over its own marginals, not over an independent
   epitope/paratope predictor.
5. **Absolute performance stays low.** Macro AUPRC 0.0225 against a 0.135%
   prevalence floor is a real signal — about 17x the floor — and still means the
   large majority of top-ranked pairs are not contacts.

## Provenance

| | |
|---|---|
| split manifest sha256 | `3da484ae31bd6b54c4bec4073cb980c88d9ab42c3213ca30933ff29186d1e396` |
| training | `slurm/pair_baseline_margin_replication.sbatch`, array 1-10, gpu_4090, ~2 h/seed |
| aggregation | `scripts/aggregate_baseline_margin.py`, `slurm/aggregate_baseline_margin.sbatch` |
| evaluated instances | 217 validation complexes, 68 clusters |
| decoder | bilinear |
