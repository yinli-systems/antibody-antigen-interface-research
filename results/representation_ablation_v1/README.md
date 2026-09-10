# Representation ablation v1

Status: development-only paired ablation on the validation split. It compares
two input representations and authorizes no test access and no model selection.
The frozen test split was never named and never read.

Date: 2026-09-10

Machine-readable: [`report.json`](report.json) (paired metrics and the
final-epoch comparison), [`width_control.json`](width_control.json) (the
width-matched control), [`margin_esm2.json`](margin_esm2.json) and
[`margin_esm2_antiberty_esmif1.json`](margin_esm2_antiberty_esmif1.json) (each
arm against its own outer-product baseline). Predictions stay on the cluster.

## Why this experiment

The ten-seed replication established that this model beats the
predicted-marginal outer product by 16%, and that every seed of both arms peaks
at **epoch 1** and then overfits for eleven more. 763,401 parameters against
3,229 training instances is data-limited, and the lever for a data-limited model
is better frozen features, not more capacity.

Two encoders were added to the frozen ESM-2 input: **AntiBERTy**, which carries
antibody statistics 3,229 pairs cannot supply, and **ESM-IF1**, which carries
structure the sequence encoder never sees. Ten seeds per arm, the same ten in
both, differenced within a seed.

## Result: every metric improves, and every interval excludes zero

| macro | ESM-2 | + AntiBERTy + ESM-IF1 | paired difference | 95% CI | seeds |
|---|---:|---:|---:|---|---:|
| **auprc** | 0.022460 | **0.028453** | **+0.005993** | [+0.004463, +0.007396] | **10/10** |
| auroc | 0.921790 | 0.932876 | +0.011086 | [+0.009929, +0.012227] | 10/10 |
| mcc | 0.073644 | 0.082334 | +0.008690 | [+0.007489, +0.009978] | 10/10 |
| f1 | 0.022513 | 0.025931 | +0.003418 | [+0.002647, +0.004255] | 10/10 |
| precision at true contact count | 0.029109 | 0.038700 | +0.009591 | [+0.004647, +0.014400] | 9/10 |

AUPRC: **+26.7%**, CI [+19.9%, +32.9%], effect size *d* = 2.355, exact sign
lower bound 0.7411 against a 0.5 threshold. Established on both preregistered
criteria.

**No metric disagrees in direction.** That is the contrast with the decoder
ablation, where bilinear led on AUPRC while hadamard led on AUROC and MCC — the
signature of noise. Here five metrics move together.

AUROC gains only 1.2% because it is near saturation at a 0.09% positive rate;
0.922 was already high and it was never the informative number here. The
operationally meaningful one is precision at the true contact count: taking the
top N pairs where N is the true contact count, **2.9% correct becomes 3.9%**.

## Result: the marginals improved too, and the margin grew faster than they did

The outer-product baseline is built from each model's own marginals, so the two
arms have different baselines and the treatment cannot be compared to the
control's.

| | ESM-2 | + AntiBERTy + ESM-IF1 |
|---|---:|---:|
| own outer-product baseline | 0.019347 | 0.022737 (+17.5%) |
| margin over own baseline | +0.003113 | +0.005716 (+83.6%) |
| 95% CI on that margin | [+0.001969, +0.004290] | [+0.004476, +0.006875] |
| margin as a share of own baseline | 16.1% | **25.1%** |
| effect size *d* | 1.547 | 2.792 |

Both arms establish a margin over their own baseline at 10/10 seeds. The
treatment's is nearly double.

**The evidence for the margin *increase* is weaker than for everything above,
and the two preregistered criteria split on it:**

| paired change in margin | |
|---|---:|
| mean | +0.002603 |
| bootstrap 95% CI | [+0.000805, +0.004032] — excludes zero |
| favourable seeds | **8/10** |
| exact sign lower bound | **0.4931** — below the 0.5 threshold |

Seed 3 moved −0.004166 and seed 10 −0.000055. So the increase is **meaningful
but not significant** under this project's own decision rule. The paired change
in the baseline itself is unanimous by comparison (+0.003391, CI
[+0.002371, +0.004290], 10/10).

The honest reading: absolute performance is established, and the
partner-specific share of it grew from 16.1% to 25.1% of baseline, but the
growth of the margin itself is not established at ten seeds. At an 8/10 rate the
sign bound needs roughly fifteen to twenty seeds to clear 0.5.

## Result: the width confound is measured, and it is not the explanation

The treatment carries 26.2% more parameters than ESM-2 (763,401 → 963,471), all
of it in the input projections that consume the wider input. A gain measured
only against the narrow arm cannot separate the content of the added columns
from their width.

`random_control` settles it. It appends fixed random linear projections of the
base embedding, mirroring the treatment's coverage exactly: **identical widths
(2306 / 1793) and an identical parameter count (963,471)**. The added columns
are a linear function of the model's own input and carry nothing new.

| macro auprc | mean | paired difference | 95% CI | seeds | verdict |
|---|---:|---:|---|---:|---|
| `random_control` vs `esm2` | 0.021865 | −0.000595 (−2.6%) | [−0.002151, +0.000970] | 4/10 | **no decision** |
| `esm2_antiberty_esmif1` vs `random_control` | 0.028453 | **+0.006588 (+30.1%)** | [+0.005277, +0.007869] | **10/10** | **decided** |
| `esm2_antiberty_esmif1` vs `esm2` | 0.028453 | +0.005993 (+26.7%) | [+0.004463, +0.007396] | 10/10 | decided |

**Two hundred thousand extra parameters of width, carrying no new information,
do not help.** The point estimate is slightly negative and the interval contains
zero at 4/10 seeds.

The analysis plan registered before these runs said that if the width-matched
control gained, the headline would have to be restated as the smaller number.
That branch did not fire. The opposite did: because `random_control` sits a
little below `esm2`, removing the confound makes the estimate **larger** and its
interval **narrower**.

Both numbers are true and they mean different things. **+26.7%** is against a
smaller model. **+30.1%** is against a model with exactly the same parameter
count, and is the number that survives the obvious objection.

## The selection metric disagrees between arms, but is not picking bad epochs

The treatment is worse on `partner_rank` — the checkpoint selection metric — in
**10/10 seeds**, while being better on macro AUPRC in **10/10 seeds**. Same
models, both unanimous, opposite directions.

That raised the question of whether selection was picking the wrong epoch. It is
not, at least not against the only available alternative:

| macro auprc | selected (`best.pt`) | final epoch (`last.pt`) | final better in |
|---|---:|---:|---:|
| ESM-2 | 0.022460 | 0.018354 | 1/10 |
| + AntiBERTy + ESM-IF1 | 0.028453 | 0.024719 | 0/10 |

The selected checkpoint wins decisively in both arms. Only two checkpoints are
stored per run, so this compares epoch 1 against epoch 10–12 and cannot say
whether epoch 2 or 3 would be better. **No evidence that selection is harmful**
is the claim, not that it is optimal.

What does survive: `partner_rank` cannot be used to compare models across
representations, since it ranks these two arms in the opposite order from every
metric that was actually reported.

## Reproduction checks

Three independent checks, all exact:

1. The control arm's per-seed macro AUPRC equals the ten-seed replication's,
   digit for digit, across a code change and a different output directory.
2. The control arm's full margin report is identical to the replication's:
   +0.003113, CI [+0.001969, +0.004290], *d* 1.547, 10/10.
3. Two independent code paths for macro AUPRC — recomputed from the prediction
   parquets, and averaged per instance in the evaluation receipt — agree to
   1.7e-17 across ten seeds.

## Limits

1. ~~The treatment has 26.2% more parameters and this is not separated here.~~
   **Resolved above.** The width-matched control was run: extra width alone does
   not help (−2.6%, interval containing zero, 4/10 seeds), and the confound-free
   estimate of the encoders is +30.1%.
2. **Ten instances have no ESM-IF1 artifact** and are zero-filled with a
   presence column set to 0, rather than dropped, so that the cohort stays
   identical to the ten-seed baseline. Five of them are in the training cohort.
3. **Both arms still peak at epoch 1.** Better features raised the ceiling; they
   did not change the shape of the problem.
4. Validation split, one architecture, one dataset. Nothing here transfers to
   the sealed test split.
5. **Absolute performance is still low.** Macro AUPRC 0.0285 against a 0.135%
   prevalence floor is 21x the floor — up from 17.8x — and still means most
   top-ranked pairs are not contacts.

## Provenance

| | |
|---|---|
| split manifest sha256 | `3da484ae31bd6b54c4bec4073cb980c88d9ab42c3213ca30933ff29186d1e396` |
| training | `slurm/representation_ablation.sbatch`, array 1-20, gpu_4090 |
| aggregation | `scripts/aggregate_representation_ablation.py`, `scripts/aggregate_baseline_margin.py` |
| representations | `results/pretrained_representation_coverage_v1/` |
| widths | antibody 1280 + 2x(512+1) = 2306; antigen 1280 + (512+1) = 1793 |
| evaluated | 217 validation complexes, 68 clusters |
