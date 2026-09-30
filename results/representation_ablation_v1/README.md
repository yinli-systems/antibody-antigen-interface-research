# Representation ablation v1

Status: development-only paired ablation on the validation split. It compares
two input representations and authorizes no test access and no model selection.
The frozen test split was never named and never read.

Date: 2026-09-10

Machine-readable: [`report.json`](report.json) (paired metrics and the
final-epoch comparison), [`width_control.json`](width_control.json) (the
width-matched control), [`arms.json`](arms.json) (every arm against `esm2`),
[`margin_esm2.json`](margin_esm2.json) and
[`margin_esm2_antiberty_esmif1.json`](margin_esm2_antiberty_esmif1.json) (each
arm against its own outer-product baseline). Predictions stay on the cluster.

## Why this experiment

The ten-seed replication established that this model beats the
predicted-marginal outer product by 16%, and that every seed of both arms peaks
at **epoch 1** and then overfits for eleven more. 763,401 parameters against
3,229 training instances is data-limited, and the lever for a data-limited model
is better frozen features, not more capacity.

> **Scope of the width control, checked 2026-09-11.** `random_control` matches
> `esm2_antiberty_esmif1` exactly — 2306/1793 input widths and 963,471
> parameters against the same — and that is the pair this control was built for.
> It is **wider than `esm_if1_only`** (1793/1793, 896,781), so any comparison
> against that arm is across widths, in the conservative direction. Only
> `esm_if1_shuffled` is width-matched to the aligned arm.

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

## Result: ten fresh seeds reproduce it, and twenty are unanimous

Seeds 11-20 were added under a stopping rule fixed in advance
(`protocols/REPRESENTATION_ABLATION_EXTENSION_V1.md`). They took no part in
anything reported above.

| macro auprc | n | ESM-2 | treatment | paired difference | 95% CI | seeds | sign bound |
|---|---:|---:|---:|---:|---|---:|---:|
| seeds 1-10, as first reported | 10 | 0.022460 | 0.028453 | +0.005993 (+26.7%) | [+0.004463, +0.007396] | 10/10 | 0.7411 |
| seeds 11-20, fresh | 10 | 0.021069 | 0.028350 | +0.007282 (+34.6%) | [+0.006080, +0.008488] | 10/10 | 0.7411 |
| **all twenty** | **20** | 0.021764 | 0.028402 | **+0.006638 (+30.5%)** | [+0.005598, +0.007625] | **20/20** | **0.8609** |

Twenty seeds out of twenty favour the treatment. The pooled estimate, +30.5%,
lands on the width-matched estimate of +30.1% from an independent comparison.

One detail is worth more than the headline. Across the fresh seeds the **control
fell** (0.022460 → 0.021069) while the **treatment barely moved** (0.028453 →
0.028350). The larger relative gain in the extension comes mostly from the
control being worse, which is to say the treatment is **less sensitive to the
seed**. That is what a data-limited model looks like when the constraint is
loosened: with stronger features it depends less on what a small sample happened
to teach it.

## Result: it is the structure encoder, and the antibody language model hurts

Two single-encoder arms decompose the gain. Ten seeds each, paired against
`esm2`.

| macro auprc | paired difference | 95% CI | seeds | verdict |
|---|---:|---|---:|---|
| `antiberty_only` | **−0.001773 (−7.9%)** | [−0.002820, −0.000768] | **1/10** | **decided — worse** |
| `esm_if1_only` | **+0.005111 (+22.8%)** | [+0.003780, +0.006439] | 10/10 | decided |
| `random_control` | −0.000595 (−2.6%) | [−0.002151, +0.000970] | 4/10 | no decision |
| `esm2_antiberty_esmif1` | +0.006638 (+30.5%) | [+0.005598, +0.007625] | 20/20 | decided |

And against each single-encoder arm:

| macro auprc | paired difference | 95% CI | seeds | verdict |
|---|---:|---|---:|---|
| both vs `antiberty_only` | +0.007766 (+37.5%) | [+0.006511, +0.008964] | 10/10 | decided |
| both vs `esm_if1_only` | +0.000883 (+3.2%) | [−0.000327, +0.002134] | 7/10 | **no decision** |

Three things follow.

**AntiBERTy on its own makes the model worse**, and that is decided: nine seeds
of ten move down and the interval lies entirely below zero. An encoder trained
specifically on antibodies, added to the antibody side, costs 7.9%.

**ESM-IF1 on its own delivers almost all of the gain.** The lever is
*structure*, not antibody-specific sequence statistics.

**Adding AntiBERTy on top of ESM-IF1 buys nothing measurable.** The interval
contains zero at 7/10 seeds.

So the configuration to carry forward is not the arm that produced the headline.
It is **ESM-2 + ESM-IF1**: narrower (1793 / 1793 against 2306 / 1793), smaller
(896,781 parameters against 963,471), and statistically indistinguishable from
the pair. Simpler, smaller, and as good.

The two encoders do interact. Summing the single-encoder differences gives
+0.003338 against +0.005993 observed together, so AntiBERTy is harmful alone and
not harmful alongside ESM-IF1. That ratio is **not** reported as "1.8x
superadditive": one of the two components is negative, which makes the ratio a
misleading way to describe what is really an interaction, and ten seeds cannot
establish an interaction anyway.

This also confirms the width control from a second direction. `random_control`
is **wider** than `esm_if1_only` (2306 against 1793) and scores 2.6% *below*
`esm2`, while `esm_if1_only` scores 22.8% above it. Width does not explain the
gain; its sign is not even the same.

## Result: the marginals improved too, and the margin grew faster than they did

The outer-product baseline is built from each model's own marginals, so the two
arms have different baselines and the treatment cannot be compared to the
control's.

Every arm establishes a margin over its own baseline, at every seed it was run
for. The share of the prediction that is partner-specific separates them:

| arm | n | model | own baseline | margin | share of baseline | *d* |
|---|---:|---:|---:|---:|---:|---:|
| `esm2` | 20 | 0.021764 | 0.018972 | +0.002792 | 14.7% | 1.833 |
| `random_control` | 10 | 0.021865 | 0.019303 | +0.002562 | 13.3% | 1.819 |
| `antiberty_only` | 10 | 0.020687 | 0.018244 | +0.002443 | 13.4% | 2.261 |
| `esm_if1_only` | 10 | 0.027571 | 0.022498 | +0.005073 | **22.5%** | 3.145 |
| `esm2_antiberty_esmif1` | 20 | 0.028402 | 0.022296 | +0.006106 | **27.4%** | 3.356 |

Only the structure encoder raises that share. Extra width (`random_control`,
13.3%) and AntiBERTy (13.4%) both **lower** it against `esm2`'s 14.7%: those two
arms lean more on the marginals and less on the pairing, which is the opposite
of what this workstream is trying to build.

At ten seeds the two preregistered criteria **split** on the increase: the
bootstrap interval excluded zero while only 8 of 10 seeds were favourable, so
the exact sign lower bound was 0.4931 against a 0.5 threshold — meaningful but
not significant. Ten more seeds were then run under a stopping rule fixed in
advance (`protocols/REPRESENTATION_ABLATION_EXTENSION_V1.md`): twenty seeds,
analysed once, reported whatever it gave, with no further seeds to be added on
the basis of the outcome.

| paired increase in margin | n | mean | 95% CI | seeds | sign bound | verdict |
|---|---:|---:|---|---:|---:|---|
| seeds 1-10, as first seen | 10 | +0.002603 | [+0.000805, +0.004032] | 8/10 | 0.4931 | criteria split |
| seeds 11-20, fresh | 10 | +0.004025 | [+0.002858, +0.005282] | 10/10 | 0.7411 | both met |
| **all twenty — the endpoint** | **20** | **+0.003314** | [+0.002180, +0.004318] | **18/20** | **0.7174** | **established** |

**Both criteria are met at twenty seeds.** The increase in the partner-specific
margin is established.

At twenty seeds each arm's own margin is:

| | ESM-2 | + AntiBERTy + ESM-IF1 |
|---|---:|---:|
| model | 0.021764 | 0.028402 |
| own outer-product baseline | 0.018972 | 0.022296 |
| margin over own baseline | +0.002792 | **+0.006106** |
| 95% CI | [+0.002172, +0.003448] | [+0.005322, +0.006863] |
| seeds | 20/20 | 20/20 |
| effect size *d* | 1.833 | **3.356** |
| **margin as a share of own baseline** | **14.7%** | **27.4%** |

The partner-specific share of the prediction nearly doubles. The baseline also
rose, unanimously (+0.003324, CI [+0.002604, +0.003998], 20/20), so better
features improved the marginals as well as the pairing — but the pairing gained
faster.

What makes this believable is the order the work was done in, not the numbers.
The stopping rule was committed before the seeds were submitted, and it said the
result would be reported whatever it was. Without that, "ten seeds missed, so we
ran twenty and it passed" is indistinguishable from searching for a threshold.

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
