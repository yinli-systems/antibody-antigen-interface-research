# Representation arms: analysis plan

Registered 2026-09-10, **while the arms were still training and before any of
their evaluations existed**. The two arms that already have results
(`esm2`, `esm2_antiberty_esmif1` at seeds 1-10) are not re-decided here.

## The arms

| arm | antibody width | antigen width | parameters | what it adds |
|---|---:|---:|---:|---|
| `esm2` | 1280 | 1280 | 763,401 | — |
| `antiberty_only` | 1793 | 1280 | 830,091 | AntiBERTy, antibody side only |
| `esm_if1_only` | 1793 | 1793 | 896,781 | ESM-IF1, both sides |
| `random_control` | 2306 | 1793 | **963,471** | fixed random projections of the base |
| `esm2_antiberty_esmif1` | 2306 | 1793 | **963,471** | both encoders |

`random_control` and the encoder arm are identical in width and parameter count.
Nothing distinguishes them but the contents of the added columns.

## The question each comparison answers, fixed now

**1. Does width alone explain the gain?** `random_control` vs `esm2`.

The added columns are a fixed linear function of the base embedding, so a model
that gains here gained from the wider input projection, not from information.

- If this comparison is **decided and positive**, the headline +26.7% is partly
  a width effect and **must be restated** as the `random_control` comparison
  below. The abstract number to report becomes the smaller one.
- If it is **not decided**, width alone does not account for a measurable share,
  and the comparison against `esm2` stands as reported.
- If it is **decided and negative**, the extra width costs something, and the
  encoders are overcoming that cost as well as adding signal.

**2. What do the encoders contribute, free of the width confound?**
`esm2_antiberty_esmif1` vs `random_control`.

This is the number that survives review. Decision rule, same as everywhere in
this project: exact sign lower bound > 0.5 **and** bootstrap CI excluding zero.

**3. Which encoder does the work?** `antiberty_only` and `esm_if1_only`, each
against `esm2`.

Reported descriptively. These two arms are not width-matched to each other or to
anything, so they decompose the gain but do not isolate content. Any statement
of the form "encoder X is responsible" is a statement about this dataset and
this width, and must say so.

**4. Are the two encoders additive?** Compare the sum of the two single-encoder
gains against the gain of the combined arm.

No threshold is set for this, because there is no preregistered effect size for
it. It is reported as an observation. A combined gain that is far below the sum
would suggest the encoders carry overlapping information; far above would
suggest an interaction. Neither will be claimed as established from ten seeds.

## What will not be done

No arm will be added on the basis of these results in order to move a criterion
across its threshold. The twenty-seed extension registered in
`REPRESENTATION_ABLATION_EXTENSION_V1.md` is the only extension, its endpoint is
fixed there, and it is analysed once.

If comparison 2 comes out undecided, the recorded finding is that the encoders'
content is not established as the source of the gain at ten seeds — not that it
is absent, and not a reason to run more seeds of it.

## Metrics

`auprc` is the endpoint. `auroc`, `mcc`, `f1` and `precision_at_true_contact_count`
are reported alongside it for every comparison, because agreement across metrics
was what separated this result from the decoder ablation, where they disagreed
in direction. No metric is promoted to the endpoint after the fact.

Metrics are read from the evaluation receipts, which agree with recomputation
from the prediction parquets to 1.7e-17 on this data.
