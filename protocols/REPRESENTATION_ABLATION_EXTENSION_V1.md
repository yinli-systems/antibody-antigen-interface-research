# Representation ablation, extension to twenty seeds

Registered 2026-09-10, **before** seeds 11-20 were submitted.

## What the first ten seeds decided, and what they did not

Ten paired seeds put ESM-2 + AntiBERTy + ESM-IF1 26.7% above ESM-2 on macro
AUPRC, at 10/10 seeds, bootstrap CI [+0.004463, +0.007396], sign lower bound
0.7411. That is established and this extension does not revisit it.

One endpoint came out split. Each arm beats its own predicted-marginal outer
product, and the treatment's margin is nearly double the control's, but the
**paired increase in that margin** was favourable in only 8 seeds of 10:

| | |
|---|---:|
| mean | +0.002603 |
| bootstrap 95% CI | [+0.000805, +0.004032] — excludes zero |
| favourable seeds | 8/10 |
| exact sign lower bound | 0.4931 — below the 0.5 threshold |

Meaningful but not significant, under this project's own rule.

## The stopping rule

Seeds 11-20 are added to both existing arms, giving twenty paired seeds.

**The extension is analysed once, at twenty seeds, and the result is reported
whatever it is.** No further seeds will be added to this endpoint on the basis
of its outcome. Extending a run until a criterion is met and stopping there is
how a null result is converted into a false positive, and the only defence that
costs nothing is to fix the endpoint now.

If twenty seeds leave the sign bound below 0.5, that is the recorded finding:
the margin increase is not established, whatever the interval says.

## What is decided at twenty seeds

| endpoint | criterion |
|---|---|
| paired increase in margin over own baseline | exact sign lower bound > 0.5 **and** bootstrap CI excluding zero |
| paired macro AUPRC difference | already established at ten seeds; reported at twenty for completeness, not re-decided |

At an 8/10 favourable rate the sign bound reaches 0.5 at sixteen of twenty, so
twenty seeds can decide this endpoint if the rate holds. It is not guaranteed
to, and that is the point of fixing the analysis now.

## Unchanged from the first ten seeds

Same split manifest (`3da484ae31bd6b54c4bec4073cb980c88d9ab42c3213ca30933ff29186d1e396`),
same cohort, same architecture, same twelve-epoch budget, same selection metric,
same frozen representations. Seeds 11-20 are new only in the seed.
