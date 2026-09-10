# Which relational objective is the problem?

Registered 2026-09-11, **before the three-way gradient geometry was read and
before any ablation arm was trained**. The two-way in-situ measurement that
motivates it is already on disk and is quoted below; nothing else is.

## Where this came from

Four observations were recorded separately and each was explained on its own:

1. Removing pair supervision leaves node metrics unchanged or slightly better.
2. A λ sweep found a joint optimum near 0.25 rather than at either extreme,
   which was written up as loss-scale imbalance.
3. Selecting the checkpoint on `partner_rank` picks a worse epoch than selecting
   on total loss, worth +0.028 mean per-complex MCC on group validation.
4. PCGrad projects on ~1.6 of 2 ordered pairs per step, against a registered
   prediction of zero based on a checkpoint cosine of +0.70.

Observation 4 turned out to rest on a decomposition mismatch: the +0.70 was
measured with the partner-ranking term absent from both tasks, because
`interaction_first_loss` returns zero for it without negative scores. Measured
in situ with negatives present, the cosine between the node gradient and the
combined relational gradient is **−0.059 and −0.050** in the two smoke epochs,
with the relational gradient 1.7-2.4× the node gradient in norm.

That reopens all four. The hypothesis this protocol tests is that they are one
finding:

> **The optimisation pathology localises to the ranking auxiliary objective
> rather than to relational supervision as a whole.**

## Part one: the three-way geometry

The objective is partitioned into exactly three task losses whose sum is checked
against the full loss at every step:

| task | terms |
|---|---|
| node | epitope and paratope BCE, dice, count, consistency |
| edge | contact-map supervision, at its configured weight |
| rank | complex-level partner ranking, at its configured weight |

On real training batches, with the negative samples the ranking term requires,
on the **shared trunk** — everything that is not a node head and not the pair
decoder — the `measure` arm reports every pairwise cosine and every task
gradient norm, averaged over the epoch:

    cos(g_N, g_E)   cos(g_N, g_R)   cos(g_E, g_R)
    ||g_N||         ||g_E||         ||g_R||

Norms are of the **weighted** gradients, since the task losses carry their
configured λ. The gradient is linear in that scalar, so the unweighted norms are
these divided by λ, and both λ values are already recorded in the run config.

`measure` applies the ordinary summed gradient, so this costs the geometry and
changes nothing: a unit test asserts it lands on the same gradient as `none`.

### What each outcome would mean, fixed now

**`cos(g_N, g_E) > 0` and `cos(g_N, g_R) < 0`** — the hypothesis holds. Contact
supervision is compatible with node representation; ranking is the mismatch.

**Both negative** — relational supervision as a whole conflicts, the loss-scale
story stands, and the ranking term is not special.

**Both positive** — the aggregate negative cosine comes from the interaction of
the two rather than from either, and the hypothesis is refuted in a way that
needs its own explanation. This outcome is reported, not explained away.

`cos(g_E, g_R)` is reported in all three cases. A strongly negative value there
would say the two relational objectives fight each other, which no current
result anticipates.

## Part two: the causal ablation

Geometry is correlational. Four arms, ten seeds each, AsEP `epitope_group`,
**validation only** — the group test split is permanently frozen and is not read
for this or anything else.

| arm | `--edge-loss-weight` | `--partner-rank-weight` |
|---|---|---|
| node only | 0 | 0 |
| node + edge | configured | 0 |
| node + rank | 0 | configured |
| node + edge + rank | configured | configured |

No new code: both flags exist and are train-only. The node-only arm already
exists as the supervision-parity reference and is re-run here so all four arms
share seeds, data and code version.

### Endpoints, fixed now

Node side: mean per-complex epitope MCC. Pair side: macro pair AUPRC. Both
reported for every arm. Paired by seed against `node + edge + rank`, with
percentile bootstrap intervals and the exact sign bound, as everywhere else.

An arm counts as better or worse only when the sign bound clears 0.5. Arms that
do not clear it are reported as undecided rather than ranked.

### The prediction

If the hypothesis holds: **node + edge** keeps pair performance and improves the
node metric over the full objective, while **node + rank** is worse than
**node + edge** on both. The claim then available is that the pathology is in
the ranking objective specifically, not in relational supervision.

If **node + edge** does not improve the node metric, the hypothesis fails on its
central prediction whatever the geometry says, and the finding is recorded as
loss-scale imbalance across the relational branch as previously written.

## What this does to the balancer intervention

`MULTITASK_BALANCER_INTERVENTION_V1.md` registered PCGrad as a negative control
on the strength of the +0.70 cosine. That premise is now known to be false for
the objective actually trained, so **PCGrad is a treatment here, not a control**,
and `results/gradient_conflict_v1/` must be reported with its scope narrowed to
node versus contact edges — which is what it measured.

The balancer arms are not launched until this protocol's part one is read. If
the pathology localises to the ranking term, the informative comparison is no
longer which gradient-surgery method wins but whether any of them recovers what
deleting the term recovers, and that comparison is registered separately once
the ablation is in.
