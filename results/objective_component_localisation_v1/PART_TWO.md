# Part two: removing the ranking term improves both endpoints

> **Read `RUN_CONTRACT.md` first.** `selection_metric` drives the learning-rate
> schedule and the checkpoint choice as well as the stopping counter, and all
> four arms monitor a `total` composed of whichever terms they enable. The arms
> therefore differ in schedule and selected epoch as well as in the loss. The
> effect survives a fixed-epoch check at eleven of twelve epochs, so it is not a
> selection artefact — but this is an intervention on the training procedure,
> not an isolation of the ranking gradient.

Four arms, ten seeds each, AsEP `epitope_group` validation, `--selection-metric
total`. The arms differ only in two loss weights; everything else — data, split,
seeds, epochs, capacity, representations — is identical.

| arm | `edge` | `partner_rank` | pair AUPRC | node MCC | node AUPRC |
|---|---|---|---:|---:|---:|
| node only | 0 | 0 | 0.00724 | **0.2798** | **0.4299** |
| **node + edge** | 1.0 | 0 | **0.09055** | 0.2773 | 0.4244 |
| node + rank | 0 | 0.5 | 0.00670 | 0.2050 | 0.3558 |
| node + edge + rank (the trained objective) | 1.0 | 0.5 | 0.06917 | 0.2404 | 0.3884 |

## Paired against the full objective, ten seeds

| arm | pair AUPRC | node MCC | node AUPRC |
|---|---|---|---|
| node only | −0.0619, 0/10 — **worse** | +0.0395, 10/10 — **better** | +0.0416, 9/10 — better |
| **node + edge** | **+0.0214, 10/10 — better** | **+0.0369, 10/10 — better** | +0.0361, 10/10 — better |
| node + rank | −0.0625, 0/10 — worse | −0.0354, 0/10 — worse | −0.0326, 0/10 — worse |

**Deleting the ranking term improves both endpoints, on every seed.** Not a
trade-off, not a wash: +0.0214 pair AUPRC and +0.0369 node MCC, unanimous.

Both *measured* endpoints. The ranking objective's own retrieval metrics are not
in this table, so this is not a Pareto improvement across all tasks and is not
described as one.

## Contact supervision buys pair prediction with no detectable node cost

| node + edge vs node only | | |
|---|---|---|
| pair AUPRC | +0.0833, 10/10 | decidedly better |
| node MCC | −0.0026, 3/10 | **undecided** |
| node AUPRC | −0.0055, 5/10 | **undecided** |

Contact-edge supervision raises pair AUPRC from 0.0072 to 0.0905 — an order of
magnitude — **with no detectable cost to node prediction**. Undecided is not
equivalence: no non-inferiority margin was registered in advance, and one may
not be read off these intervals now. The claim is the absence of a detected
loss, not a demonstrated absence of loss. And `node + rank` is
worse than `node only` on the node metrics by 0.075 MCC, so the ranking objective
damages node learning even with no contact supervision present to interact with.

## What this establishes

> In this training and selection procedure, disabling the **ranking auxiliary
> term** improves both prediction endpoints, while removing contact supervision
> does not. The two relational objectives are not interchangeable.

Contact supervision and node prediction are compatible: one is nearly free given
the other, and together they beat either alone on its own endpoint. The
complex-level ranking term is what costs both.

## What it does not establish, and this matters

**First-order shared-trunk gradient geometry does not explain it.** And the
converse inference — that a small gradient rules the term out — was made earlier
in this project and is withdrawn: updates pass through AdamW's accumulated
moments, so a short-window trunk gradient norm bounds neither the term's long-run
effect on the parameters nor its effect through paths outside the trunk.

`results/objective_component_localisation_v1/` measures the ranking gradient on
the shared trunk at **23-40× smaller in norm** than the node gradient, with
cosines against both other tasks within noise of zero. A term that small and that
orthogonal cannot, at first order on the trunk, be worth 0.037 node MCC.

So the effect is real and its mechanism is somewhere the trunk geometry does not
look. The candidates are the ones that were named before this ran: parameters
outside the shared trunk — the ranking term reaches a complex-scoring path the
cosine never measured — the training trajectory, an interaction with checkpoint
selection, and higher-order effects. Nothing here distinguishes them.

The honest statement is that the ablation locates the term and the geometry does
not account for it, and both halves are the result.

## What this revises

**The dose-response reading.** `results/joint_task_competition_v1/` found an
interior optimum near λ = 0.25 and it was written up as loss-scale imbalance
across the relational branch. The branch is not homogeneous. What benefits from
down-weighting is the ranking part; down-weighting the contact part to zero costs
an order of magnitude in pair AUPRC.

**The supervision-parity reading.** Node-only was compared to the full objective
and found close, which supported the benchmark comparison being fair. At ten
seeds here node-only is decidedly **better** than the full objective on node
metrics (+0.0395 MCC). The parity result was measured under a different
configuration and cohort split and is not contradicted, but "the relational
branch does not feed the node heads" understates what the full objective costs.

## Scope

Validation only. The `epitope_group` test split is permanently frozen and was not
read. One cohort, one architecture, one setting of the ranking weight (0.5); the
arms establish that removing the term helps, not what the best weight is.
