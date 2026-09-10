# Part one: the three-way gradient geometry

Registered in `protocols/OBJECTIVE_COMPONENT_LOCALISATION_V1.md`. Read once, on
two epochs of AsEP `epitope_group` training with `d_model` 128 and ESM-IF1
representations. Gradients on the shared trunk, on real training batches with
the negative samples the ranking term requires.

| | epoch 1 | epoch 2 |
|---|---|---|
| cos(g_node, g_edge) | −0.0590 | −0.0506 |
| cos(g_node, g_rank) | −0.0053 | −0.0032 |
| cos(g_edge, g_rank) | −0.0101 | +0.0017 |
| ‖g_node‖ | 4.938 | 2.197 |
| ‖g_edge‖ | 8.265 | 5.315 |
| ‖g_rank‖ | **0.218** | **0.163** |

Norms are of the weighted gradients; both λ are in the run config and the
gradient is linear in them, so unweighted norms follow by division.

## The hypothesis is refuted

The hypothesis was that the optimisation pathology localises to the ranking
auxiliary objective. It does not.

The ranking gradient on the shared trunk is **23-40× smaller than the node
gradient** and 25-33× smaller than the contact gradient, and its cosine against
both is within noise of zero. A term that small cannot be the source of a
conflict; it barely moves the trunk at all.

Every bit of the negative cosine is **node against contact edges** — relational
supervision itself, not the auxiliary. Under the registered outcomes this is the
"both negative" branch, which says the loss-scale account across the relational
branch stands and the ranking term is not special. That is the honest reading,
with the caveat that calling cos(g_node, g_rank) = −0.005 "negative" reads more
into a number ten times smaller than its neighbour than it can carry.

## What it does explain

The checkpoint-selection failure now has a mechanism that fits rather than
competing with this. `partner_rank` is a poor early-stopping signal **because**
the model barely optimises it on the shared trunk: selecting on a quantity the
training dynamics hardly touch picks an epoch close to arbitrary. That is a
different claim from "ranking gradients conflict", and it is the one the data
supports.

## What it does not explain, and what is running

`results/gradient_conflict_v1/` reports cos(node, edge) = **+0.70**, negative on
none of 240 complexes. That is the same decomposition as the −0.059 here, once
the ranking term is isolated. So the decomposition mismatch is real but is **not**
what flips the sign, and the earlier attribution of the discrepancy to
`partner_rank` was wrong.

Two candidates remain, and they are separable:

- **Stage.** The +0.70 was taken at a converged checkpoint; this is epochs 1-2
  from initialisation. The trend across the two epochs is toward zero.
- **Cohort.** The +0.70 was taken on sabdab2; this is AsEP.

A twelve-epoch `measure` run on each cohort is in flight. If the cosine climbs
toward +0.70 as training converges on both, the disagreement is stage and the
conflict is a transient of early optimisation. If AsEP stays negative while
sabdab2 climbs, it is the cohort. Either way one of the two numbers currently in
the repository needs its scope narrowed in print, and which one is decided by
that run.

## Part two

The causal ablation — node only, node+edge, node+rank, node+edge+rank, ten seeds,
validation only — is running. Its prediction is sharpened by the geometry above
and recorded here before its numbers exist: if the ranking gradient is negligible
on the trunk, **node+rank should behave close to node only**, and **node+edge
close to the full objective**. A result where node+rank differs materially from
node only would contradict the geometry and is the outcome most worth watching.
