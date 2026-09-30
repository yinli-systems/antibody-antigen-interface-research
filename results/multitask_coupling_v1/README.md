# The pair branch does not feed the node heads

Status: a paired ablation on one cohort, ten seeds, measured on AsEP's
`epitope_group` test split. It was run to answer a different question — whether
this project's benchmark comparison was fair — and this fell out of it.

Date: 2026-09-10

Machine-readable: [`report.json`](report.json).

## The question it answers

This architecture predicts antibody-antigen residue **pairs** and derives node
marginals from them, trained on a joint objective: `edge 1.0`,
`partner_rank 0.5`, and `0.25` on each of four node terms. The pair terms carry
most of the weight.

The ablation switches off `edge` and `partner_rank`, leaving node labels alone,
and changes nothing else. Same seeds, same split, same frozen representations.

## Result

Paired within seed, ten seeds, AsEP `epitope_group` test, at the fixed 0.5
threshold the evaluation receipts use:

| | pair-supervised | node labels only | difference | seeds |
|---|---:|---:|---:|---:|
| **pair macro AUPRC** | 0.038797 | 0.005838 | **−85.0%** | 0/10 |
| **pair macro MCC** | 0.082316 | −0.000001 | **−100.0%** | 0/10 |
| pair precision at true count | 0.048519 | 0.003152 | −93.5% | 0/10 |
| epitope node AUPRC | 0.237767 | 0.245205 | **+3.1%** | **10/10** |
| epitope node MCC | 0.138968 | 0.145497 | +4.7% | 7/10 |
| epitope node AUROC | 0.741 | 0.742 | +0.1% | 5/10 |
| paratope node MCC | 0.521465 | 0.523513 | +0.4% | 8/10 |

**The pair task collapses to chance and the node tasks do not move.**

Pair MCC lands at −0.000001. That is not a degradation, it is the pair head
predicting nothing at all once its supervision is withdrawn — which is the
control working: the pair supervision is doing exactly what it is supposed to,
for the pair task.

## What follows

**Pair supervision is necessary for the pair task.** Removing it costs
everything: 85% of pair AUPRC, all of pair MCC, 94% of precision at the true
contact count.

**Pair supervision contributes nothing to the node tasks.** Every node figure is
unchanged or marginally better without it.

**So the multi-task coupling does not transfer pair signal to node prediction.**
The pair branch learns a real, supervised, non-trivial function — on the sabdab2
cohort its head beats the predicted-marginal outer product by 14.7% of baseline
at 20/20 seeds — and none of what it learns reaches the node heads.

### Correction: this is a badly set loss weight, not a competition

The paragraph below was written when only the two endpoints existed. A dose
response over λ ∈ {1.0, 0.5, 0.25, 0}, scaling both pair terms together at ten
paired seeds on the same split, shows the competition framing is wrong.

| λ | node AUPRC | node MCC | pair AUPRC |
|---:|---:|---:|---:|
| 1.00 | 0.376727 | 0.220828 | 0.066119 |
| 0.50 | 0.405256 | 0.244554 | 0.070619 |
| **0.25** | **0.426923** | **0.266839** | **0.073286** |
| 0.00 | 0.423680 | 0.253765 | 0.006315 |

Paired against λ=1.0: at λ=0.5 node AUPRC gains 0.0285 (decided, 9/10) **and
pair AUPRC gains 0.0045 (decided, 7/10)**. At λ=0.25 node gains 0.0502
(decided) and pair gains 0.0072 (8/10, undecided).

**Both tasks improve together as the pair weight falls from 1.0 to 0.25.** There
is no trade to observe over that range. The competition appears only at λ=0,
where pair AUPRC collapses by 0.0598 at 0/10 seeds while node performance is no
longer improving.

`protocols/PAIR_SUPERVISION_DOSE_RESPONSE_V1.md` registered a readability
condition before the runs: *if the pair metrics do not fall with λ, the
multiplier is not doing what the protocol assumes and nothing else can be read.*
Pair metrics **rise** between λ=1.0 and λ=0.25, so that condition fires and the
capacity reading is not available in that range. The condition was written to
block exactly this, and it does.

What the data supports instead is the plainer account, and it is one of the
alternatives the protocol listed: **loss-scale imbalance**. The edge term at
weight 1.0 dominates an objective whose four node terms carry 0.25 each, and
simply reducing it improves the optimisation of everything. The joint optimum
for both tasks sits near λ=0.25, not at 0.

The usable statement is therefore:

> The default pair-loss weight is too high for both tasks. Reducing it to a
> quarter improves node AUPRC by 13.3% and pair AUPRC by 10.8% at once;
> removing it entirely collapses the pair task without buying anything further
> for the nodes.

That is an actionable hyperparameter result rather than a claim about
architectural coupling, and it is easier to check.

### The original endpoint observation, and why its framing was wrong

Epitope node AUPRC is 3.1% higher without the pair losses, at **10/10 seeds**.
Unanimity across ten seeds is hard to read as noise. The most likely account is
capacity: the pair terms carry 1.5 of the 2.5 total loss weight, and the shared
trunk spends parameters on them that the node heads would otherwise use.

That makes this mild **negative** transfer, not merely absent transfer.

## Limits

1. **One cohort, one architecture.** This shows that *this* coupling does not
   transfer. It does not show that no coupling could, and it does not say the
   pair task is not worth solving — on sabdab2 it is the task the whole
   workstream exists for.
2. AsEP's antibody side is CDR residues only. A coupling might transfer
   differently where the antibody is a full variable region.
3. Metrics here are at a fixed 0.5 threshold, as written in the receipts. The
   benchmark record uses a validation-selected threshold and its node MCC values
   differ accordingly (0.157 and 0.159); the two conventions must not be mixed.
4. The node-only arm was run to test supervision parity for the benchmark
   comparison. This finding is a by-product, and no protocol was registered for
   it in advance.

## Provenance

| | |
|---|---|
| arms | `esm2_esmif1`, `node_supervision_only` |
| difference | `--edge-loss-weight 0 --partner-rank-weight 0` |
| protocol | `protocols/ASEP_SUPERVISION_PARITY_V1.md` (registered for the parity question) |
| split | AsEP `epitope_group`, 169 test complexes |
