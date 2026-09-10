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

### It may be slightly worse than nothing

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
