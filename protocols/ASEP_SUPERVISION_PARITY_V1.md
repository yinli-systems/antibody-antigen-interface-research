# AsEP supervision parity

Registered 2026-09-10, **before the arm was submitted**.

## The problem this addresses

The test measurement in `results/asep_benchmark_v1/BENCHMARK.md` reports MCC
0.189 against a best published figure of 0.171 on the same split, at the same
convention. The record already names the reason that comparison is not yet a
claim about method quality:

> LF3DRoPE trains on antigen residue labels only. "All model components are
> optimized jointly through this residue-level objective." (arXiv 2608.01092)

This model trains on residue **pairs** — which antibody residue contacts which
antigen residue — and derives node marginals from them. Its default loss weights
are `edge 1.0`, `partner_rank 0.5`, and `0.25` on each of the four node terms.
The edge and partner-rank terms use supervision a node-label method does not
have. The comparison therefore cannot separate a better method from a richer
training signal, and the richer signal is the more likely explanation of the two.

## The arm

`node_supervision_only`: identical in every other respect, with

    --edge-loss-weight 0 --partner-rank-weight 0

leaving only `epitope_bce`, `epitope_dice`, `paratope_bce` and `paratope_dice`.
Ten seeds, the same ten, the same twelve-epoch budget, the same frozen
representations, the same split.

`partner_rank` is switched off as well as the edge loss because it is trained
against sampled wrong partners, which is pairing information a method given one
antibody and one antigen does not receive.

## What each outcome means, fixed now

**If the node-only arm still exceeds 0.171** on validation-selected,
test-measured micro MCC: the comparison is on parity of supervision and the
published-table comparison stands as a statement about the method.

**If the node-only arm falls to or below the published figures**: the reported
advantage comes from the richer supervision. The benchmark record is then
corrected to say so, the headline becomes the node-only number, and the
pair-supervised figure is reported separately and explicitly labelled as using
supervision the baselines do not.

**If it lands between**: both numbers are reported, with the node-only one
leading, because it is the one measured at parity.

## Test access

This arm is evaluated on validation to select its threshold and on test to
report, exactly as the frozen arm was. That is a second reading of the test
split, and it is the reason this protocol exists rather than the measurement
simply being taken: the first reading answered "how does the frozen
configuration score", and this one answers a different question — "is that score
attributable to the method or to the supervision" — which could not be answered
without it and cannot be answered on validation alone, since the published
numbers are on test.

No third reading follows from this one. If the node-only arm raises further
questions, they are answered on validation.
