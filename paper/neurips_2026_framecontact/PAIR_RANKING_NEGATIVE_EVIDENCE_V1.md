# Pair-ranking negative evidence and benchmark contribution outline

## What can be reported now

The active jointly encoded pair model was subjected to two matched,
PINDER-Train-only canaries. The first replaced the symmetric low-rank decoder
with two steps of explicit complete-bipartite edge refinement. The second kept
the decoder fixed and replaced adaptive degree-balanced negative weighting with
class-balanced focal loss. Backbone, systems, holdout, optimization budget,
seeds, and test firewall were held fixed.

The edge-refinement candidate failed its preregistered ranking gate. Its paired
difference-in-differences contact average-precision gain was
`-0.0000424`, with bootstrap 95% interval
`[-0.0002954, 0.0001756]`; Recall@1 remained zero. The focal-loss candidate
also failed: contact AP `-0.0001504 [-0.0003098, -0.0000154]`, AUROC
`-0.008748 [-0.015029, -0.001861]`, and MRR
`-0.000863 [-0.001740, -0.000121]` relative to the adaptive control.

These experiments reject two substitutions under this frozen canary. They do
not show that graph link prediction or focal loss is generally ineffective,
and they do not authorize tuning on the same holdout.

## Proposed paper contribution

The publishable unit should combine:

1. an authenticated physical-system antibody/PPI resource;
2. four-view VH/VL/concatenated-antibody/antigen cold-split construction;
3. exact crossing and infeasibility receipts;
4. a pair-contact curriculum with partner swaps and multiplicity controls;
5. preregistered positive and negative model decisions;
6. a sealed or post-cutoff external evaluation.

The failed canaries belong in a failure-analysis section or appendix. The main
claim requires a passing B32K/larger-scale result and zero-crossing evaluation.

## Required figures and tables

- data-flow and label-firewall diagram;
- split component/crossing graph with discarded bridges;
- pair AP, MRR, Recall@1, and matched-minus-swapped curves;
- canary decision table including rejected variants;
- per-PDB or per-cluster uncertainty, not row-weighted confidence intervals;
- claim ledger separating train-only, analyst-exposed, and sealed evidence.

## Forbidden wording

- "SOTA" without same-split, same-label, same-metric comparison;
- "blind" for SABLE, MIPE, or previously opened SAbDab2 labels;
- "generative antibody design" for contact-guided sequence reranking alone;
- "RFantibody comparable" without backbone/pose generation and experimental
  validation;
- "40%-cold" unless every required sequence view passes the independent audit.
