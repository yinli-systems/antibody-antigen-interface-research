# Pair-level architecture decision v1

## Verified diagnosis

The old MIPE replay result (`macro AP = 0.006231`, `Recall@1 = 0`) is a valid
diagnosis of the archived attention-only score, but it is not the architecture
of the active PINDER contact-pretraining line. The current 3B backbone already
jointly encodes concatenated protein pairs with hybrid intra-chain RoPE and
inter-chain content attention in every transformer layer. Its B objective also
already includes explicit residue-pair labels, marginal auxiliary labels,
legal partner swaps, confidence-weighted hard negatives, endpoint-degree
balancing, and physical-system multiplicity weights.

The remaining minimal weakness is therefore narrower: the active decoder is an
independent symmetric low-rank score for each pair. It has no explicit pair
state and cannot propagate evidence among edges that share an endpoint.

## What the primary literature supports

- [AsEP/WALLE](https://proceedings.neurips.cc/paper_files/paper/2024/file/15add6732964d5b1f0954058bf3ccc88-Paper-Datasets_and_Benchmarks_Track.pdf)
  formalizes all antibody-CDR/antigen-surface residue pairs as a complete
  bipartite edge-classification problem. WALLE itself uses a simple inner
  product or concatenation-plus-linear decoder and weighted reconstruction
  loss. The useful transfer is the explicit edge target and graph context, not
  a claim that its decoder is intrinsically stronger than the existing head.
  Its main table is an epitope-node benchmark; its pair-link table is a separate
  baseline and is not directly comparable to our full-axis AP.
- [Pair-EGRET](https://pmc.ncbi.nlm.nih.gov/articles/PMC11495673/) applies an
  explicit FFN classifier to every cross-attended residue pair and a separate
  node classifier to interface residues. This supports keeping pair and
  marginal supervision explicit. It does not by itself justify replacing the
  current hard-negative scheme, because the paper's main architectural
  contribution is the pair classifier after Siamese graph encoders and
  cross-attention.
- [PPLM](https://www.nature.com/articles/s41467-026-70457-5) jointly models
  concatenated sequence pairs with hybrid intra/inter attention and trains on
  more than 3.3 million pairs. That direction is already represented by the
  PairAwareESM2 Stage-A0 backbone. PPLM-Contact is not a pure sequence decoder:
  its strongest contact pipeline combines PPLM attention, MSA features,
  monomer distance maps, and 12 inter-protein transformer blocks. It therefore
  motivates explicit pair-state refinement but cannot be cited as a
  like-for-like sequence-only contact baseline.
- EpiGraph is most relevant to external biological validation and wet-lab
  evidence, not to the first decoder ablation. RLEAAI's handcrafted sequence
  features are an inexpensive later control, but mixing CKSAAP with a new
  decoder would prevent attribution.

## Implemented candidate

`BipartiteEdgeRefinementContactHead` creates one compact state for every valid
cross-chain residue pair from symmetric endpoint features:

`[h_i + h_j, |h_i - h_j|, h_i * h_j]`.

Two lightweight refinement steps then:

1. score all edge states;
2. pool messages across edges incident to each left and right residue;
3. combine the two endpoint contexts with symmetric sum, absolute-difference,
   and product features;
4. update each edge state with a gated residual MLP;
5. add the refined score to the existing symmetric low-rank logit.

The operation is linear in the number of candidate edges and edge dimension.
It does not use cross-chain coordinates, distance maps, labels, or structure as
model inputs. Shared parameters make partner-order reversal exactly equivalent
to transposing the contact map.

## Frozen matched canary

The first decision is a two-arm, PINDER-Train-only canary:

| Item | Frozen value |
|---|---|
| Backbone | Stage-A0 `segment-000130`, content SHA `be22eb18...f10f839` |
| Data | same authenticated 512-system B-canary manifest |
| Holdout | same 16 PDB-disjoint train-only systems |
| Arms | symmetric low-rank vs two-step 32-d edge refinement |
| Budget | 64 updates, accumulation 2, four RTX 4090 GPUs per arm |
| Unchanged | backbone layers, MLM, pair/marginal/swap losses, sampling, seed, optimizer |
| Primary | mean-per-system contact AP difference-in-differences |
| Secondary | AUROC, MRR, Recall@1, Recall@L/10, Precision@L/10, marginals, swap separation |
| Forbidden | test/temporal access, intermediate tuning, changing loss weights |

The edge head advances only if it improves contact AP learning gain, remains at
least as good on MRR and Recall@1, stays above prevalence, and passes finite,
memory, checkpoint, source, and protocol audits. A failed canary is archived;
it does not authorize iterative tuning on the holdout.

## Frozen decoder decision

Both canary arms completed successfully and passed their execution and
provenance checks. The edge-refinement endpoint had slightly higher raw contact
AP than the low-rank control (`0.002521` versus `0.002370`), but the
pre-registered paired difference-in-differences contact-AP gain was
`-0.0000424` with a 95% bootstrap interval of
`[-0.0002954, 0.0001756]`. Contact MRR and AUROC learning gains were also lower,
and Recall@1 remained zero in both arms. Although partner-swap separation
improved by `+0.008826 [0.000865, 0.016533]`, this did not improve pair ranking.

The frozen decision is therefore
`FAIL_BIPARTITE_EDGE_REFINEMENT_RANKING_GATE`: retain the low-rank decoder,
archive this edge-refinement candidate, and do not tune it against the canary.
This result also corrects an overly broad reading of AsEP: explicit bipartite
formulation is useful, but adding an edge-state module is not sufficient by
itself to improve the active jointly encoded model.

The separate loss canary also completed. With the decoder fixed to the current
low-rank head, class-balanced focal loss produced lower paired learning gains
than the adaptive-balanced control: contact AP
`-0.0001504 [-0.0003098, -0.0000154]`, AUROC
`-0.008748 [-0.015029, -0.001861]`, and MRR
`-0.000863 [-0.001740, -0.000121]`. Recall@1 remained zero and partner-swap
separation decreased. The frozen decision is
`FAIL_BALANCED_FOCAL_PAIR_RANKING_GATE`; retain the adaptive-balanced pair loss.
This is a controlled rejection of the focal-loss substitution, not evidence
against focal loss under PPLM's different data, contact definition, features,
or model architecture.

## Sequenced follow-up

1. Do not promote the failed edge-refinement decoder to B32K or alter the active
   low-rank B32K job.
2. Retain the adaptive-balanced objective; the matched focal-loss substitution
   failed its ranking gate and must not be promoted to B32K.
3. Evaluate the selected model on the official AsEP pair-link task with its
   exact graph membership and split definitions. Report it as an external
   benchmark, not by mixing AsEP node MCC with our pair AP.
4. Add CKSAAP only as a frozen low-cost sequence-feature control if the decoder
   result leaves a clear residual gap. Do not make it part of the main model
   without a matched ablation.
5. Preserve SABLE and historical SAbDab2 as analyst-exposed evidence. Final
   claims still require a genuinely sealed post-cutoff antibody-and-antigen
   double-cold cohort.
