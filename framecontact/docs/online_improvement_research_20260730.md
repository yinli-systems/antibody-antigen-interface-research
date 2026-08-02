# Online research update: what to improve next

Updated: 2026-07-30

## Bottom line

The next high-information experiment is not raw `xyz` RoPE. It is a controlled
replacement of the current hard spatial mask with a **dense sequence path plus a
zero-gated sparse local-frame path**, followed by a loss ablation and an
antigen-surface branch. A genuinely rotary local-pose score should be retained as
an equal-parameter research control, not promoted before it beats an additive
geometry bias on strict validation and an external benchmark.

This ordering follows both the project's real CHIMERA failure mode and recent
primary work. It separates three questions that the existing hard-mask ablation
confounds:

1. does a spatial neighbour graph help;
2. do distance, direction, and orientation add information on the same graph;
3. can geometry help without deleting global ESM-mediated communication?

## 2025-2026 method triage

| Method/family | What is useful here | Direct or adapted | Decision |
|---|---|---|---|
| AlphaFold2/IgFold IPA | learned local-frame points alter attention while preserving rigid-transform invariance | adapted | strong intra-chain baseline after additive bias |
| ProteinMPNN/LigandMPNN edge geometry | RBFs of multiple backbone-atom distances on a sparse graph | direct | add nine `N/CA/C` distances first; later include `O` and virtual `CB` |
| SparseIPA | union of sequence, spatial, and sampled/global neighbours | adapted | use the graph recipe; do not copy benchmark-tuned neighbour counts |
| FlashIPA | factorized IPA with FlashAttention-compatible linear-memory execution | adapted | engineering baseline after correctness; arbitrary pair MLP bias is not automatically factorable |
| EpiFormer | two independent encoders, interleaved gated bidirectional content cross-attention, sparsity-aware loss | adapted in phase two | use content-only cross-attention for unaligned monomers; ablate the cardinality term separately |
| SurfBind | surface patches, normals/curvature/chemistry, geodesic self-attention, coarse-to-fine prediction | adapted | motivates an antigen-only exposure gate, then optional surface tokens |
| ProTact | invariant monomer encoders plus trigonometric refinement of a cross-chain pair map | research phase two | promising for explicit contact logits; larger change than self-attention PE |
| AsEP/WALLE | epitope-group split, sequence-plus-graph fusion, bipartite link prediction | direct evaluation target | highest-priority prospective external benchmark after decontamination |
| PEPNet | residue-aware atom point clouds plus antibody-antigen cross-attention | higher-cost baseline | useful all-atom/surface upper bound, not the first backbone-only change |
| EPT | pretrained all-atom E(3)-equivariant scalar/vector Transformer | high-cost upper bound | evaluate only if the compact local-frame module saturates |
| STRING, LieRE, MASA/ComRoPE | learned multidimensional orthogonal rotary maps | research control | raw global coordinates are not invariant to rotating a monomer; require local-frame adaptation |
| 3D-GS-RoPE / PointTransformerX | efficient learned orthonormal 3D basis in point-cloud attention | research control | attractive implementation idea, but its published task does not establish protein SE(3) invariance |
| GeoPE | quaternion/Lie-algebra coupling of spatial axes | reject as direct drop-in | vision/structured-tensor evidence does not solve independent monomer-frame invariance |
| GATr / full E(3) equivariant attention | exact geometric algebra or scalar/vector state throughout layers | high-cost upper bound | scientifically clean but too invasive for the first ESM-fusion ablation |

Primary sources: [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2),
[IgFold](https://www.nature.com/articles/s41467-023-38063-x),
[ProteinMPNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC9997061/),
[SparseIPA](https://www.nature.com/articles/s42256-025-01100-z),
[FlashIPA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/07ea874e9e4f71ec6680a3574a485a36-Abstract-Conference.html),
[EpiFormer](https://arxiv.org/abs/2606.04154),
[SurfBind](https://arxiv.org/abs/2606.23830),
[ProTact](https://www.nature.com/articles/s42003-026-09809-2),
[AsEP](https://papers.nips.cc/paper_files/paper/2024/hash/15add6732964d5b1f0954058bf3ccc88-Abstract-Datasets_and_Benchmarks_Track.html),
[PEPNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC12919445/),
[EPT](https://www.nature.com/articles/s41467-026-69185-7),
[STRING](https://proceedings.mlr.press/v267/schenck25a.html),
[LieRE](https://proceedings.mlr.press/v267/ostmeier25a.html), and
[PointTransformerX](https://openaccess.thecvf.com/content/CVPR2026W/ECV/papers/Reichardt_PointTransformerX_Portable_and_Efficient_3D_Point_Cloud_Processing_without_Sparse_CVPRW_2026_paper.pdf).

## Recommended self-attention block

For each antibody or antigen encoder independently:

```text
x_dense = DenseSelfAttention(x, sequence_rope)

g_ij = [
  RBF(||a_j^q - a_i^p||) for p,q in {N,CA,C},
  R_i^T(CA_j - CA_i) / ||CA_j - CA_i||,
  vec(R_i^T R_j),
  sequence_separation,
  frame_validity
]

x_geom = PackedSparseAttention(x, union_graph, pair_bias=g_ij)
x_next = x + x_dense + tanh(alpha) * x_geom
```

`alpha` is initialized to zero, so the starting function is exactly the dense
sequence baseline. Antibody and antigen blocks keep separate parameters. The
neighbour graph is the deduplicated union of same-chain sequence neighbours,
CA-spatial neighbours, and a small deterministic landmark/global set.

Required same-graph comparisons:

| Variant | Graph | Pair feature | Scientific question |
|---|---|---|---|
| dense | all same-chain | none | sequence reference |
| dense + topology residual | union graph | none | adjacency alone |
| dense + distance residual | same union graph | CA-distance RBF | radial value |
| dense + nine-distance residual | same union graph | all `N/CA/C` distances | richer backbone geometry |
| dense + direction residual | same union graph | distances + local direction | ordered orientation |
| dense + full pose residual | same union graph | distance + direction + relative rotation | full frame value |
| dense + frozen-frame IPA | same union graph | learned local points | mature protein baseline |
| dense + local-pose rotary | same union graph | pair-conditioned orthogonal score | proposed research control |

## Innovation candidate: Local-Pose Orthogonal Attention

For each head, learn a small set of skew-symmetric generators `G_hm`. An
invariant pair network maps local pose `g_ij` to bounded phases `phi_ijhm`:

```text
U_ijh = exp(sum_m phi_ijhm * G_hm)
s_ijh = q_ih^T U_ijh k_jh / sqrt(d_h)
logit_ijh = content_ijh + tanh(beta_h) * s_ijh + bias_h(g_ij)
```

Because `g_ij` is built from `R_i^T(CA_j-CA_i)` and `R_i^T R_j`, this score is
invariant to one shared proper rigid transform of a monomer. It is intentionally
called **pair-conditioned orthogonal attention**, not standard RoPE: it cannot in
general be factorized into one position rotation per token. It starts with
`beta_h=0` and runs only on the sparse graph.

Acceptance requires all of the following:

- FP32 rigid-transform invariance and missing-frame tests;
- an equal-parameter additive-bias control;
- three-seed improvement on the locked validation metric;
- stability to 0.1/0.25/0.5-A coordinate noise;
- non-zero counterfactual attention change attributable to the rotary term;
- A100 time and memory no worse than the locked budget;
- no architecture selection from frozen-test behavior.

## Objective and output changes

The first loss matrix should keep the architecture fixed:

1. current residue-pooled weighted BCE;
2. per-complex normalized weighted BCE;
3. per-complex BCE plus a separately weighted predicted-positive-count penalty;
4. the previous loss plus Dice;
5. an AP surrogate only as a higher-risk control.

EpiFormer's ablation motivates testing a sparsity/cardinality term, but does not
justify transferring its coefficient. All weights and thresholds must be chosen
on strict validation. For partner-specific interaction, train an explicit
antibody-by-antigen pair-contact head and derive residue marginals from that map;
attention weights are diagnostics, not contact predictions.

## Prospective external evaluation

The evidence upgrade should use the public AsEP epitope-group benchmark as a new
prospectively locked evaluation target, with an additional DB5.5/unbound or
predicted-structure stress set if inputs can be reproduced. Before reading model
scores:

1. decontaminate against SAbDab2 train/validation by PDB identity, antibody
   sequence cluster, antigen sequence cluster, and epitope group;
2. freeze adapter, checkpoints, ensemble rule, threshold rule, seeds, exclusions,
   metrics, and bootstrap unit;
3. hash the protocol, split membership, structures, ESM embeddings, checkpoints,
   and container/runtime;
4. report coverage and exclusions before inference;
5. evaluate exactly once, retaining every complex-level prediction.

Report macro-per-complex and pooled AUPRC, AUROC, MCC, F1, precision, recall,
top-k precision/recall, Brier/ECE, and cluster-bootstrap confidence intervals.
For a pair-contact head, also report pair AUPRC/AUROC, precision at `L/10`,
`L/5`, and `L`, and partner-shuffle specificity. Stratify by antigen length,
positive prevalence, experimental versus predicted structure, and frame
coverage.

## Execution order

1. Finish the expanded CHIMERA adapter rerun as secondary robustness evidence.
2. Implement dense plus zero-gated topology residual and verify exact dense
   initialization.
3. Add nine-distance, direction, and full-pose branches on the identical graph.
4. Align loss with per-complex evaluation; isolate the count penalty and Dice.
5. Add low-risk antigen exposure features before surface tokens.
6. Implement a real packed `[B,H,L,K]` reference; benchmark A100 memory/runtime.
7. Freeze the winner and run the decontaminated AsEP external protocol.
8. Only then start content-only inter-chain cross-attention and pair-map triangle
   refinement inspired by EpiFormer and ProTact.

## Evidence boundary

Recent EpiFormer and SurfBind numbers are preprint evidence. Results from AsEP,
PEPNet, FlashIPA, or point-cloud RoPE do not transfer to this codebase without a
reproduction under the same split and metrics. The completed CHIMERA adapter-v3
coverage audit is post-access and can support a secondary rerun only. No runtime,
memory, external-generalization, or benchmark improvement claim is permitted
until its corresponding receipt exists.
