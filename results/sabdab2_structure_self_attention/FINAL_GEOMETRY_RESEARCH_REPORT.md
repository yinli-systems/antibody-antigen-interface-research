# Geometry-aware self-attention: final research and evidence report

## Executive decision

For the advisor's current intra-chain self-attention stage, the recommended
encoding is **distance RBF plus receiver-frame direction as an additive
per-head attention bias**. It is the smallest representation that:

1. is invariant to independent rigid transforms of antibody and antigen;
2. lets each residue weight same-chain spatial neighbors;
3. preserves frozen ESM residue embeddings;
4. has an equal-parameter topology-only control; and
5. shows a positive development signal across three available seeds.

The full relative rotation matrix is not justified by the present evidence. It
does not outperform the simpler direction encoding. A method named
`SE(3)-RoPE` would also be technically misleading for the implemented module:
ordinary sequence RoPE is retained for one-dimensional order, while structure
enters as invariant pair features and an additive attention-logit bias. The
module produces invariant scalar embeddings, not equivariant vector features.

This completes the advisor-requested **self-attention prototype, fusion
interface, ablation, and AB03/A100 reproduction**. It does not complete the
separate partner-specific interaction problem. That requires inter-chain
fusion, a native residue-pair contact head, matched-partner controls, and a
two-axis leakage-controlled evaluation.

## Advisor requirements and completion

| Requirement | Delivered evidence | Status |
|---|---|---|
| Survey Transformer geometry/SE(3) methods | Method comparison and primary-source references below | Complete |
| Survey protein relative geometry encodings | Distance, local direction, relative rotation, IPA, GVP, EGNN, SE(3) attention | Complete |
| Study sequence/structure fusion | Frozen ESM plus residual geometry bias; alternatives compared below | Complete |
| Implement three intra-chain encodings | Distance, distance+direction, full local relative pose | Complete |
| Separate antibody/antigen self-attention | Parameter-unshared encoders with independent rigid-transform tests | Complete |
| Real N/CA/C and masking interface | Chain, padding, valid-frame, and missing-frame contracts | Complete |
| Equal-parameter ablation | Dense sequence, topology, distance, direction, full pose | Complete |
| AB03/A100 deployment reproduction | Sealed two-A100 archive, 48/48 gates, exact manifests | Complete |
| Robust statistical audit | Per-complex component bootstrap/permutation and independent verifier | Complete for one final-checkpoint seed |
| Confirmatory multi-seed gate | Three historical seeds exist, but not the preregistered five-seed set | Incomplete |
| Frozen outer test | Deliberately unread | Not started |
| Partner-specific/native pair prediction | Outside the current self-attention stage | Not started |

## Literature synthesis

### What successful structure models actually do

The literature does not support a single universal "3D positional encoding."
Successful systems differ according to whether the output is an invariant
classification, an equivariant coordinate update, inverse folding, or complex
structure generation.

| Method | Geometric representation and injection | Strengths | Cost or mismatch for this stage | Decision |
|---|---|---|---|---|
| AlphaFold2 IPA | Learned query/key/value points in residue-local frames; squared point distances contribute to shared attention weights | Global-frame invariant; tightly couples residue and geometric reasoning | Designed around iterative residue frames and coordinate generation | Adopt the local-frame and attention-bias principle, not the full structure module |
| ProteinMPNN | Backbone-derived relative edge features with graph message passing | Efficient and directly suited to N/CA/C backbone context | Autoregressive inverse-folding objective differs from interface classification | Direct conceptual precedent for compact backbone edge features |
| GVP / ESM-IF1 | Separate scalar and vector channels, then combine invariant geometric processing with a Transformer | Explicit directional information and proven sequence/structure fusion pattern | Replaces more of the encoder and introduces vector-channel state | Strong second-stage ablation if scalar bias saturates |
| EGNN | E(n)-equivariant message passing with coordinate-dependent scalar messages | Simpler than higher-order tensor-field models | Coordinate updates are unnecessary for fixed monomer encodings; reflection equivariance may erase handedness information | Do not use as the first drop-in module |
| SE(3)-Transformer | Irreducible representations and spherical-harmonic attention kernels | Rigorous roto-translation equivariance | Higher implementation and compute cost; intrusive for frozen ESM fusion | Reserve for a separately budgeted architecture study |
| AlphaFold3 Pairformer | Single and residue-pair representations with repeated pair reasoning; a diffusion module operates on atom coordinates | Strong evidence that interaction tasks need an explicit pair state | Not a drop-in positional encoding; large architectural change | Use as precedent for the later inter-chain pair head |
| Current module | RBF distance, receiver-frame direction, optional relative rotation, and sequence separation projected to per-head bias | Minimal, invariant, auditable, equal-parameter ablations | Scalar invariant output; dense pair tensors; no partner interaction | Recommended for the advisor's phase-one self-attention |

Primary sources:

- [AlphaFold2 and invariant point attention](https://www.nature.com/articles/s41586-021-03819-2)
- [ProteinMPNN](https://doi.org/10.1126/science.add2187)
- [Geometric Vector Perceptrons](https://openreview.net/forum?id=1YLJDvSx6J4)
- [ESM and ESM-IF1 implementation](https://github.com/facebookresearch/esm)
- [E(n)-equivariant graph neural networks](https://proceedings.mlr.press/v139/satorras21a)
- [SE(3)-Transformer](https://proceedings.neurips.cc/paper_files/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html)
- [AlphaFold3 and Pairformer](https://www.nature.com/articles/s41586-024-07487-w)

### Why the selected representation is mathematically appropriate

For residue `i`, define a right-handed frame from `N`, `CA`, and `C`:

```text
origin_i = CA_i
x_i = normalize(C_i - CA_i)
y_i = normalize((N_i - CA_i) - projection_on_x_i)
z_i = normalize(x_i cross y_i)
R_i = [x_i, y_i, z_i]
```

For an ordered same-chain edge `(i, j)`, the implemented features are:

```text
d_ij = ||CA_j - CA_i||
u_ij = normalize(R_i^T (CA_j - CA_i))
Q_ij = R_i^T R_j
```

Distance is expanded with 16 Gaussian radial basis functions. Signed and
absolute sequence separation are also available in the fixed-width pair tensor.
For any common rotation `G` and translation `t`, both frames transform to
`G R`, so `d_ij`, `u_ij`, and `Q_ij` remain unchanged. Antibody and antigen are
encoded independently, so their monomer coordinate systems do not need to be
aligned.

The attention logits are:

```text
logit_ij^h =
    dot(q_i^h, k_j^h) / sqrt(d_head)
    + MLP_h(geometry_ij)
```

The last geometry projection is zero-initialized, making geometry a residual
signal rather than an arbitrary initial bias. Ordinary sequence RoPE remains on
queries and keys. A same-chain `k=32`, 16-A spatial graph selects neighbors.
Missing or degenerate frames retain a safe self edge and zero geometric
features.

## Data and experimental contract

- Authenticated development authority before SAbDab2 intersection:
  3,393 train and 52 validation instances.
- Native SAbDab2 training-role intersection:
  2,986 train and 43 validation candidates.
- Strict structure/canonical-axis acceptance:
  2,985 train and 43 validation.
- Fixed `antigen_length <= 384` runtime cohort:
  2,676 train and 36 validation complexes.
- Native-release test structures were excluded before mmCIF parsing.
- The 36 validation complexes contain 5,149 antibody residues and 9,572 antigen
  residues and form 16 exact endpoint-connected components.
- All spatial modes have 271,670 trainable parameters, the same optimizer,
  sample order, three-epoch budget, and fixed-final-epoch reporting.
- Frozen ESM2 residue embeddings are input features; ESM is not fine-tuned.
- Metrics from the small development validation set cannot be reported as an
  identical-benchmark or external-generalization result.

The dense sequence-only mode is essential. Comparing it with topology-only
separates the effect of the spatial neighbor graph from the additional edge
features. Comparing direction with topology-only then measures the incremental
value of geometry after controlling for spatial topology.

## AB03/A100 engineering reproduction

The recovered archive authenticates an exact A100 replay:

- hardware: 2 x NVIDIA A100-PCIE-40GB;
- PyTorch: 2.5.1 in the replay receipt;
- CUDA runtime: 12.4;
- parent image:
  `sha256:5d64c7063a9cb2d0be961e43191b6638bb93bdfba1cff7f02c847d06fa7f2274`;
- additive runtime image:
  `sha256:700902df319a2b2ebb8e455f0402cd435c5928582ea9170ba23c89d79116551c`;
- data authority: 8,727 case-sensitive files;
- data manifest:
  `0bd4079b9b57cd43b7dbb2c7c7525acfe02215e62ac58ded5ba703e80dadd441`;
- archive:
  `b6365093db7305b2d054c8a801e9aed002423156aae328075faf73466ab9c75b`;
- scientific replay: 48/48 gates passed;
- exact result:
  `7dfdeb1e9725500addf02f0f52e658def4c0d5adfec9df68e964c803bf6e321e`;
- native test read: false.

The independent archive verifier rejects path traversal, links, duplicate
members, malformed digests, recomputation disagreement, failed gates, and
manifest disagreement. It recomputed 80 multi-seed summary statistics and
verified the seal, original-run, evidence, and data-authority manifests.

## Results

### Fixed-final A100 pooled-residue result

| Encoding | Paratope AUPRC | Epitope AUPRC | Joint macro AUPRC |
|---|---:|---:|---:|
| topology only | 0.630066 | 0.253012 | 0.441539 |
| distance | 0.629135 | 0.258904 | 0.444020 |
| distance + direction | **0.654958** | **0.263567** | **0.459263** |
| full relative pose | 0.650543 | 0.258911 | 0.454727 |

All 16 AUPRC/AUROC replay comparisons were within the locked `0.01` tolerance
of the reference environment. These pooled metrics are descriptive because
larger complexes contribute more residues.

### Exact A100 checkpoints: per-complex paired audit

The exact A100 final checkpoints were replayed in FP32 on local CPU using the
same 36-complex membership. The independent implementation passed 441 checks.

| Encoding | Paratope AUPRC | Epitope AUPRC | Joint macro AUPRC |
|---|---:|---:|---:|
| topology only | 0.660414 | 0.305707 | 0.483061 |
| distance | 0.650777 | 0.309229 | 0.480003 |
| distance + direction | **0.671929** | **0.313944** | **0.492936** |
| full relative pose | 0.672236 | 0.312127 | 0.492182 |

For the preregistered primary comparison, distance+direction minus
topology-only:

| Endpoint | Mean difference | Component-bootstrap 95% CI | One-sided sign-flip p |
|---|---:|---:|---:|
| paratope AUPRC | +0.011514 | [+0.001956, +0.023375] | 0.024798 |
| epitope AUPRC | +0.008237 | [-0.006809, +0.020394] | 0.195180 |
| joint macro AUPRC | +0.009876 | [+0.001836, +0.016695] | 0.027497 |

The three-comparison Holm-adjusted one-sided p-value for the joint endpoint is
`0.082492`. The strict single-seed gate therefore fails, despite the unadjusted
positive joint interval. Epitope uncertainty still crosses zero.

### Three available fixed-budget A100 seeds

The sealed archive contains seeds `20260728`, `20260729`, and `20260730`. These
are historical stability evidence, not the five seeds preregistered later.

| Mode | Mean pooled joint AUPRC | Sample SD | Mean delta vs topology | Positive seeds |
|---|---:|---:|---:|---:|
| dense sequence only | 0.369418 | 0.004382 | -0.036007 | 0/3 |
| topology only | 0.405426 | 0.031575 | reference | - |
| distance | 0.403087 | 0.035525 | -0.002339 | 1/3 |
| distance + direction | **0.414591** | 0.038689 | **+0.009165** | **3/3** |
| full relative pose | 0.411939 | 0.037164 | +0.006513 | 3/3 |

For distance+direction, the seedwise joint deltas are `+0.017724`,
`+0.008886`, and `+0.000886`. The corresponding mean paratope delta is
`+0.015614` with 3/3 positive seeds. The mean epitope delta is only `+0.002716`
with 2/3 positive seeds.

## Scientific interpretation

The result supports four precise conclusions:

1. Spatial topology is itself an important structural prior: topology-only is
   substantially above the dense sequence-only control on these three short
   development runs.
2. Local direction adds a small but consistent development signal beyond that
   topology. The signal is strongest and most stable on the antibody/paratope
   side.
3. Distance alone is insufficient after controlling for topology.
4. The full relative rotation matrix adds complexity without a demonstrated
   benefit over direction.

The result does **not** show that the model recognizes the correct antibody-
antigen partner. The antibody and antigen encoders do not exchange information,
and no native `residue_i x residue_j` contact matrix is predicted. A monomer
encoder can learn which positions are generally interface-prone without learning
which residues contact a particular partner.

This limitation matches current interaction literature. SPPIDER-seq conditions
predictions on the partner through cross-attention, while EpiFormer reports gains
from interleaved antibody-antigen fusion. Two-dimensional interaction splitting
is also necessary because either endpoint can leak across partitions:

- [SPPIDER-seq](https://pubmed.ncbi.nlm.nih.gov/42289971/)
- [EpiFormer](https://arxiv.org/abs/2606.04154)
- [DataSAIL two-dimensional splits](https://datasail.readthedocs.io/en/stable/workflow/splits.html)

## Highest-value next experiment

The next architecture should keep the selected direction-aware monomer encoders
and add an explicit, coordinate-frame-independent inter-chain pair state:

```text
antibody invariant residues ----\
                                  -> bipartite pair tensor [Lab, Lag, Cpair]
antigen invariant residues ------/             |
                                                v
                                  alternating row/column pair updates
                                                |
                                                v
                              native contact logit for every residue pair
```

Raw coordinates from unaligned monomers must never be subtracted across chains.
The pair tensor can be initialized from invariant residue embeddings, their
elementwise products/differences, chain-local geometry summaries, residue type,
and confidence masks. Bidirectional cross-attention and pair updates then learn
alignment implicitly. Bound-complex coordinates are used only to derive training
labels and never as inference inputs.

The frozen evaluation must include:

1. per-complex macro native pair-matrix AUPRC as the primary endpoint;
2. epitope/paratope marginals as secondary endpoints;
3. cognate versus matched-partner shuffle;
4. antibody-only, antigen-only, topology, composition, and degree-aware controls;
5. bidirectional Recall@1/5/10, MRR, normalized rank, and pairwise win rate;
6. two-axis sequence- and structure-leakage-controlled outer folds;
7. component-bootstrap confidence intervals and all seed-level results;
8. calibration, P@K/R@K, MCC, Dice/F1, and failure-stratum analysis.

The self-attention evidence should be completed before the outer test by running
the preregistered seeds `17`, `42`, `73`, `101`, and `137` with immutable
predictions and per-complex aggregation. Historical seeds cannot be renamed to
satisfy that protocol.

## Claim boundary

The strongest defensible statement is:

> On a sealed 2,676/36 SAbDab2 development split with frozen ESM2 inputs,
> equal-parameter three-epoch ablations, and fixed-final checkpoints,
> receiver-frame direction was the most consistently useful intra-chain
> geometry feature. The exact A100 checkpoints showed a +0.00988
> per-complex joint macro-AUPRC difference over topology-only, but the
> preregistered multiplicity-adjusted and multi-seed confirmatory gate did not
> pass, and epitope improvement remains uncertain.

No frozen test, partner-specific, external-generalization, clinical, benchmark-
improvement, or SOTA claim is authorized.

