# Intra-chain geometry-aware self-attention research memo

Updated: 2026-07-30

## Decision in one paragraph

For the first antibody/antigen self-attention stage, the best engineering and
scientific default is **ordinary sequence RoPE plus an invariant, ordered-pair
local-frame representation projected to additive per-head attention bias**. It
uses the real `N/CA/C` structure, preserves chirality and relative orientation,
lets antibody and antigen live in unrelated global frames, and can start as an
exact zero residual. The additive pair MLP and fixed-frame IPA-style point bias
are implemented confirmatory conditions. Two post-confirmatory candidates—an
interpretable multi-basis bias and a pair-conditioned local-pose rotary score—
are implemented behind a separate development-only registry and protocol.
Raw `xyz` RoPE, inertial/PCA canonicalization, full spherical-harmonic
SE(3)-Transformers, discrete structure tokens, and global equivariant Hyena are
useful comparisons or later architectures, but they are not equally safe
drop-ins for a frozen-ESM, residue-classification model.

This memo covers only intra-chain self-attention. It does not subtract antibody
and antigen coordinates in cross-attention when the two monomers are unaligned.

## What successful geometry modules actually inject

Let residue `i` have a local right-handed frame `(R_i, t_i)` built from `N`,
`CA`, and `C`, where `t_i = CA_i`. For an ordered pair `(i, j)`:

```text
d_ij = ||t_j - t_i||
u_ij = R_i^T (t_j - t_i)
O_ij = R_i^T R_j
```

`d_ij`, `u_ij`, and `O_ij` are unchanged by applying the same proper rotation
and translation to a monomer. `u_ij` and `O_ij` retain orientation and chirality
that distance alone cannot express. The current implementation expands `d_ij`
with Gaussian RBFs, uses the normalized `u_ij`, flattens `O_ij`, and appends
signed/absolute sequence separation. Cross-chain edges are excluded by integer
chain IDs.

AlphaFold 2's invariant point attention independently supports the same core
principle: queries and keys include learned 3D points expressed in residue-local
frames, while squared distances after transforming them to a shared frame make
attention invariant to the global reference frame
([Nature paper](https://www.nature.com/articles/s41586-021-03819-2),
[official code](https://github.com/google-deepmind/alphafold)). IgFold is even
closer to this project: it fixes template coordinates and uses IPA explicitly as
structure-aware self-attention over language-model representations
([Nature Communications](https://www.nature.com/articles/s41467-023-38063-x)).

## Method comparison and integration decision

| Family | Geometry enters as | Main advantage | Main failure/cost | Decision for this model |
|---|---|---|---|---|
| Distance RBF | pair feature / per-head bias | cheap, stable, naturally invariant | mirror-blind; no direction or orientation | direct baseline |
| Local direction | `R_i^T(t_j-t_i)` | ordered and chirality-sensitive | frame noise affects direction | direct baseline |
| Full relative pose | direction + `R_i^T R_j` | captures backbone orientation | more channels; may overfit small data | direct baseline |
| IPA | learned local query/key points and pair bias | geometry acts directly on attention and has strong protein precedent | more projections and pair-distance work | next strong baseline |
| ProteinMPNN-style edges | many inter-atom distances on a kNN graph | robust protein-design precedent | orientation is implicit; architecture is message passing | adapted distance-rich baseline |
| GVP/EGNN side stream | scalar/vector equivariant messages | preserves vector features through layers | new feature type and more integration risk | second-stage baseline |
| SE(3)-Transformer/Equiformer | irreducible representations and spherical harmonics | exact high-order equivariance | expensive basis construction; heavy dependency stack | high-cost upper bound, not first drop-in |
| Standard 3D/Lie RoPE | token-wise rotations generated from coordinates | modifies the native Q/K path | translation-relative does not imply arbitrary rotation invariance | raw xyz version rejected |
| Pair-conditioned pose rotary | pair phase from `(d,u,O)` | SE(3)-invariant QK modulation | generally cannot factor into one rotation per token; pairwise cost | experimental sparse branch |
| Structure tokens | discrete token appended/fused with sequence | reuses language-model machinery | quantization can lose continuous pair direction | later multimodal baseline |
| Geometric Hyena | equivariant long convolution | global context below quadratic attention cost | not a positional-encoding plugin | long-antigen architecture track |

Primary references behind these decisions include ProteinMPNN's backbone-distance
graph ([official repository](https://github.com/dauparas/ProteinMPNN)), GVP's
scalar/vector representation ([official repository](https://github.com/drorlab/gvp-pytorch)),
the original SE(3)-Transformer
([NeurIPS paper](https://papers.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html),
[official repository](https://github.com/FabianFuchsML/se3-transformer-public)),
and GeoMFormer's invariant/equivariant two-stream formulation
([ICML paper](https://proceedings.mlr.press/v235/chen24ac.html)).

## Why raw 3D RoPE is not the main proposal

RoPE guarantees a relative-position identity only for the coordinate system used
to generate its rotations. Applying axial or learned 3D RoPE directly to global
`xyz` can be translation-relative yet still change when the whole monomer is
rotated. STRING generalizes learnable translation-invariant rotations
([OpenReview](https://openreview.net/forum?id=3HLEfAFmox)); LieRE learns dense
Lie-group rotations and performs well on 2D/3D vision
([ICML/OpenReview](https://openreview.net/forum?id=yMJAYbGcCc),
[official code](https://github.com/StanfordMIMI/LieRE)). Neither result by itself
proves protein-frame SE(3) invariance.

Inertial-frame canonicalization followed by 3D RoPE is an interesting comparison
([GeoRoPE/InertialAR submission](https://openreview.net/forum?id=de4bbZedv3)),
but principal axes have sign ambiguities and become unstable near repeated
eigenvalues. Per-residue backbone frames avoid a single global canonicalization
and fail locally when atoms are missing, which can be masked explicitly.

A recent protein-specific preprint shows that a standard Transformer can learn a
Gaussian-like Euclidean distance filter from linear coordinate embeddings
([arXiv](https://arxiv.org/abs/2502.01533)). This is a useful learned-invariance
control, not a reason to remove analytic invariance: it requires the training
distribution and augmentation to teach robustness and offers no guaranteed
direction/chirality semantics.

## Sequence and structure fusion options

### 1. Concatenation

`Linear([ESM_i, structure_i])` is simple and is used in several protein models.
PepNN-Struct combines a local-coordinate graph representation with ProtBert
features ([Communications Biology](https://www.nature.com/articles/s42003-022-03445-2)).
The risk is scale imbalance: a high-dimensional ESM vector can dominate a small
geometry vector, or the model can overfit a redundant structure channel.

### 2. Additive pair bias

```text
attention_ijh = softmax_j(content_ijh + bias_h(g_ij))
```

This is the smallest controlled change. Zero-initializing the final geometry
projection makes the initial function exactly content-only on the selected graph.
It also exposes interpretable head-wise bias and supports a counterfactual that
removes only geometry while holding Q/K and the graph fixed.

### 3. Zero-initialized gated residual adapter

```text
x <- x + tanh(alpha) * StructureBlock(x, geometry)
```

This protects a pretrained sequence path and yields a scalar measurement of how
much the model uses structure. Modality dropout or coordinate corruption should
be applied only after the additive-bias baseline, so robustness regularization is
not confused with representation quality.

### 4. Interleaved sequence and equivariant graph layers

PPDiff's SSINC network interleaves global self-attention with local kNN
equivariant graph convolution
([ICML paper](https://proceedings.mlr.press/v267/song25d.html)). This is the most
relevant architecture-level alternative if a pair bias saturates: sequence
attention supplies global correlations and the graph layer supplies local 3D
messages. It is more invasive and must be parameter/compute matched.

### 5. Early bidirectional cross-chain fusion

EpiFormer interleaves antibody-antigen cross-attention inside GNN layers rather
than fusing only at the output
([arXiv](https://arxiv.org/abs/2606.04154),
[official code](https://github.com/mansoor181/epiformer)). ProTact builds
SE(3)-invariant monomer encodings and then reasons over a cross-chain interaction
map with geometry/physics-inspired modules
([Communications Biology](https://www.nature.com/articles/s42003-026-09809-2)).
These inform phase two. They do not authorize subtracting coordinates from two
unaligned input monomers in the current phase.

## Frozen confirmatory matrix now implemented

| Experiment | Attention graph | Pair features | Question answered |
|---|---|---|---|
| `sequence_dense` | all same-chain residues | zero | pure same-chain sequence Transformer |
| `spatial_topology` | same-chain 3D kNN/radius | zero | value of coordinate-derived adjacency alone |
| `distance` | same spatial graph | distance RBF | extra value of radial geometry |
| `distance_direction` | same spatial graph | distance + local direction | extra value of ordered orientation |
| `full_pose` | same spatial graph | distance + direction + relative rotation | extra value of full backbone pose |
| `frozen_frame_ipa` | same spatial graph | learned points in fixed residue frames | IPA-style bias, not complete IPA |
| `framecontact` | factorized dense/spatial paths | full pose + late contact-gated cross-attention | partner-aware late fusion |
| `framecontact_interleaved` | factorized dense/spatial paths | full pose + per-layer contact-gated cross-attention | partner-aware early fusion |

The pair-MLP feature ablations retain one fixed-width pair tensor, so unused
channels are zeroed and their trainable parameter counts match. The IPA-style
and partner-aware conditions have different parameterizations and report their
actual counts. Checkpoint selection, calibration, and ensemble construction are
validation-only and are frozen before any one-time test opening.

The previous four-mode development run used only one seed, three epochs, and 36
validation complexes. Its `none` control was spatial-topology-only. Those numbers
are useful for deciding what to carry forward, but do not establish a benchmark
gain. The current formal matrix uses seeds 17/42/73/101/137, a maximum of 200
epochs, patience 20, and the runtime-only v5 amendment. It is running on a
ParaCloud RTX 4090; this is not an A100 or completed-result claim.

## Attention diagnostics and what each one means

For every partner, layer, and head, validation now reports:

- normalized entropy over that query's actual graph support;
- total variation between learned attention and a counterfactual with only the
  additive geometry bias removed;
- attention mass on pairs with two valid frames;
- conditional expected C-alpha distance;
- conditional mass within 8, 12, and 16 Angstrom;
- mass lift above a uniform distribution on the same graph;
- attention mass within sequence separations 8, 16, and 32;
- mean absolute and RMS geometry-bias magnitude.

This distinction matters. A spatial graph mechanically limits distance even when
geometry features are zero; only `spatial_topology` versus `distance*` and the
same-graph counterfactual can attribute a change to learned geometry bias.

## Implemented exploratory innovation: Local-Pose Multi-Basis Attention

The first innovation candidate is deliberately incremental and falsifiable:

```text
b_ijh = gate_radial_h * f_radial(d_ij)
      + gate_direction_h * f_direction(u_ij)
      + gate_rotation_h * f_rotation(O_ij)

logit_ijh = q_i^T k_j / sqrt(d_h) + b_ijh
```

Each basis has its own predictor and bounded per-head gate. The gates—not both
the predictor and gate—start at zero; this preserves exact content-only logits
while giving the gates a non-zero first-step gradient. The three gated
components are returned separately in diagnostics. The method is development-
only under `framecontact_local_pose_exploratory_v1.json` and advances only under
its paired-seed margin, contact-AUPRC non-inferiority, invariance, intervention,
and calibration gates.

## Implemented exploratory innovation: pair-conditioned local-pose rotary score

For complex pairs of Q/K channels, generate an invariant phase from local pose:

```text
phi_ijhm = MLP_hm(RBF(d_ij), u_ij, O_ij)
rot_score_ijh = sum_m Re(q_ihm * conjugate(k_jhm) * exp(i * phi_ijhm))
logit_ijh += tanh(beta_h) * rot_score_ijh
```

Because `(d,u,O)` is invariant, the score is invariant under a shared proper
rigid transform. Because `phi_ij` depends on the ordered pair, it is not in
general factorizable into one precomputed rotation for token `i` and token `j`;
calling it a drop-in standard RoPE would be inaccurate. It should run only on the
sparse spatial graph, start with `beta_h = 0`, and be compared against an
equal-parameter additive-bias branch.

The implementation rotates a bounded prefix of Q/K complex channel pairs, masks
every invalid-frame pair, returns its per-head gates, and keeps the initial
additive score at exactly zero. It is excluded from the eight-condition
confirmatory registry. Falsification gates are:

1. exact independent-monomer SE(3) invariance test;
2. finite behavior at missing/degenerate frames;
3. reflection-sensitivity task where distance-only must fail;
4. coordinate-noise stress at 0.1/0.25/0.5 Angstrom;
5. five-seed fixed-budget development comparison;
6. wall-time and peak-memory profiling on the same A100;
7. no frozen-test access until architecture/epoch selection is frozen.

## Longer-term alternatives, not immediate scope

Geometric Hyena supplies sub-quadratic global equivariant context and reports a
large efficiency advantage at 30k tokens in its own all-atom benchmarks
([ICML paper](https://proceedings.mlr.press/v267/moskalev25a.html)); it deserves a
separate long-antigen track, not a plug-in claim. Structure tokenization also
deserves an independent track: StructTokenBench finds no single tokenizer wins
all criteria ([ICML paper and official code](https://proceedings.mlr.press/v267/yuan25i.html)),
while GeoBPE learns multi-scale geometric motifs
([ICLR paper](https://openreview.net/forum?id=55e5f3GVFc)). Neither replaces
continuous ordered-pair geometry for the present self-attention question.

The rapidly emerging antigen-conditioned models AAMFM
([preprint](https://arxiv.org/abs/2607.20057),
[official code](https://github.com/XL-S224/AAMFM)), AgForce
([preprint](https://arxiv.org/abs/2605.21610),
[official code](https://github.com/mansoor181/ag-force)), and ConTact
([preprint](https://arxiv.org/abs/2605.21600)) are tracked for phase-two controls:
partner shuffle, modality dropout, anti-shortcut gates, explicit contact-first
reasoning, and cross-modal adapters. Their claims are not transferred to the
current dataset without independent split and code audits.

## Evidence boundary

- `literature_catalog.json` contains 68 methods and labels peer-reviewed,
  preprint, and under-review evidence separately.
- Local tests demonstrate numerical contracts, invariance, masking, factorized
  graphs, prediction serialization, and aggregation; they do not demonstrate a
  biological improvement.
- Earlier exploratory runs and the original partial-coverage CHIMERA scoring
  are secondary historical evidence only.  The current formal five-seed
  development matrix is running on a ParaCloud RTX 4090, not AB03/A100.  No new
  A100, multi-seed, or benchmark-improvement claim is authorized until the
  corresponding AB03 receipts, per-run manifests, and checksums exist.
- The new 548-complex adapter-v3 cache is materialized and hash-bound, but its
  frozen-checkpoint secondary rerun does not exist until new AB03 prediction and
  result receipts are present.
- CHIMERA row-level annotations have now been accessed. Any adapter-v3 rerun is
  explicitly secondary; new method selection remains isolated from its scores.
