# Phase 1: intra-chain geometry-aware self-attention

## Scope

This implementation covers the first stage agreed with the advisor:

- antibody and antigen are encoded by two self-attention stacks with no shared
  parameters;
- one residue aggregates information only from spatial neighbors in the same
  chain;
- antibody and antigen coordinates may be expressed in unrelated global frames;
- no inter-chain coordinate encoding is added to cross-attention in this phase.

The implementation now includes trained-development execution paths, but code
and development validation do not establish a frozen-test or benchmark
improvement. A real Boltz Pairformer extraction/adapter path is implemented;
the official checkpoint hash, extracted-weight receipt, strict 48-block load,
and synthetic real-weight forward now exist. Until identical-split five-seed
runs and scorer receipts exist, it is not a Pairformer benchmark result. There
is still no fused-kernel speedup claim.

## Geometry and attention

For residue `i`, a right-handed local frame is built from its `N`, `CA`, and `C`
atoms. The origin is `CA`; the x axis points from `CA` to `C`; the y axis is the
Gram--Schmidt component of `CA -> N`; and the z axis is their cross product.

For an ordered same-chain pair `(i, j)`, the pair feature contains:

1. an RBF expansion of `||CA_j - CA_i||`;
2. the unit displacement `R_i^T (CA_j - CA_i)` in residue `i`'s frame;
3. the relative orientation `R_i^T R_j`;
4. signed and absolute sequence separation.

These quantities are invariant to any global rotation and translation of the
monomer. The antibody and antigen can therefore be transformed independently
without changing their encoded outputs.

Each attention head uses

```text
softmax_j(q_i^T k_j / sqrt(d_head) + MLP_head(g_ij))
```

over a same-chain k-nearest/radius-filtered neighbor set. Q and K also receive
ordinary 1D sequence RoPE. The final geometry-to-head projection is initialized
to zero, so geometry starts as a controlled residual signal instead of imposing
an arbitrary random bias.

This is “PairFormer-like” only in the precise sense that a pair representation is
projected to additive per-head attention biases. It is not a full MSA PairFormer:
there is no MSA axis, triangle multiplication, triangle attention, or pair-stack
recycling in this phase.

Two isolated exploratory modes are also available. `local_pose_multibasis`
returns separately gated radial, local-direction, and relative-rotation biases.
`local_pose_rotary` predicts an invariant ordered-pair phase and rotates a
bounded subset of Q/K complex channel pairs; it is not token-factorizable
standard RoPE. Both mask missing frames, start at exactly zero additive bias,
and are absent from the frozen eight-condition registry.

## Tensor contract

| Input | Shape | Meaning |
|---|---:|---|
| residue features | `[B, L, C]` | sequence/model embedding |
| residue mask | `[B, L]` | `True` for non-padding residues |
| backbone | `[B, L, 3, 3]` | atom order `N, CA, C`; final axis is xyz |
| chain IDs | `[B, L]` | optional integer IDs; separates H/L or multichain antigen |
| positions | `[B, L]` or `[L]` | optional sequence positions for 1D RoPE |

Degenerate or non-finite frames are marked invalid. A residue with a missing frame
keeps only a self edge, receives zero geometric features, and stays numerically
finite. Padding queries are returned as exact zeros.

## Authenticated SAbDab2 batch adapter

`scripts/materialize_sabdab2_structure_cache.py` builds the real structure side
of this contract. It authenticates the SAbDab2 ML archive, the development split,
the independent Needleman--Wunsch split audit, the canonical sequence manifest,
and the prepared embedding/label inventory by caller-supplied SHA-256 digests.
It then takes the intersection of:

1. the locked development train/validation membership; and
2. the new SAbDab2 release's native `ab_ag_split=train` role.

Only mmCIF members in that intersection are read. Native-release test structures
are excluded before mmCIF parsing. Each artifact contains canonical-axis
`N/CA/C`, integer chain IDs, frame-valid and missing-frame masks, sequence hashes,
and its source-mmCIF digest. Residue numbering and amino-acid identity must match
the frozen ESM/label axes exactly; mismatches are excluded with a recorded reason.

`Sabdab2StructureEmbeddingDataset` joins those artifacts lazily with frozen ESM2
chunks and binary paratope/epitope labels. `Sabdab2StructureBatchCollator` pads
coordinates with NaNs, chain IDs with `-1`, and produces explicit residue masks.
The classifier recomputes frame validity from every batch and requires it to
equal the adapter's valid/missing partition before attention is evaluated.

## Standalone use

```python
from interaction_first.intra_chain_geometry import DualIntraChainGeometryEncoder

encoder = DualIntraChainGeometryEncoder(
    d_model=256,
    num_heads=8,
    num_layers=2,
    geometry_feature_mode="full_pose",
    num_rbf=16,
    distance_max=20.0,
    neighbor_k=32,
    neighbor_radius=16.0,
    dropout=0.1,
)

antibody, antigen, diagnostics = encoder(
    antibody_features,
    antigen_features,
    antibody_mask,
    antigen_mask,
    antibody_backbone,  # [B, Lab, 3, 3], atom order N/CA/C
    antigen_backbone,   # [B, Lag, 3, 3], atom order N/CA/C
    antibody_chain_ids=antibody_chain_ids,
    antigen_chain_ids=antigen_chain_ids,
    return_attention=True,
)
```

`encoder.antibody_encoder` and `encoder.antigen_encoder` are separate module
instances. Their QKV, output, FFN, LayerNorm, and geometry-bias parameters do not
share storage.

## Authenticated real-Pairformer bridge

`OfficialBoltzPairformerMonomerEncoder` accepts the same frozen sequence
features, residue mask, `N/CA/C`, and chain IDs. It constructs intra-chain
distance/direction/relative-rotation features and projects them into the
official Boltz-1 Pairformer pair width. Missing frames are explicit: their pair
geometry is zero and they never create an invalid-frame geometry claim.

`DualOfficialBoltzPairformerMonomerEncoder` creates two full, parameter-disjoint
instances for antibody and antigen. Both frozen trunks start from the same
authenticated official weights, while the sequence/pair/output adapters are new
trainable task parameters. The exact extraction, dependency, smoke, and
comparability boundary is documented in `PAIRFORMER_BASELINE.md` and frozen in
`BASELINE_LOCK_V4.json`.

## Frozen-ESM classifier integration

`IntraChainResidueClassifier` projects antibody and antigen ESM embeddings with
separate parameters and applies `DualIntraChainGeometryEncoder` before independent
residue heads. There is no cross-attention in this classifier. Calling it with
`return_attention=True` returns both partner stacks' head-resolved diagnostics.

## Factorized confirmatory and exploratory experiments

The legacy residue-only script remains available for the original six-mode
study. The formal FrameContact trainer freezes eight conditions:
`sequence_dense`, `spatial_topology`, `distance`, `distance_direction`,
`full_pose`, `frozen_frame_ipa`, `framecontact`, and
`framecontact_interleaved`.

The pair-MLP feature ablations retain a fixed-width tensor and equal parameter
counts. Architecturally different IPA and cross-attention baselines report their
actual counts instead of claiming a false match. The formal protocol uses five
seeds and validation-only model selection. The two local-pose innovations use
`scripts/train_framecontact_exploratory.py` and a separate hash-bound protocol;
the normal trainer CLI cannot select them.

The script additionally saves no-pickle per-complex prediction arrays and
head-wise attention diagnostics. The multi-seed aggregator verifies axes,
labels, hashes, parameters, and initialization before producing paired
development-complex bootstrap intervals.

## Known boundaries and next phase

- Geometry tensors and attention logits are still dense `O(L^2)` tensors even
  when the kNN mask is sparse. There is no GPU fused or block-sparse kernel.
- Coordinate confidence is not yet used as a pair gate. It can be added after the
  basic ablation establishes whether local pose helps.
- The implementation produces invariant scalar residue embeddings, not
  SE(3)-equivariant vector outputs.
- The current cross-chain module exchanges invariant scalar content and predicts
  an explicit contact map; it does not subtract coordinates from unaligned
  monomers. A later latent-pose/docking branch remains necessary before genuine
  inter-chain distance/orientation encoding is meaningful.

## Verification

Focused tests cover independent SE(3) transforms, local-frame invariance,
factorized dense/sequence/spatial graphs, coordinate-independent sequence-only
attention, missing frames, padding, zero initialization, parameter non-sharing,
diagnostics, prediction serialization, aggregation, and gradients:

```bash
.venv/bin/python -m pytest -q \
  tests/test_intra_chain_geometry.py \
  tests/test_sabdab2_structure.py \
  tests/test_multiseed_aggregation.py
```

The current full-suite pass count is reported in the README after a clean run.
It verifies code behavior, not biological performance or A100 execution.
