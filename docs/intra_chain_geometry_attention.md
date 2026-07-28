# Phase 1: intra-chain geometry-aware self-attention

## Scope

This implementation covers the first stage agreed with the advisor:

- antibody and antigen are encoded by two self-attention stacks with no shared
  parameters;
- one residue aggregates information only from spatial neighbors in the same
  chain;
- antibody and antigen coordinates may be expressed in unrelated global frames;
- no inter-chain coordinate encoding is added to cross-attention in this phase.

The implementation is an untrained architecture candidate. It does not establish
a biological benchmark improvement, a trained PAIRFormer result, or a fused-kernel
speedup.

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

## Existing-model integration

Both `PartnerSpecificInteractionModel` and `ProspectivePairGeometryModel` expose
the same opt-in constructor settings:

```python
model = ProspectivePairGeometryModel(
    ...,
    num_fusion_layers=0,             # self-attention-only phase/ablation
    intra_chain_geometry_layers=2,
    intra_chain_geometry_feature_mode="full_pose",
    intra_chain_geometry_neighbor_k=32,
    intra_chain_geometry_neighbor_radius=16.0,
)

output = model(
    antibody_features,
    antigen_features,
    antibody_mask,
    antigen_mask,
    candidate_pair_mask,
    antibody_backbone=antibody_backbone,
    antigen_backbone=antigen_backbone,
    antibody_chain_ids=antibody_chain_ids,
    antigen_chain_ids=antigen_chain_ids,
)
```

The geometry stage is placed after the independent residue projections and before
any partner fusion. Its layer count defaults to zero. With the default, the old
state-dict keys and numerical execution path are unchanged. Supplying coordinates
while the stage is disabled fails loudly instead of silently ignoring structure.

## Locked first experiment

`scripts/train_sabdab2_structure_ablation.py` keeps cross/partner fusion at zero,
uses frozen sequence embeddings, and compares five equal-parameter controls:

1. dense same-chain sequence-only attention (`dense_sequence`);
2. spatial topology only (`topology_only`);
3. distance RBF;
4. distance RBF plus local direction;
5. full local relative pose plus sequence separation.

The four spatial modes use the same `k=32`, 16-A same-chain graph. All five use
the same pair-tensor width, geometry MLP, model initialization, optimizer, sample
order, training budget, and fixed-final-epoch reporting rule. The dense control
uses `geometry_feature_mode="none"` and
`attention_topology="dense_sequence"`, so its attention graph is independent of
coordinates and frame validity. The topology-only control uses the same all-zero
pair feature with `attention_topology="spatial"`; its only structural signal is
which edges survive the kNN/radius graph.

The feature selector accepts `none`, `distance`, `distance_direction`, or
`full_pose`; the independent topology selector accepts `spatial` or
`dense_sequence`. All modes keep the same fixed-width pair tensor and geometry
MLP. Unused channels are zeroed, so all five ablations have identical trainable
parameter counts.

The training script reports residue-level paratope and epitope AUPRC/AUROC plus
fixed-threshold metrics on validation. It does not perform validation-best
checkpoint selection. Results remain development-only: the native release test
and the repository's frozen test stay sealed, and a difference on this small
validation intersection is not a confirmatory benchmark improvement.

## Known boundaries and next phase

- Geometry tensors and attention logits are still dense `O(L^2)` tensors even
  when the kNN mask is sparse. There is no GPU fused or block-sparse kernel.
- Coordinate confidence is not yet used as a pair gate. It can be added after the
  basic ablation establishes whether local pose helps.
- The implementation produces invariant scalar residue embeddings, not
  SE(3)-equivariant vector outputs.
- Inter-chain geometry remains deferred. A later cross-attention design needs a
  learned frame-alignment or docking-hypothesis mechanism; raw coordinates from
  two unaligned monomers must not be subtracted directly.

## Verification

Focused tests cover independent SE(3) transforms, local-frame invariance,
same-chain neighbor masking, missing frames, padding, zero initialization,
parameter non-sharing, integration, and gradients:

```bash
.venv/bin/python -m pytest -q \
  tests/test_intra_chain_geometry.py \
  tests/test_pair_geometry.py \
  tests/test_frontier_model.py \
  tests/test_interaction_first.py
```

The full portable suite retains its pre-existing physical-evidence failures for
files that are intentionally absent from the curated snapshot. Those failures are
not related to this module and are not counted as passes.
