# Post-CHIMERA improvement plan

Updated: 2026-07-30

## Decision

The next iteration should **not** begin with a more elaborate `SE(3)-RoPE`.
The external result identifies three earlier bottlenecks:

1. the strict CHIMERA adapter unnecessarily depends on exact PDB residue axes;
2. the spatial graph replaces, rather than augments, global sequence attention;
3. intra-chain geometry is useful for paratope prediction, but antibody-specific
   epitope prediction additionally needs surface priors, an imbalance-aware
   objective, and eventually partner communication.

The recommended sequence is therefore:

1. PDB-coordinate-anchored CHIMERA adapter and coverage-only audit;
2. dense-sequence plus sparse-geometry dual path;
3. antigen surface/exposure features and complex-balanced loss;
4. frozen-frame sparse IPA as the stronger intra-chain baseline;
5. only then, gated content cross-attention without unaligned inter-chain
   coordinate differences.

## Evidence from the current result and implementation

The real external result showed a reproducible qualitative asymmetry:

- `spatial_topology` improved pooled paratope AUPRC on both CHIMERA splits;
- mean per-complex epitope AUPRC improved slightly, including a positive
  pre-gate temporal confidence interval;
- pooled epitope AUPRC fell on both splits;
- only about 66% of each split passed the strict adapter.

The code audit supplies two concrete explanations that must be tested rather
than asserted as conclusions:

- `spatial_topology` hard-masks attention to spatial neighbours, so it removes
  global ESM-mediated residue communication instead of adding a geometry path;
- training uses one dataset-wide positive weight per task and averages two
  residue-pooled BCE losses. Long antigens can therefore dominate optimization,
  while the model-selection metric is macro AUPRC and external reporting includes
  both macro and pooled AUPRC.

The current spatial implementation also materializes dense `[B,L,L]` geometry
and `[B,H,L,L]` attention tensors before masking. It is a graph ablation, not an
actual `O(LK)` sparse implementation.

## P0: PDB-coordinate-anchored official-contact adapter v3

### Why this is first

The official CHIMERA release contains sequence-aligned fixed-width coordinate
arrays, PDB structures, and explicit residue-contact tuples
([official repository](https://github.com/mansoor181/chimera-bench)). An initial
proposal was to recompute the documented 4.5-A contact rule directly from the
feature atom14 arrays. A complete release audit falsified that proposal:

- across 375 complexes accepted by the original exact-axis adapter, the release
  supplied 16,721 contact pairs;
- the feature atom14 arrays recovered only 11,226 of them (67.14%);
- only 5,597 atom14 minimum distances agreed with the stored distance to 0.01 A;
- full PDB all-atom distances agreed with all 16,721 stored values to 0.01 A;
- 584 contacts in 35 complexes were hydrogen-only at the 4.5-A threshold;
- residue-specific distal atoms such as TYR `OH`, THR `OG1`, and part of TRP's
  ring are absent from the fixed feature arrays.

Therefore feature atom14 cannot be treated as the label-generating atom set.
The official contact tuples remain the sole label authority.

### Adapter contract

For each official feature file:

1. keep `heavy_sequence`, `light_sequence`, and `antigen_sequence` as the model
   residue axes and `atom14[:, :3]` as `N/CA/C` input frames;
2. parse all supported first-model PDB residues, including hydrogens and explicit
   insertion codes;
3. anchor PDB identities onto feature indices only when CA coordinates match to
   0.001 A, residue types agree, and the one-to-one anchors are monotonic;
4. allow PDB-only terminal residues and feature-only unresolved residues, but
   fail if an official positive tuple refers to an unanchored residue;
5. map the provided contact tuples directly. Use stored all-atom distance only
   to resolve annotation keys that collapse distinct insertion-coded residues;
6. derive paratope and epitope labels by marginalizing those mapped official
   pairs; never regenerate positives from feature atom14 or a new cutoff;
7. fail closed on ambiguous mapping, duplicate mapped pairs, annotation/contact
   disagreement, hash drift, or missing source artifacts.

The completed coverage-only audit read no model scores. It accepted 548/561
unique complexes. Coverage is 283/292 (96.92%) on `antigen_fold.test` and
288/293 (98.29%) on `temporal.test`, passing the predeclared 95% coverage gate.
The 13 exclusions remain fail-closed: 12 official positive residues are absent
from the coordinate-anchored axes and one insertion-code ambiguity cannot be
uniquely resolved by the stored distance. The machine-readable audit is
`evaluation/CHIMERA_PDB_ANCHORED_COVERAGE_AUDIT_V3.json`.

### Evidence boundary

Changing the adapter after viewing the existing test result cannot make a rerun
on those same rows retrospectively confirmatory. The expanded-subset rerun is a
transparent robustness analysis. A new confirmatory claim needs an untouched,
cluster-audited external set or a future release whose labels were not used for
architecture, loss, threshold, or mapping decisions.

## P1: preserve sequence context and add geometry as a residual

### Dual-path block

Replace the hard choice between dense and spatial attention with:

```text
x_dense = DenseSequenceAttention(x, sequence_rope)
x_geom  = SparseGeometryAttention(x, neighbour_graph, pair_geometry)
x_next  = x + x_dense + tanh(alpha) * x_geom
```

`alpha` starts at zero. Antibody and antigen retain separate parameters. This
makes the initial function exactly the dense baseline and lets the experiment
measure whether geometry is learned as an addition rather than whether global
context can survive a graph bottleneck.

The first controlled comparison is:

| Mode | Dense path | Sparse topology | Pair geometry | Purpose |
|---|---:|---:|---:|---|
| dense reference | yes | no | no | existing sequence baseline |
| topology residual | yes | yes | no | adjacency value without pair features |
| distance residual | yes | yes | distance RBF | radial contribution |
| local-pose residual | yes | yes | distance + direction + rotation | pose contribution |
| frozen-frame SparseIPA | yes | yes | learned local points + pair bias | strong protein baseline |

AlphaFold 2's IPA combines scalar attention, pair bias, and distances between
learned points expressed in residue-local frames; the construction is invariant
to a global rigid transform
([Nature](https://www.nature.com/articles/s41586-021-03819-2)). IgFold provides
a closer integration precedent: fixed-coordinate IPA incorporates template
structure into language-model representations
([Nature Communications](https://www.nature.com/articles/s41467-023-38063-x)).

### Multi-graph neighbourhood

Pure Euclidean kNN is too destructive as the only communication graph. Use a
deduplicated union of:

- 16 nearest residues by same-chain sequence index;
- 16 nearest residues by C-alpha distance;
- a small deterministic global/landmark set or distance-biased sampled set.

SparseIPA uses the same general principle: sequence neighbours, spatial
neighbours, and distance-biased random neighbours; it also uses multi-backbone-
atom distances, local directions, and relative rotations
([Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01100-z)).
The exact counts above are starting points to tune only on strict validation,
not transferred benchmark optima.

### Richer pair features without a new input dependency

Before introducing surface meshes or side-chain coordinates, add all nine
`N/CA/C` inter-residue distances, each with shared RBF centers, alongside the
existing local direction and relative rotation. A later atom14 experiment can
add `O`, true/virtual `CB`, and residue-specific side-chain distances. This
separates the benefit of feature richness from the benefit of adding new atoms.

### Real sparse execution

After the reference block is validated, store neighbours as `[B,L,K]`, gather
K/V and pair features, and compute logits as `[B,H,L,K]`. Required checks:

- equality to the masked dense reference in FP32;
- BF16 forward/backward tolerance and finite-gradient tests;
- missing-frame and chain-boundary tests;
- A100 time and peak-memory curves at lengths 128, 256, 512, and 1,024;
- identical checkpoints and data batches for dense-versus-packed comparison.

A Triton/fused kernel is justified only after the packed PyTorch reference is
correct. No speedup claim is allowed before the A100 receipts exist.

## P2: antigen-specific surface/exposure branch

Backbone topology is a strong fit for CDR/paratope geometry but an incomplete
description of a discontinuous antigen surface. ScanNet learns binding sites
from local spatio-chemical atom neighbourhoods
([Nature Methods](https://www.nature.com/articles/s41592-022-01490-7)); PeSTo
progressively expands geometric neighbourhoods from local to longer range
([Nature Communications](https://www.nature.com/articles/s41467-023-37701-8)).

The newest direct evidence is SurfBind, a 2026 preprint. It represents molecular
surfaces as patches, uses invariant geodesic-distance RBFs in self-attention,
adds surface chemistry, and performs coarse-to-fine epitope prediction
([arXiv](https://arxiv.org/abs/2606.23830)). CHIMERA already supplies 128 surface
points, normals, two curvatures, and six chemical channels per partner.

Implement this in two risk levels:

1. **Direct, low-risk:** derive residue exposure features consistently for both
   SAbDab2 and CHIMERA: neighbour count, half-sphere exposure in the local frame,
   distance-to-local-convex-hull proxy, and kNN distance statistics. Feed them to
   an antigen-only zero-gated residual. Do not hard-mask buried residues.
2. **Adapted, higher-cost:** create surface tokens from points/normals/curvature/
   chemistry, encode them with geodesic/local attention, then cross-attend from
   antigen residues to their own surface tokens. The same surface-generation
   pipeline must be applied to training and external data.

The proposed innovation is **Surface-Gated Local-Pose Attention**:

```text
g_i = sigmoid(MLP(exposure_i, curvature_i, confidence_i))
x_i <- x_i + tanh(alpha) * g_i * SparseLocalPoseAttention(x, geometry)_i
```

It is falsified if gains disappear after controlling for topology, if it only
improves thresholded metrics through calibration drift, or if it is unstable to
0.1/0.25/0.5-A coordinate noise.

## P3: align the objective with the reported task

The current loss is dataset-wide positive-weighted BCE pooled over residues.
The first loss study should use the same architecture and compare:

1. current pooled weighted BCE;
2. per-complex normalized weighted BCE;
3. per-complex BCE plus a small epitope-cardinality/sparsity penalty;
4. the previous loss plus Dice;
5. a per-complex AP surrogate as a research control.

EpiFormer combines reweighted BCE, Dice, and a sparsity/cardinality term. Its
ablation reports that the sparsity term provided the largest F1 improvement,
whereas Dice alone did not, and InfoNCE reduced performance
([2026 preprint](https://arxiv.org/abs/2606.04154)). This is a reason to ablate
the terms separately, not to copy the full loss.

Direct AP optimization is possible but should remain a higher-risk control;
SOAP is a principled stochastic AUPRC objective
([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/0dd1bc593a91620daecf7723d2235624-Abstract.html)).
It changes optimization and batching assumptions, so it must not be bundled with
an architecture change.

All calibration, threshold selection, and task-loss weights belong to strict
validation. Report both macro per-complex AP and pooled AP, plus Brier score/ECE,
top-k precision/recall, and performance stratified by antigen length and epitope
prevalence. EpiFormer itself reports worsening F1 and greater AUPRC variation for
large antigens as prevalence falls, making this stratification essential.

## P4: partner-specific interaction without aligned monomer frames

For unaligned monomers, the first cross-attention must be content-based:

```text
H_ag <- H_ag + tanh(alpha_ag) * CrossAttn(H_ag, H_ab, H_ab)
H_ab <- H_ab + tanh(alpha_ab) * CrossAttn(H_ab, H_ag, H_ag)
```

There is no raw `x_ag - x_ab`, inter-chain distance, or relative-frame bias.
SurfBind explicitly omits the inter-partner geometric term when relative
distances are unknown. EpiFormer interleaves gated bidirectional cross-attention
between two independent geometric encoders and predicts a bipartite interaction
matrix; its reported ablations support early rather than output-only fusion
([arXiv](https://arxiv.org/abs/2606.04154)).

This phase should predict residue-pair contact logits first, then obtain
epitope/paratope marginals from the pair map. Mandatory controls are partner
shuffle, wrong-antibody retrieval, swapped chain direction, and a late-fusion
baseline. Attention weights alone are not contact predictions.

AlphaFold 3 supplies an important training warning: intrachain/local metrics
learned much earlier than protein-protein interface metrics
([Nature](https://www.nature.com/articles/s41586-024-07487-w)). Therefore the
cross-chain phase needs a separate budget and checkpoint rule; failure after a
short run is not evidence that partner fusion is useless.

## Locked next experiment matrix

### Phase A: data and diagnosis

1. adapter-v3 coverage audit only, with no model scores (completed: 548/561,
   with both split coverages above 95%);
2. expanded-subset rerun of the existing frozen checkpoints, labeled secondary;
3. break down delta AP by antigen length, prevalence, frame coverage, and
   predicted-positive count;
4. reliability curves, Brier/ECE, and macro-versus-pooled contribution plots.

### Phase B: self-attention development on strict validation

1. dense reference;
2. hard spatial topology, retained as the old control;
3. dense + topology residual;
4. dense + distance residual;
5. dense + full local-pose residual;
6. dense + nine-distance local-pose residual;
7. dense + frozen-frame SparseIPA;
8. winning geometry with per-complex BCE;
9. winning geometry with sparsity and Dice terms isolated;
10. winning configuration plus low-risk antigen exposure gate.

Use seeds 17/42/73, identical initialization where shapes permit, a fixed epoch
budget, validation-only selection, complex/cluster bootstrap, coordinate-noise
stress, and parameter/runtime matching.

### Phase C: evidence upgrade

- freeze architecture, loss, threshold, seeds, and adapter before new labels;
- evaluate a PDB-, sequence-cluster-, and epitope-group-decontaminated AsEP
  protocol, and separately stress unbound/predicted structures when available;
- separately report residue classification, pair-contact prediction, calibration,
  and A100 throughput/memory;
- do not call the existing CHIMERA residue task its canonical CDR-design
  leaderboard task. CHIMERA's official primary task and 12 metrics are for
  epitope-conditioned sequence-structure co-design
  ([official repository](https://github.com/mansoor181/chimera-bench)).

## Priority table

| Priority | Change | Status | Expected information gain |
|---|---|---|---|
| P0 | PDB-coordinate-anchored official-contact adapter | completed secondary audit | passes the coverage gate without relabeling |
| P0 | Macro/pool/calibration/length failure analysis | direct | explains the epitope metric disagreement |
| P1 | Dense + zero-gated sparse geometry residual | direct | tests geometry without deleting global context |
| P1 | True packed `O(LK)` attention | direct after reference | engineering scalability and honest A100 benchmark |
| P1 | Per-complex BCE + isolated sparsity term | direct | aligns training with per-complex evaluation |
| P2 | Nine `N/CA/C` pair distances | small adaptation | richer geometry without new source atoms |
| P2 | Frozen-frame SparseIPA | medium adaptation | strongest mature intra-chain comparison |
| P2 | Antigen exposure/surface gate | medium adaptation | targets the paratope/epitope asymmetry |
| P3 | Interleaved content cross-attention | phase-two adaptation | partner specificity without frame alignment |
| P3 | Surface-token binder matching | research track | highest potential and highest data-pipeline risk |
| P4 | Pair-conditioned local-pose rotary score | research control | novel but lower priority than fixing data/fusion |
| P4 | FlashIPA/factorized IPA execution | engineering adaptation | linear-memory mature baseline after correctness |
| P4 | ProTact-style triangular pair-map refinement | phase-two research | explicit many-body inter-chain contact reasoning |

## Claims that remain prohibited

- The current result does not prove a CHIMERA benchmark improvement.
- The positive temporal macro-AP interval does not override the failed coverage
  gate or the pooled epitope decline.
- A masked dense implementation is not a sparse-kernel speedup.
- A geometry attention map is not a residue-pair contact prediction unless it is
  trained and evaluated as one.
- Same-test adapter expansion is not a new frozen test.
- Feature atom14 cannot reproduce the official all-atom CHIMERA labels and must
  not be used to regenerate positives.
- Recent EpiFormer and SurfBind results are preprint evidence until independently
  reproduced under the project's split and metric definitions.
