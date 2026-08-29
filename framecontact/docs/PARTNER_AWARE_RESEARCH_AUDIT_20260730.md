# FrameContact partner-aware research audit (2026-07-30)

## Decision

The primary scientific question is not whether geometry improves single-chain
interface propensity. It is whether a model assigns a cognate antibody a higher
interaction score than plausible non-cognate antibodies and moves the antigen
epitope map in the correct, partner-specific direction.

FrameContact v1 therefore keeps the already validated, parameter-unshared
intra-chain encoders but binds residue predictions to an explicit bipartite
contact head. Model selection uses the minimum of the two interaction-only
partner-retrieval margins. The official AsEP test remains sealed until the
five-seed rule, calibration, scorer, decontamination audit, and test partner
panel are frozen.

## Evidence hierarchy

### AsEP: task and external authority

- Primary source: [NeurIPS 2024 Datasets and Benchmarks paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/15add6732964d5b1f0954058bf3ccc88-Abstract-Datasets_and_Benchmarks_Track.html)
- Official code/data: [biochunan/AsEP-dataset](https://github.com/biochunan/AsEP-dataset), [Zenodo v1.2.0](https://zenodo.org/records/11495514)
- Relevant design: antibody-specific epitope prediction is conditioned on both
  graphs; the stronger formulation is bipartite residue-link prediction with a
  4.5 A contact definition.
- Boundary: the official `epitope_group` split is a useful grouped benchmark,
  but it is not automatically endpoint-disjoint, temporal, or equivalent to a
  Foldseek/TM-score antigen-fold split.

### EpiFormer: early-fusion comparison

- Primary source: [arXiv:2606.04154](https://arxiv.org/abs/2606.04154)
- Code: [mansoor181/epiformer](https://github.com/mansoor181/epiformer), audited
  at commit `a08935b30b288ac0e51d61ac75c5a8b712c6ea3a`.
- Relevant design: two non-shared encoders; interleaved bidirectional
  cross-attention; joint residue, bipartite edge, count/sparsity, and distance
  objectives.
- Directly reusable: interleaved invariant-feature exchange, explicit edge
  supervision, and the comparison between late and early fusion.
- Requires modification: its distance loss is defined from the bound complex
  coordinate system and cannot be used on independently posed monomers without
  a pose hypothesis. The repository's published best script uses one seed,
  ESM2-650M/AntiBERTy-style processed inputs, a fixed threshold, and evaluates
  the test role after validation training. Those results are not directly
  comparable to FrameContact's five-seed interaction-margin protocol or the
  native AsEP ESM2-35M development run.
- Evidence level: recent preprint with a small public repository, not yet a
  peer-reviewed benchmark authority.

### ProTact: invariant monomers and a cross-chain pair trunk

- Primary source: [Communications Biology 2026](https://www.nature.com/articles/s42003-026-09809-2)
- Code: [biomed-AI/ProTact](https://github.com/biomed-AI/ProTact)
- Relevant design: each monomer is first encoded with SE(3)-invariant node and
  edge representations. The cross-protein tensor is initialized by outer
  fusion of node states, then refined by a trigonometry module that uses both
  chains' intra-chain edges. No meaningless raw `x_i - x_j` is required before
  a relative pose exists.
- Directly reusable: a sparse pair trunk driven by intra-chain edge geometry,
  with surface features as soft evidence.
- Requires modification: the published antibody transfer test contains only
  19 complexes under an earlier split and is not evidence for SAbDab2
  antigen-fold generalization. The paper also reports a decline on unbound
  complexes with large conformational changes, reinforcing the need for apo and
  coordinate-perturbation controls.

### PPLM: paired-protein pretraining

- Primary source: [Nature Communications 2026](https://www.nature.com/articles/s41467-026-70457-5)
- Code: [junliu621/PPLM](https://github.com/junliu621/PPLM)
- Relevant design: a 650M paired sequence language model with separate
  intra-protein RoPE attention and non-positional inter-protein attention,
  trained on more than 3.3 million protein pairs; a downstream pair transformer
  predicts contacts.
- Directly reusable: the principle that inter-chain attention should not inherit
  an intra-chain sequential positional prior.
- Requires modification: weights and datasets need their own license,
  decontamination, and feature-custody receipts. PPLM cannot be called a fair
  baseline until its checkpoint, training cutoff, and overlap with AsEP and
  SAbDab2 are audited. The repository states a non-commercial license.

## What v1 tests now

1. Eight conditions separate sequence-only, spatial topology, distance,
   direction, full relative pose, a frame-point invariant bias, late
   FrameContact, and interleaved FrameContact.
2. Five preregistered seeds are selected only by the minimum bidirectional
   cluster-weighted interaction-only margin.
3. Wrong partners are ranking examples, never fabricated all-zero contact maps.
4. Residue-pair contact, epitope, paratope, ranking, calibration, attention, and
   partner-retrieval diagnostics are reported together.
5. A validation-only same-antigen/different-antibody panel tests directed
   epitope migration rather than arbitrary prediction change.

## v2 gates

No v2 component should be promoted because it is fashionable. Promotion needs
a failure mode from v1 and a validation-only improvement in partner specificity.

- Pair trunk: add 2-4 sparse row/column or triangle-style blocks initialized
  from invariant monomer states and intra-chain edge features.
- Sparsity objectives: compare balanced per-complex BCE with Dice and predicted
  count regularization; audit calibration because count matching can improve F1
  while damaging probability quality.
- Paired representation: benchmark frozen PPLM embeddings only after license,
  cutoff, and dataset-overlap audit.
- Pose ensemble: introduce cross-chain distance/orientation only after predicting
  one or more relative SE(3) poses; retain an invariant compatibility branch.
- Apo control: distinguish independent coordinate frames from holo-conformation
  leakage by using experimental apo structures or predicted monomers.
- Structural panel: replace exact-sequence-only wrong-partner clusters with a
  frozen structural-cluster panel when the Foldseek/US-align audit is complete.

## Non-negotiable failure criteria

- A residue AUPRC gain with a non-positive cognate-partner margin is not evidence
  of partner specificity.
- A nonzero prediction shift after swapping the antibody is not success unless
  the new prediction moves toward that antibody's known epitope.
- A development-only AsEP result is not an external benchmark result.
- A frozen test may be opened once and may not alter architecture, seed,
  checkpoint, temperature, threshold, exclusion, or panel selection.
- Bound-chain structures separated into two files remain holo conformations;
  coordinate-frame independence is not the same as apo generalization.
