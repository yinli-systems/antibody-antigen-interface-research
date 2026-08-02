# Distogram and pose-free triangle constraint: research boundary

Status: development hypothesis and implementation note, not a performance or
novelty claim.

## What is established

- AlphaFold2 updates pair representations with triangle multiplicative updates
  and triangle attention because a pairwise description must satisfy geometric
  consistency constraints, including triangle inequalities
  ([Jumper et al., 2021](https://www.nature.com/articles/s41586-021-03819-2)).
- CDPred predicts inter-chain distance maps instead of an independent binary
  contact map and evaluates the near-distance projection with top-L precision
  ([Guo et al., 2022](https://www.nature.com/articles/s41467-022-34600-2)).
- DeepInter applies learned triangle update and triangle self-attention to
  inter-protein contact prediction
  ([Lin et al., 2023](https://www.nature.com/articles/s42256-023-00741-2)).
- ProTact separately encodes two monomer structures into SE(3)-invariant
  representations and iteratively refines a cross-chain interaction map with a
  trigonometry module and surface features
  ([ProTact, 2026](https://www.nature.com/articles/s42003-026-09809-2)).
- AsEP formulates antibody-specific epitope prediction as bipartite link
  prediction between antibody and antigen graphs, making pair-level contact
  supervision a better-defined target than two independent node classifiers
  ([Liu et al., NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/15add6732964d5b1f0954058bf3ccc88-Abstract-Datasets_and_Benchmarks_Track.html)).

Consequently, neither “use a distogram” nor “use triangle updates” is novel by
itself.

## FrameContact decision

The binary-contact and distance objectives are not represented by two
independent heads. A single categorical distogram predicts

\[
p_{ij}(b)=P(d_{ij}\in b),
\]

and contact probability is the exact registered near-bin projection

\[
P(C_{ij}=1)=\sum_{b:\,d_b<4.5\,\text{A}}p_{ij}(b).
\]

This prevents a distance head and a binary mask head from assigning mutually
inconsistent probabilities. The model still has a non-learned
`pair_candidate_mask` for missing/padded residues; it is not a second contact
predictor.

The additional regularizer does not align the antibody and antigen coordinate
frames. For antibody residues `i,k`, antigen residue `j`, known antibody
intra-chain distance `d(i,k)`, and predicted inter-chain distances `D`, it
penalizes violations of

\[
|D(i,j)-D(k,j)| \le d(i,k)+\epsilon,
\]

with the symmetric antigen-side constraint. These inequalities are invariant
to independent rigid transformations of the two monomers and are necessary,
but not sufficient, for a realizable complex geometry.

The constraint is sparsely routed to expanded IMGT CDR residues and an
isolated-antigen solvent-exposed local patch. Patch seeds come from the
antigen's intra-chain epitope-unary logits, not post-cross features. This is an
antibody-specific engineering hypothesis inspired by, but materially smaller
than, a learned Pairformer/DeepInter/ProTact triangle trunk.

## Falsifiable comparison

The five-seed development comparison is:

1. frozen binary-contact hybrid reference;
2. distogram-derived contact, no triangle regularizer;
3. the same distogram plus the sparse CDR-by-surface-patch constraint.

The triangle condition is useful only if it improves contact AUPRC and P@K or
fixed-threshold precision without retaining less than 95% of the paired-seed
recall and without materially degrading reciprocal partner-retrieval margin.
Precision gains caused only by predicting fewer positives are therefore not a
success.

## Evidence boundary

This design may be described as a pose-free, antibody-routed geometric
regularizer. It must not be described as a novel triangle architecture or a
benchmark improvement until the registered multi-seed development ablation and
a new independent confirmatory evaluation succeed. The current SAbDab2 test
role no longer supports a procedural “labels never read” claim because an
interrupted preprocessing job read a subset of its pair-label artifacts; no
model prediction or test metric was produced by that job.
