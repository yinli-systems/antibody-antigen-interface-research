# FrameContact joint-coupling research lock — 2026-08-15

## Decision

The teacher's two hypotheses are implemented as distinct, matched development
controls rather than blended into one larger model:

1. **pair to nodes:** predict a native residue-pair contact matrix and derive
   epitope/paratope probabilities with a masked noisy-OR;
2. **nodes to pairs:** predict coarse epitope/paratope probabilities and use a
   stop-gradient soft prior to concentrate, but never delete, pair candidates.

The confirmatory comparison remains development-only. AsEP frozen/official and
external test roles are not authorized for architecture, checkpoint, threshold,
loss-weight, or baseline selection.

## What the primary sources actually establish

The distinction between *implemented option*, *paper default*, and *our local
control* matters.

- [EpiFormer](https://arxiv.org/abs/2606.04154) establishes early interleaved
  cross-attention and sparsity-aware antibody-conditioned epitope prediction.
  The paper itself derives each epitope probability by taking the arithmetic
  mean of the bipartite interaction probabilities over the antibody axis
  ([Eq. 9](https://arxiv.org/html/2606.04154v1#S3.SS3.SSS1)); it does **not**
  describe noisy-OR as the reported pair-to-node reduction.
  Its official repository at immutable commit
  [`f420745`](https://github.com/mansoor181/epiformer/tree/f420745e28adf759ccdab327d06459da9a9382fe)
  implements noisy-OR as
  `1 - product(1 - p_ij)` for both axes
  ([decoder lines 689–692](https://github.com/mansoor181/epiformer/blob/f420745e28adf759ccdab327d06459da9a9382fe/model/decoder.py#L689-L692)).
  However, the checked-in model default is `mean_row`, not noisy-OR
  ([configuration](https://github.com/mansoor181/epiformer/blob/f420745e28adf759ccdab327d06459da9a9382fe/conf/model/model.yaml#L129-L139));
  the checked-in loss default is `epi_only`, and edge prediction is disabled
  ([loss configuration](https://github.com/mansoor181/epiformer/blob/f420745e28adf759ccdab327d06459da9a9382fe/conf/loss/loss.yaml#L1-L35)).
  The repository also contains an optional sum-mass-to-node MSE consistency
  loss ([implementation](https://github.com/mansoor181/epiformer/blob/f420745e28adf759ccdab327d06459da9a9382fe/model/loss.py#L1301-L1380)).
  Therefore, “EpiFormer uses noisy-OR as its reported default three-task
  objective” is not an evidence-supported statement.
- [MIPE](https://www.ijcai.org/proceedings/2024/0669.pdf) jointly estimates
  paratope, epitope, and an interaction matrix from multimodal representations.
  It is a close conceptual predecessor, but not a same-protocol AsEP native
  pair-contact comparator.
- [SurfBind](https://arxiv.org/abs/2606.23830) motivates binder-aware,
  hierarchical coarse-to-fine surface reasoning. It supports testing coarse
  node-to-pair routing, but does not license hard pruning by ground-truth
  interface labels.
- [LF3DRoPE](https://arxiv.org/abs/2608.01092) motivates local-frame geometric
  attention for epitope prediction. Its published scores must remain separate
  from same-protocol local development results.
- [VASCIF](https://pubmed.ncbi.nlm.nih.gov/41993534/) motivates joint interface
  learning and neighborhood-scale ablations. A 10 Å neighborhood may be an
  auxiliary ablation; it must not silently replace the registered native-contact
  definition.

## Executable architecture and matched controls

Let `z_ij` be the masked native-contact logit for antibody residue `i` and
antigen residue `j`. FrameContact computes noisy-OR marginals in stable log
space:

```text
p_para_i = 1 - product_j (1 - sigmoid(z_ij))
p_epi_j  = 1 - product_i (1 - sigmoid(z_ij))
```

Invalid/padded pairs contribute exactly zero contact probability. Each loss is
balanced per complex so long antigens and large pair grids cannot dominate.

The frozen six-condition matrix is:

| ID | Pair output | Residue output | Coupling changed relative to prior row |
|---|---|---|---|
| A0 | native contact | none for node loss | pair-only control; specificity ranking held fixed |
| A1 | native contact | pair-derived noisy-OR | adds epitope/paratope label gradients only through the pair matrix |
| A2 | native contact | independent unary heads | separates shared-label supervision from contact-derived residues |
| A3 | native contact | independent unary heads | adds symmetric per-complex Bernoulli JS to noisy-OR marginals |
| A4 | native contact | independent unary heads | adds detached soft coarse node-to-pair routing |
| A5 | native contact | independent unary heads | adds equal-weight intrachain Cα graph smoothness |

For A4, the routed pair logit is:

```text
z'_ij = z_ij + log(floor + (1 - floor) * p_para_i * p_epi_j)
```

The coarse probabilities are detached for this routing term, and the positive
floor prevents hard pair deletion. This tests the teacher's “shrink the contact
space” idea without an irreversible crop, label leakage, or a shortcut through
the pair loss.

A3 uses a bounded symmetric Jensen–Shannon consistency loss, separately
averaged for antibody and antigen and then equally weighted. This is more
auditable under variable partner lengths than comparing raw summed edge mass to
node probabilities with a single learned scale.

A5 uses authenticated *intrachain* Cα radius-kNN graphs only. It is a spatial
regularizer, not cross-chain distance supervision and not evidence that the
model reconstructs a docked pose.

### Registered follow-up question after the active matrix

Raw noisy-OR is the exact probability of at least one contact only when the
pair probabilities are calibrated Bernoulli events under the required
independence assumption. It also accumulates probability with partner length.
EpiFormer's arithmetic mean has different semantics and removes that direct
length accumulation, while FrameContact's existing length-normalized noisy-OR
is a third, explicitly heuristic control.

The active 30-run A0–A5 matrix must not be changed. After it reaches 30/30 and
its pre-registered aggregation succeeds, a separate development-only protocol
may compare exactly three pair-to-node reductions with the architecture, pair
loss, node-loss weights, seeds, compute budget, and selection rule held fixed:

1. raw noisy-OR (the active A1 reference);
2. arithmetic mean of pair probabilities (EpiFormer Eq. 9 matched control);
3. length-normalized noisy-OR (partner-length sensitivity control).

This follow-up is warranted only as an attribution experiment. It must not use
AsEP frozen/official or external test roles, and it must report residue metrics
stratified by partner length in addition to the locked contact and retrieval
metrics. A better residue score alone cannot promote an aggregation if contact
ranking or bidirectional partner retrieval worsens.

## Selection and interpretation lock

- Five matched seeds: 17, 42, 73, 101, 137.
- Checkpoint selection: maximize the minimum bidirectional development partner
  margin; tie-break by mean per-complex contact AUPRC, then earlier epoch.
- Node metrics never select checkpoints.
- Report pooled and per-complex contact AUPRC, fixed-budget P@K, epitope and
  paratope metrics, partner top-1/MRR/margin, seed-wise deltas, and matched
  complex-cluster bootstrap intervals.
- Do not call epitope, paratope, and contact three independent biological tasks:
  the node labels are marginals of the same native pair geometry.
- A positive contact delta without a positive partner-retrieval delta does not
  establish partner specificity.
- No condition is called superior from a live epoch snapshot or a single seed.

## Current evidence boundary

The 30-run epitope-group development matrix is active. One A3 seed has ended
normally by early stopping; the other 29 tasks remain active at this lock. Live
trajectories are monitoring evidence only. Final claims wait for all receipts,
the pre-registered checkpoint rule, five-seed aggregation, paired bootstrap,
partner retrieval, and case-study audits.
