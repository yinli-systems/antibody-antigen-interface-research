# Partner-specific frontier update — 2026-08-01

## Outcome

The newest 2026 evidence strengthens the case for explicit partner conditioning,
but it also separates four tasks that must not be collapsed into one leaderboard:

| Authority | Actual task | Direct use for FrameContact |
|---|---|---|
| SPPIDER-seq | partner-conditioned residue interface propensity from two sequences | external sequence-only residue baseline |
| PEPNet | antibody-conditioned antigen epitope classification from atomic point clouds | direct AsEP localization baseline after the external opening gate |
| CALM | bidirectional antibody-antigen retrieval with contrastive encoders | retrieval objective/baseline, not contact localization |
| AAMFM | antigen/epitope-conditioned CDR generation with ESM3, GearNet and Cal-DPO | downstream co-design comparator, not epitope discovery |

The machine-readable source, weight, license and comparability ledger is
[`BASELINE_LOCK_V6.json`](../baselines/BASELINE_LOCK_V6.json). None of the
published metrics is an identical-split FrameContact comparison.

## Evidence that changes the model plan

### SPPIDER-seq is now the minimum partner-aware residue baseline

The published Bioinformatics release uses ESM2 embeddings, cross-attention and
query-residue classification. The manuscript tag is pinned at
`9c12e8c826ad4e47b00477cab56f6d58715d8f60`; its two 32.8 MB PyTorch checkpoints
have been downloaded only for hashing, not deserialized. Because the repository
is GPLv3, FrameContact will invoke it only through a separate external environment
and will not copy its implementation.

It is still not a sufficient primary baseline by itself: the released models are
generic receptor/peptide predictors, do not expose residue-pair contacts, and do
not use the present antibody/antigen double-cold split. The earlier SAbDab2
SPPIDER-seq result therefore remains a separate identical-data retraining result;
published generic-PPI numbers cannot replace it.

### PEPNet is a serious direct epitope baseline, but its test assets are already in tree

PEPNet uses atomic point clouds, atom-to-residue hierarchical aggregation, RoPE,
masked-feature pretraining and antigen-antibody cross-attention. It is therefore
the closest new direct comparator to the teacher's original geometry-aware
self-attention question. The source is MIT licensed and pinned at
`fa0fb4002c5294a094bf4d050098692067e545f4`, but no checkpoint is included.

The repository also contains AsEP train/validation/test ID lists and labeled NPZ
examples. It must be placed behind the existing AsEP one-time opening gate rather
than imported into development. Published MCC/AUC values use the authors' task,
threshold and split and remain literature context only.

### CALM motivates a retrieval head, not a localization shortcut

CALM's trained result is a dual-encoder contrastive model; the cross-attentive
autoregressive decoder is proposed but not trained. Its full-sequence and true
5 A interface-mask retrieval experiments are separate. The stricter antigen
sequence split still does not enforce antibody clusters or antigen structural
novelty, and the reported score depends on a fixed candidate pool.

FrameContact should therefore borrow the symmetric multi-positive retrieval
objective, but never feed true epitope/paratope masks at validation or test time.
The compatible version pools residues using the model's own predicted contact
marginals.

### AAMFM defines the eventual downstream endpoint

AAMFM injects antigen geometry and supplied epitope information through a
cross-modal adapter into ESM3, then optimizes generated CDRs with a structural
preference signal. This supports the long-term plan to connect FrameContact's
contact representation to CDR co-design. It does not test whether an unknown
epitope can be discovered from an antibody-antigen pair. Its one-commit source
has no license or released checkpoints, so it is design evidence only.

## New candidate: contact-bottleneck contrastive specificity

After the current binary-contact/distogram and residue-head choices are frozen,
add one validation-only candidate that makes localization the bottleneck for
retrieval:

\[
  \Delta C_{ij} = \max\left(0,
  \sigma(\ell_{ij})-\sigma(\ell_{ij}-s_{ij})\right)
\]

where `ell` is the pre-cross total contact logit and `s` is its bilinear
interaction term. Then

\[
  w_i^{Ag} = 1-\prod_j(1-\Delta C_{ij}), \qquad
  w_j^{Ab} = 1-\prod_i(1-\Delta C_{ij})
\]

\[
  z_{Ag} = \frac{\sum_i w_i^{Ag}h_i^{Ag}}{\sum_i w_i^{Ag}+\epsilon}, \qquad
  z_{Ab} = \frac{\sum_j w_j^{Ab}h_j^{Ab}}{\sum_j w_j^{Ab}+\epsilon}
\]

Use `z_Ag` and `z_Ab` in a reciprocal contrastive loss over the frozen
wrong-partner panel. Compare two registered variants only:

1. stop-gradient through `w`, testing whether contact localization already
   contains useful specificity;
2. joint gradient through `w`, testing whether retrieval supervision improves
   the contact map or merely distorts it.

Promotion requires improvement in both retrieval directions while preserving
contact macro AUPRC, epitope/paratope AUPRC, fixed-threshold precision/recall and
calibration within the frozen non-degradation gate. A representation that wins
retrieval by producing diffuse or arbitrary masks fails.

### Engineering status and exact boundary

The head is now implemented behind
`--contact-bottleneck-projection-dim`; the default value is zero, so existing
models, state dictionaries, frozen protocols, and running arrays are unchanged.
It is intentionally restricted to `fusion_mode=late`: the pooled residue
features and the contact decomposition both come from the final independently
encoded **pre-cross** monomer states. The weights use counterfactual positive
contact lift: total contact probability minus the probability obtained after
removing the pair interaction term. Enabling it for
interleaved cross-attention fails closed because those layer states have already
seen the partner and cannot honestly be described as monomer self features.

This counterfactual form fixes an important probability-semantics failure in the
first engineering sketch: an interaction-only logit has neither the contact
bias nor the two unary terms and is therefore not a calibrated contact log-odds.
Applying noisy-OR directly to near-zero interaction logits would saturate on a
long partner and collapse toward uniform whole-chain pooling. In the corrected
form, a pure unary prediction gives exactly zero lift, while positive
partner-specific interaction receives a bounded probability attribution before
noisy-OR.

`stop_gradient` detaches only the logits used to form the noisy-OR weights; the
projectors and monomer features still receive retrieval gradients. `joint`
keeps the complete path differentiable. Candidate-pair and residue masks are
applied before pooling, accumulation is fp32 under autocast, and invalid
residues receive exactly zero weight. The objective accepts the direct
complex-level cosine through the explicit source name
`contact_bottleneck_cosine`; it never top-K pools or sigmoid-calibrates that
cosine as if it were a pair logit.

The corrected working tree passes `298/298` locally and adds an explicit
unary-shortcut rejection contract; its source authority is frozen separately
from the superseded v14 engineering sketch. The suite covers mask isolation,
both gradient modes, independent antibody/antigen SE(3) transforms, absent-score
failure, default-off state-dict compatibility, and activation-checkpoint
recomputation of the new score. This is engineering evidence only. No protocol
registers the candidate, no GPU development job has been launched, and no
quality metric or improvement is claimed.

Evaluation also records attribution concentration without choosing a threshold:
zero-lift complex count, positive-weight fraction, normalized entropy, effective
support, and top-1 mass fraction for antibody residues, antigen residues, and
pair contact lift. These diagnostics expose two failure modes that retrieval
alone can hide: a dead bottleneck with zero lift, and a diffuse bottleneck that
collapses toward whole-chain pooling. They are descriptive development outputs;
their thresholds are not tuned on frozen or external test data.

The present executable objective has one authenticated cognate per focal
example and multiple deterministic wrong partners in each reciprocal direction.
It is therefore a reciprocal listwise contrastive MVP, **not yet** a true
multi-positive InfoNCE implementation. True multi-positive training requires a
separate frozen authority defining which complexes are valid positives without
using validation/test labels; until that exists, describing the code as
multi-positive would be an overclaim.

## Fair contact reporting

The primary contact definition remains the preregistered 4.5 A axis. To make
PEPNet, DeepSSInter, ProTact and other 6 A work interpretable, report 4.5, 6 and
8 A contact axes from the same predicted distance distribution. These are
predeclared reporting axes, never a menu from which the best test cutoff is
selected. Candidate pair scope, missing-residue masks, chain scope and cropping
must remain identical.

## Execution order

1. Finish the running v3.1 development arrays and freeze pair factorization and
   residue aggregation on validation only.
2. Retrain the selected model, the same-code geometry-ablated control and the
   existing partner-aware sequence control on the double-cold split.
3. Run the contact-bottleneck contrastive candidate only if the base contact
   model passes localization and calibration gates.
4. Add SPPIDER-seq as a separate-environment residue control and PEPNet only
   behind the AsEP opening gate.
5. Use AAMFM/CHIMERA as the downstream co-design stage after specificity is
   demonstrated, not as evidence for the contact model itself.
