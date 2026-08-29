# Frontier contact-baseline audit — 2026-08-01

## Outcome

Four 2026 authorities materially change the baseline plan, but none currently
authorizes a head-to-head performance claim on FrameContact's task:

| Authority | What it tests | Use now | Blocking difference |
|---|---|---|---|
| [Handshake](https://www.biorxiv.org/content/10.64898/2026.06.04.730144v1) | structure-pretrained ProstT5 plus partner-aware cross-attention and contact supervision | clean-room sequence-only control | upstream source has no license/checkpoint; split and inference procedures are not our double-cold frozen protocol |
| [ProstT5](https://huggingface.co/Rostlab/ProstT5) | whether structure-aware pretraining, without input coordinates, closes the geometry gap | frozen representation factorial after weight custody | current official weight is an approximately 11.3 GB pickle artifact; cutoff/overlap and safe conversion are unverified |
| [DeepSSInter](https://doi.org/10.1002/pro.70667) | ESM2+SaProt, monomer graphs/distances and a triangle-aware contact trunk | external inference control first; retrain only after license/data audit | 6 A contact labels, upstream data/crop policy and ambiguous repository license differ |
| [ProTact](https://www.nature.com/articles/s42003-026-09809-2) | invariant monomer encoding, surface complementarity and trigonometric pair updates | independent design comparator | 6 A labels, 256-residue label-informed crops, different data, and no source license |

The immutable machine-readable authority is
[`BASELINE_LOCK_V5.json`](../baselines/BASELINE_LOCK_V5.json). Published scores
remain literature context only.

## Pinned source evidence

- Handshake source: `nurith/Handshake` at
  `cfb10ffcf0d5745473fe1deb3a6e849b9f5986f4`.
- ProstT5 model: `Rostlab/ProstT5` at
  `d7d097d5bf9a993ab8f68488b4681d6ca70db9e5`; config SHA-256
  `69c1915585d20b1d3c1d44a566968a27c7d9cf784f223569ffc5dc2a9e0f892c`.
- DeepSSInter source: `huang-laboratory/DeepSSInter` at
  `d24e9e02b35c857a553e5a9d7d7def51c633d166`; included checkpoint SHA-256
  `7ea97d065210c0fa4f6ceaed2c6f64caaba027bf661684affb3261b96103765d`.
- ProTact source: `biomed-AI/ProTact` at
  `ed0d75e2adf159915e5a1d2fb4ead73f8d278702`.

Source and checkpoint authentication proves identity only. It does not prove
that preprocessing works, that a checkpoint is safe to deserialize, or that a
model is comparable on antibody–antigen data.

## Design consequences for FrameContact

### 1. Separate representation from geometry

The geometry claim needs a two-factor comparison, not only ESM versus the full
model:

1. frozen ESM2 + geometry-ablated partner-aware contact model;
2. frozen ESM2 + FrameContact geometry;
3. frozen ProstT5 + geometry-ablated partner-aware contact model;
4. frozen ProstT5 + FrameContact geometry.

This distinguishes explicit input geometry from structural knowledge already
stored in a protein language model. The ProstT5 factor cannot enter the frozen
matrix until the revision, tokenizer, weight hash, safe converted artifact,
embedding cache and pretraining-overlap boundary are fixed.

### 2. Triangle consistency is not yet a triangle pair trunk

The current FrameContact triangle loss regularizes predicted distances inside
CDR/surface patches. DeepSSInter and ProTact instead contain learned triangle or
trigonometric interaction updates. These are distinct interventions. If the
current loss passes validation, the next equal-budget ablation should compare:

- no triangle mechanism;
- triangle consistency loss only;
- sparse learned row/column triangle updates only;
- both mechanisms.

The sparse update must use prediction-routed antigen patches and soft CDR bias,
never true epitope labels, at validation or test time.

### 3. Harmonize the contact task before measuring a winner

DeepSSInter and ProTact use minimum-heavy-atom distance below 6 A. The running
FrameContact authority uses its frozen 4.5 A pair definition, while residue
epitope/paratope labels and many antibody papers use other cutoffs. A fair
comparison must either retrain every model on one contact label or pre-register
parallel 4.5/6/8 A evaluation axes without selecting among them on test. Pair
candidate scope, missing residues, chain scope, cropping and monomer source must
also match.

### 4. Keep localization and retrieval as separate gates

The frontier contact models are optimized mainly for P@K or contact maps.
FrameContact additionally claims partner specificity. Promotion therefore
requires both:

- localization: contact macro/pooled AUPRC, P@10, P@Lmin/10, P@Lmin/5,
  precision and recall;
- specificity: both retrieval directions, worst-direction margin, R@1/R@5/MRR,
  and directed same-antigen/different-antibody epitope migration.

A model that improves P@K while ignoring partner identity, or improves
retrieval while degrading localization/calibration beyond the frozen gate, is
not promoted.

## Execution priority

1. Finish the current v3.1 development arrays and freeze binary-versus-
   distogram residue aggregation on validation only.
2. Retrain the winner and its same-code geometry-ablated control on the strict
   double-cold split.
3. Run the clean ESM representation factorial first; it is the cheapest causal
   geometry test.
4. Materialize and audit ProstT5 only after the preceding gate passes.
5. Add DeepSSInter as a clearly labeled external-pretrained inference control;
   retraining or code reuse requires resolving its license and preprocessing.
6. Treat ProTact as prior art/design evidence until a licensed, reproducible
   identical-task adapter exists.

This order maximizes information per GPU-hour and prevents a large baseline
matrix from consuming the one-shot frozen-test budget.
