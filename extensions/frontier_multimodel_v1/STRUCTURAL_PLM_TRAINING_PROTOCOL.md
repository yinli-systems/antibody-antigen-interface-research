# SaProt / ProSST downstream layer-sweep training protocol v1

Frozen 2026-09-30 before any downstream SaProt/ProSST training metric was read.
This protocol is additive to `STRUCTURAL_PLM_LAYER_SWEEP_PROTOCOL.md` and does
not amend the permanently closed representation-paper test sets.

## Question and primary endpoint

Do genuine structure-token protein language model features improve partner-aware
residue-pair contact prediction relative to an exact-width random projection of
the same ESM-2 base input, and which predeclared layer transfers best?

Primary endpoint: validation macro mean-per-complex pair average precision (AP)
from the existing complete candidate residue-pair grid. Secondary endpoints are
pooled pair AP/AUROC, epitope and paratope macro AP, outer-product margin,
partner-ranking diagnostics, training stability and per-component heterogeneity.
No partner-ranking endpoint may replace the primary endpoint after results are
opened. Validation is development selection, not blind confirmation.

## Frozen feature authority

The only eligible feature authority is the completed audit under
`structural_plm_full_v2`: 33,450 of 33,450 expected artifacts, 3,345 unique
side/sequence units, and no contact labels loaded during extraction.

- SaProt-650M receives genuine Foldseek amino-acid + 3Di tokens. Layers:
  `{0, 8, 16, 24, 33}`, width 1280.
- ProSST-2048 receives tokens from the official 2048-state GVP quantizer.
  Layers: `{0, 3, 6, 9, 12}`, width 768.

Each layer has two immutable generic source specifications, one for antibody and
one for antigen. Artifact model name, model revision, structure authority,
model weight, tokenizer and source-specification hashes are verified before
training. Missing admitted structures remain zero-filled with the existing
presence bit; no sequence is dropped or synthesized.

## Frozen task grid and exact controls

Real arms: ten layers above, seeds `{1,2,3}`. Controls: one exact-width and
exact-coverage random-projection arm for SaProt width 1280, and one for ProSST
width 768, also seeds `{1,2,3}`. Total formal tasks: 36.

A control is built from the same pair of antibody/antigen source specifications
as the corresponding real family. It therefore has the same appended width,
coverage and downstream parameter count. Its fixed projection seed is 20260930.
The random projection contains no information not already present in ESM-2.

## Downstream parity contract

All tasks use the historical SAbDab2 interaction-first development authority:
3,229 training complexes and 217 validation complexes; the same strict split,
base ESM-2 store, candidate contact store, architecture and objective.

- d_model 128, pair_dim 64, one cross-attention layer, eight heads, dropout 0.2.
- Bilinear pair decoder; log-mean-exp marginal and complex-score pooling.
- Four partner negatives in both directions, hard pool 64.
- Edge weight 1.0, node BCE/Dice weights 0.25 each, partner-rank weight 0.5.
- AdamW, learning rate 5e-5, weight decay 5e-4, gradient clipping 1.0.
- Maximum 12 epochs, ReduceLROnPlateau factor 0.5/patience 3, early-stopping
  patience 10, and the historical partner-rank checkpoint rule.

The partner-rank checkpoint rule is retained only for direct historical parity.
Because the preceding score-dissociation experiment found that partner ranking
and contact AP can move in opposite directions, both validation-best and final
epoch checkpoints are evaluated. Only validation-best is the registered primary
checkpoint; final epoch is a diagnostic and cannot replace it after inspection.

## Selection and uncertainty

Within each model family, the development-selected layer is the registered layer
with the highest three-seed mean primary AP. If two layer means differ by at most
0.001 absolute AP, the shallower layer is selected. Every layer remains reported.

Each real layer is compared with its same-width control using paired training
seeds. Complex-level effects use equal-component weighting: AP is computed per
complex, averaged within authority component and seed, then a crossed
seed/component bootstrap is reported. With only three training seeds, seed-level
intervals alone are not treated as evidence of biological generalisation.
All intervals are pointwise and exploratory; no multiple-comparison-adjusted
superiority claim is authorized by this sweep.

## Execution gates

1. Source, input, split, coverage-audit and source-specification hashes freeze.
2. Targeted generic-source/control tests pass from the exact source snapshot.
3. One engineering smoke: SaProt layer 16, seed 1, one epoch, at most 48 train
   and 24 validation instances. Smoke metrics are never scientific results.
4. Smoke must verify real source identity, expected per-side width 2561, finite
   loss/evaluation output, immutable input hashes and no test access.
5. Formal 36-task grid runs in fresh exclusive directories. Failed jobs remain;
   a retry requires a new attempt directory and receipt.
6. Aggregation requires all 36 terminal receipts and all 217 validation IDs per
   checkpoint. Missing tasks, cases or seeds fail closed rather than reducing
   the denominator.

## Explicit exclusions

No test or rejected OOD label is read. No fine-tuning of SaProt/ProSST weights,
new architecture search, affinity endpoint, RePaRank comparison, co-folding
comparison or learned realignment occurs in this stage. Extraction completion is
not performance completion; downstream smoke is not a completed layer sweep.
No SOTA, blind, first-place or production claim is authorized.
