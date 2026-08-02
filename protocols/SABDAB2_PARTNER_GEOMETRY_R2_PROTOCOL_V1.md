# SAbDab2 partner-geometry R2 protocol v1

**Status:** implementation authorized; scientific training blocked at Gate 0  
**Machine-readable authority:** `sabdab2_partner_geometry_r2_v1.json`  
**Frozen base commit:** `f4753dde6e752a20274f7f009043b58215a0837c`

## Question

This protocol tests whether explicit antibody-by-antigen residue-pair modeling
learns the cognate partner and its native contact matrix rather than only
learning separable paratope and epitope propensities.

The protocol inherits the Gate 0, data, split, label, runtime, evaluation, and
claim boundaries in `PAIRFORMER_EXECUTION_PROTOCOL_V1.md`,
`pairformer_gate_manifest_v1.json`, and `frontier_model_contract_v1.json`.
It narrows the next candidate; it does not weaken or replace those contracts.

## Current authorization

Source implementation, synthetic tests, and train-only engineering diagnostics
are authorized. Scientific training, inner selection, outer inference, and
external-test access remain blocked until Gate 0 passes.

The external test may not be mounted, read, schema-inspected, or scored during
implementation. After all candidate choices are frozen, each outer fold is
scored once. A final refit and external-test registry are then frozen before
one-shot test inference. Any prior schema access must remain disclosed, so the
eventual result is a provided external test-split evaluation with pre-scoring
schema access, not a pristine blind test.

## Frozen monomer representation

Antibody and antigen use separate, parameter-unshared monomer encoders. The
geometry feature mode is fixed to intra-chain C-alpha distance RBF plus
local-frame C-alpha direction. Bound-complex inter-chain distance, direction,
pose, and labels are forbidden inputs.

The monomer confirmation uses seeds `17`, `42`, `73`, `101`, and `137`, exactly
20 epochs, the final epoch, AdamW at learning rate `3e-4` and weight decay
`1e-4`, batch size four per GPU, and BF16. It has no early stopping or
validation-selected checkpoint. Dense sequence, topology-only, and
distance-only models are parameter-matched controls. This confirmation does
not select a downstream checkpoint or authorize test access.

## Fixed candidate ladder

- **D0:** locked dual monomer encoder, 58 epochs, global batch 1024.
- **R1:** D0 plus two early bidirectional residue cross-attention layers,
  40 epochs, global batch 32.
- **R2:** R1 plus an explicit pair tensor, two alternating row/column pair
  update blocks, and a native contact head, 40 epochs, global batch 32.

Every stage uses its fixed final epoch. Early stopping is forbidden.

The R2 pair tensor is oriented
`[batch, antibody residue, antigen residue, channels]`. Each update first
attends over antigen residues within every antibody row, then over antibody
residues within every antigen column, and then applies a feed-forward update.
Contact, distance, relative-orientation, interaction, confidence, and
uncertainty heads operate on the resulting pair state. Marginal paratope and
epitope predictions must be derived from pair-contact logits.

## Labels, splits, and candidate domain

The primary native contact is any non-hydrogen atom pair at or below
`4.5 Angstrom`; `5`, `6`, and `8 Angstrom` are sensitivities. A negative or an
exact distance requires complete evidence for both residues.

The assignment is a strict dual-axis nested SAbDab2 `abag_split`: three outer
folds and three dependent inner folds per outer fold. All antibody
concatenated, VH, VL, and antigen sequence boundaries are audited exactly.
After assignment freeze, Foldseek interface coverage and iAlign provide
independent structure-leakage checks; the union of their hits fails closed.

Architecture, loss, negative policy, epoch, checkpoint, calibration, and
threshold choices use the nine inner folds only. Evaluation uses the complete
frozen Cartesian residue-pair domain without negative subsampling.

## Required controls and endpoints

Controls are matched-partner shuffle, antibody-only, antigen-only, null,
composition-only, strongest separable marginal, degree-aware random,
topology-only, and raw frozen embedding.

The primary endpoint is per-complex macro native pair-matrix AUPRC. Pair AUROC,
MCC, Dice/F1, Precision@K, Recall@K, NLL, Brier, adaptive ECE, and
risk-coverage are secondary. Epitope and paratope metrics are secondary.

Both antibody-to-antigen and antigen-to-antibody retrieval report Recall@1,
Recall@5, Recall@10, MRR, normalized rank, and pairwise win rate over the
complete frozen fold library. Inference uses 10,000 paired cluster-bootstrap
replicates and fails closed below ten independent clusters.

## Gate 1

Gate 1 passes only if every condition holds:

1. The paired cluster-bootstrap 95% confidence-interval lower bound for pair
   AUPRC improvement over the strongest separable baseline is above zero.
2. Matched-partner shuffle significantly degrades pair AUPRC or cognate score.
3. Both retrieval directions exceed degree-aware permutation and D0.
4. Unary, null, composition, and similarity controls do not explain the gain.
5. At least four of five seeds are positive in every primary specificity cell.
6. Major antigen strata preserve the direction.

If Gate 1 fails, the permitted response is to improve labels, masks, partner
supervision, or hard negatives. Parameter scaling is not permitted.

## Baseline and claim boundary

SPPIDER-seq and an EpiFormer/PInet-class baseline must be reproduced on the
same admitted complexes, split, input regime, contact definition, candidate
domain, metric implementation, aggregation, seed, checkpoint, calibration,
and uncertainty contract. Missing code, checkpoints, license, or compatible
inputs must be reported rather than silently approximated.

No result is called state of the art or world-leading without identical
benchmark evidence. A larger model is not itself a scientific result.
