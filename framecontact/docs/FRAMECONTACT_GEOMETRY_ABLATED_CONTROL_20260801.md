# Partner-aware geometry-ablated controls — 2026-08-01

## Purpose

The original `sequence_dense` condition is a valid single-chain geometry
ablation, but it disables cross-attention and cannot test partner-specific
interaction. A strong causal comparison for FrameContact must keep the pair
head, reciprocal specificity loss, cross-attention, residue aggregation and
parameter topology while removing only geometric information from the input
path.

Two future-only conditions are now registered:

- `framecontact_geometry_ablated_contact_hybrid`;
- `framecontact_geometry_ablated_distogram`.

Both retain the full-pose modules and parameter shapes but freeze each
intra-chain geometry residual gate at exactly zero. The binary or distogram pair
head, gated-hybrid residue head, noisy-OR marginal and late bidirectional
contact-gated cross-attention remain trainable. This makes model outputs
coordinate-independent without turning the task into two independent unary
classifiers.

The data loader still reads authenticated structures because real contact and
distance supervision, residue identity alignment and missing-residue auditing
come from those structures. “Geometry-ablated input control” is therefore the
precise label; “structure-free training” would be false. At deployment the
learned forward function does not use coordinate values when the frozen zero
gate is enforced.

## Evidence

Contract tests establish that:

1. both conditions retain cross-attention and the requested pair head;
2. the geometry path uses the same full-pose module topology as FrameContact;
3. geometry gates are frozen at zero;
4. after cross-attention and contact-to-residue gates are activated, shuffling
   valid coordinates leaves contact, interaction, epitope and paratope logits
   bitwise unchanged;
5. swapping the antigen sequence features changes partner-aware contact and
   epitope outputs.

These are architecture tests, not model-quality evidence.

## Frozen comparison design

After the current residue-head/distogram development selection is complete,
freeze exactly one matching pair:

- selected geometry-aware binary or distogram model;
- its geometry-ablated counterpart with every other configuration identical.

Train five paired seeds on the double-cold train role, select checkpoints only
on double-cold validation, and report paired deltas for contact, epitope,
paratope, calibration and reciprocal retrieval. The comparison must include
geometry-shuffle and coordinate-perturbation interventions; the ablated control
should be exactly invariant, while a promoted geometry model must lose useful
signal under targeted corruption without failing global SE(3) invariance.

Do not add these conditions to the already-running v3.1 arrays and do not open
any frozen test to decide whether to run or promote them.
