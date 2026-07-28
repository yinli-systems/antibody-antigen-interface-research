# PAIRScale E1/E2 Unary-Freeze Protocol V2

## Why this revision exists

The first E1/E2 implementation passed its real-cache GPU smoke but failed a
train-only decomposition diagnostic before any development-holdout access.
Across seeds 17, 42, and 73, additive unary macro AUPRC was approximately
0.208--0.211 at epoch 20 and only 0.019--0.023 at epoch 40. Pair AUPRC rose,
but that apparent pair-minus-unary gain was confounded by collapse of the
reference unary path.

The mechanism was implementation-specific. Phase 2 detached the unary logits
inside the contact loss and prevented specificity gradients from entering the
two unary heads. It did not freeze the projections and final normalization
that produce the monomer representations. Pair and specificity losses could
therefore move the shared monomer representation underneath fixed unary heads.

Jobs 1334012, 1334013, and 1334014 were cancelled at recorded epoch 47. Their
authority-train metrics and epoch-40 checkpoints are retained. No holdout
access receipt, holdout evaluation, or final report was created.

## Exact phase ownership

Phase 1, epochs 1--20, trains only the complete unary path:

- antibody projection;
- antigen projection;
- antibody final normalization;
- antigen final normalization;
- antibody unary head; and
- antigen unary head.

Phase 2, epochs 21--220, freezes every parameter above and trains only:

- both interleaved bidirectional fusion layers;
- antibody and antigen pair-add projections;
- antibody and antigen pair-gate projections;
- pair refinement;
- contact interaction-residual head; and
- independent specificity-residual head.

The distance and orientation heads are outside the E1/E2 objective and remain
frozen in every phase. Phase 3, epochs 221--260, unfreezes both unary and pair
paths and applies the already locked 0.1 learning-rate multiplier. It does not
unfreeze the unused geometry heads.

Every model parameter must belong to exactly one of the unary, pair, or unused
groups. Unknown or multiply classified parameters fail closed.

## Required integrity evidence

Before a production restart, unit tests and a real-cache GPU smoke must prove:

1. only unary parameters change during phase 1;
2. the full unary-path state and unary logits are bitwise identical before and
   after phase 2;
3. pair and specificity parameters change during phase 2;
4. unary parameters may change only after the low-rate phase-3 transition;
5. unused distance and orientation heads never change;
6. checkpoint replay remains exact for model and optimizer state; and
7. every epoch records trainable and frozen parameter counts.

The epoch-20 and epoch-220 checkpoints provide the production byte-level
unary-state audit. A non-identical hash permanently invalidates the run before
any scientific gate is interpreted.

## Claim boundary

This revision uses only authority-train diagnostics to repair training
semantics. It does not authorize checkpoint selection, scaling, confirmatory
evaluation, comparative superiority, or a state-of-the-art claim. The reused
192/64 assignment remains development data, not an untouched final test. The
separately frozen E1/E2 Gate Audit Protocol remains unchanged.

