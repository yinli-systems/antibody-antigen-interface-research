# FrameContact-ICG implementation and development-run receipt

Cutoff: 2026-08-04 CST.

## Outcome

The proposed improvement was implemented as an auditable architecture matrix,
not as a larger pair-head MLP. The existing parameter-unshared antibody and
antigen intra-chain encoders remain intact. Partner information is introduced
by simultaneous bidirectional cross-attention after each self-attention layer.
The routing signal is a pre-registered factor:

- `none`: ordinary cross-attention; the constant bias cancels exactly in the
  softmax;
- `full_contact`: the pair score includes antibody and antigen unary terms;
- `interaction_only`: only the jointly dependent pair term controls routing.

The interaction-only condition is the key shortcut-control: partner-independent
surface or CDR propensity may still contribute to the final pair prediction,
but it cannot decide which partner messages cross the chain boundary.

## Contact-first residue output

`mean_probability` computes the masked arithmetic mean of pairwise contact
probabilities and returns its logit. It is length-normalized and deterministic.
M2-M4 retain M1's noisy-OR pooling so the routing experiment does not also
change the residue aggregation rule; M5 is the explicit strict-marginalization
change.

## Joint direct-contact and distance formulation

`JointPairContactDistogramHead` shares normalized low-rank antibody/antigen pair
factors between:

1. a direct Bernoulli contact head, which remains the primary sparse-interface
   prediction; and
2. a categorical distogram, whose registered near-distance bins project to a
   second contact probability.

The two Bernoulli probabilities can be regularized with a symmetric,
per-complex Jensen-Shannon loss. The development matrix includes both
joint-without-consistency and joint-with-consistency arms, so distance
supervision and consistency are not conflated. Bound inter-chain distances are
supervision only for distance arms and never model inputs.

## Frozen development matrix

The protocol is
`protocols/framecontact_icg_exploratory_v1.json`. It contains eight conditions
and five seeds (40 tasks): the authenticated M1 reference, ordinary interleaved,
full-contact routing, interaction-only routing, strict contact marginalization,
joint/no-consistency, joint/consistent, and distogram-only.

Selection uses antigen-fold train/validation roles only. SAbDab2 frozen test and
external test are forbidden until all development choices are frozen. Promotion
requires seed consistency, paired cluster-bootstrap evidence, no
precision/recall collapse, and improved reciprocal partner specificity.

Protocol evidence:

- protocol file SHA-256:
  `d0b0ac1dba8f9dabcc246fe7b1571379649078fb2982a6c81975b101f942f27a`;
- protocol receipt SHA-256:
  `f676fc95a8ff22b57736cc45fb15b785316f5a70b79885dfa9fb7c7d3cdc1508`.

## Validation

- Python syntax compilation passed for the changed model, objective, and
  trainer.
- 81 directly affected tests passed.
- The broader suite previously reached 366 passed and 14 skipped; six unrelated
  environment/artifact tests remain unavailable locally (Biopython, Parquet
  engine, and external receipt/source paths). This is not represented as a
  fully green repository suite.
- The sealed ParaCloud source manifest is
  `8201a5a846fd1567094c252863cd9e3121fc72689fa734dc7dee2888cb1dd162`;
  tree SHA-256 is
  `2209778a5d780ba3ccb93db6dccf92a83d6c6a2cb9f6a7b5ab5d2bf0b19e0616`.

## Live execution boundary

The account had 96 submitted tasks under an association limit of 100, so the
full 40-task array was correctly rejected by Slurm preflight with
`AssocMaxSubmitJobLimit`. Four available slots were used for seed-17 mechanism
checks: M2 ordinary, M3 full gate, M4 interaction-only gate, and M6 joint
consistent. They run as Slurm array `1351883` (indices 5, 10, 15, and 30).
At the last audit all four were running, had authenticated the sealed source,
and had empty stderr. No epoch metric was yet available.

This establishes deployment and training startup only. It is not evidence of
validation improvement, convergence, frozen-test performance, external
generalization, or benchmark superiority.
