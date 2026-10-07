# Label-free gated rescue protocol v1 — frozen 2026-09-30

## Motivation

The completed three-seed development evaluation shows replicated synthetic-corruption recovery but also replicated clean-input damage. An unconditional rescue transform is therefore not promotable. This extension asks whether the matcher can **abstain** and leave already-aligned inputs unchanged.

## Gate signal

For each biological chain, compute the trained matcher's square logit matrix on the observed extra features. Define

`assignment_gain = (maximum one-to-one assignment logit sum - identity-diagonal logit sum) / residue_count`.

This uses no contact labels and no hidden corruption permutation. A complex-level gate uses the maximum score across its present antibody/antigen sides. Missing-feature sides never trigger rescue.

## Calibration

For each rescue seed independently:

1. Use only frozen **training** sequence keys with complete ESM-IF1 features; no validation contact labels, test or OOD data.
2. Select up to 256 calibration keys by deterministic SHA-256 ordering.
3. Compute clean assignment-gain scores and the three already-registered synthetic corruptions.
4. Freeze the per-seed threshold at the empirical 99.5th percentile of clean sequence-level scores (`method=\"higher\"`). This targets a conservative clean-side trigger rate; it is not a formal deployment guarantee.
5. Record clean false-trigger rate and corruption trigger rates on calibration keys before opening development contact effects.

No threshold is selected from contact AP, clean AP damage, test/OOD labels, or the 217-complex development contact outcomes.

## Development evaluation

Reuse the already completed 1953 AP records; do **not** rerun the downstream contact model. For each seed/complex/condition, compute only the label-free gate score. If the corrupted complex triggers, use its already-recorded rescued AP; otherwise use corrupted AP. If the clean complex triggers, use its already-recorded clean-rescued AP; otherwise preserve aligned AP.

Primary descriptive endpoints are equal-component/equal-paired-seed gated-rescue minus corrupted AP and gated-clean minus aligned AP, with the same 68-component crossed bootstrap as the completed rescue qualification. Also report trigger coverage, each seed separately, and identity/never-rescue plus unconditional-rescue controls.

## Promotion boundary

This is still development-only synthetic-corruption evidence. It cannot establish OOD robustness, biological misregistration prevalence, production safety, or SOTA. No clean noninferiority margin is retrofitted. A useful gate should reduce clean damage while retaining a substantial fraction of the registered rescue effect; the complete risk/coverage trade-off is reported rather than selecting a post-hoc favourable point.
