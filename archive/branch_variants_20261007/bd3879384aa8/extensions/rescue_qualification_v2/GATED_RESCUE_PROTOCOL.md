# Train-only confidence gate for correspondence rescue

This experiment addresses the replicated clean-input cost of always applying the learned correspondence matcher. It does **not** retrain the matcher and does not use test/OOD data.

For each rescue seed independently, a gate is calibrated from the frozen **training sequence keys only**. For an input feature bank, define assignment gain as the mean per-residue increase in learned matching score of the maximum-weight one-to-one assignment over the identity assignment, summed independently within biological chains. The gate threshold is the deterministic empirical 95th percentile of the gain on *clean aligned training inputs*. Contact, epitope, paratope, partner or affinity labels are never read during calibration. This fixes an approximately 5% training-clean activation budget before validation contact metrics are opened.

The same procedure is applied to a seeded **untrained matcher** as a no-training control. Synthetic same-AA, shift-8 and global permutations may be evaluated during calibration only to report gate sensitivity; they do not change the threshold.

After all three thresholds are frozen and hash-bound, the 217 development-validation complexes are evaluated once per paired rescue/downstream seed. Report aligned, corrupted, always-rescued, gated-rescued, clean-gated, untrained-gated and activation rates. Missing ESM-IF1 sides preserve historical zero-feature/zero-presence semantics and never invoke a matcher. Final inference is paired by the frozen 68 split-authority components and three paired seeds. No post-hoc noninferiority margin is introduced.
