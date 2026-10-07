# Frozen parameter-free score correction diagnostic — 2026-09-30

Stage specified before opening ANY transformed contact metric. Baseline RePaRank
AP was already published, so this is explicitly exploratory development, NOT
blind confirmation. It reuses original score NPZ files; no model inference,
retraining, sweep change or additional GPU allocation occurs.

Question: does subtracting residue-local background scores improve exact contact
ranking, or does it erase genuinely useful interface propensities?

Exactly five fixed arms on the original 217 validation complexes:
- identity: S, the stored negative Euclidean distance.
- row_center: S - mean_row(S), removes antigen-row propensity.
- column_center: S - mean_column(S), removes antibody-column propensity.
- double_center: S - mean_row(S) - mean_column(S) + global_mean(S).
- local_scale_10: 2*S - mean_top10_row(S) - mean_top10_column(S), k capped by
  candidate-axis size. This is CSLS-inspired applied to existing negative-distance
  scores, NOT the exact MUSE cosine-embedding protocol or an antibody SOTA model.

Primary contrast: local_scale_10 minus identity in mean-per-complex AP. All four
contrasts are retained. Do not choose k, combine methods, change the candidate
grid, retune thresholds or hide negative results after the readout.

Primary AP is sklearn non-interpolated average_precision_score. Secondary output:
precision at top 10 with fractional boundary-tie handling. Keep every complex and
all original pair labels; duplicate/missing candidates fail. No true-contact-count
or labels are used by the transformations. Do not force one-to-one contacts.

Baseline AP must reproduce the existing receipt to absolute tolerance 1e-7,
otherwise HOLD. For each contrast: paired difference per complex, then equal
weight per original split component; 5,000 fixed-seed component bootstrap draws,
pointwise exploratory intervals only. Also show the equal-complex effect and
positive-component fraction. No training-seed CI: the checkpoint is deterministic.

Every NPZ and identity receipt is content-hashed before the metric read and again
at closure. Source/plan/split hashes are recorded. Other train/test/OOD labels and
the active 36-task sweep outputs are never opened. This diagnostic cannot establish
new biological generalization or remove the checkpoint's unknown training overlap.

Primary-source motivation:
- Conneau et al., Word Translation Without Parallel Data (ICLR 2018),
  https://arxiv.org/abs/1710.04087 ; official MUSE evaluation code:
  https://github.com/facebookresearch/MUSE/blob/main/src/evaluation/word_translation.py .
  Local-neighborhood score correction was introduced to reduce embedding hubs.
- Vorberg et al., Synthetic protein alignments by CCMgen quantify noise in
  residue-residue contact prediction, PLoS Computational Biology 2018,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6237422/ . APC removes a factorized
  background under particular assumptions; those assumptions do NOT guarantee
  improvement for learned antibody-antigen embeddings.

A result is allowed to reject the proposed improvement. No first-place or SOTA
claim, even if a transformed development AP improves.
