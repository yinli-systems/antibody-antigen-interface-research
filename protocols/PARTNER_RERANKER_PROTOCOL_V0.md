# Pair-aware reranker and contact-head protocol v0

Status: research preregistration draft. It does not authorize training, fixed-boundary
evaluation, or access to the frozen SAbDab2 test. It may be locked only after the
authority-train-only S2 folds pass an independent exact four-view Needleman--Wunsch
audit.

Date: 2026-07-23 CST

## Question and scope

Can a lightweight pair-aware model improve cognate antibody--antigen retrieval because
it uses the supplied partner, rather than because it memorizes marginal antibody or
antigen propensity?

The primary task remains bidirectional full-library retrieval. Contact prediction is an
auxiliary mechanistic task and may not replace retrieval evidence. Unobserved SAbDab
pairs are non-cognate decoys, not experimentally verified non-binders.

## Evidence motivating the design

- [CALM](https://www.biorxiv.org/content/10.64898/2026.02.25.707916v1) motivates a
  symmetric antibody--antigen retrieval objective and establishes an 80%-identity
  leakage-controlled reference setting.
- [DataSAIL](https://www.nature.com/articles/s41467-025-58606-8) requires separate
  similarity-aware assignments on both entity axes for an S2 paired-input split and
  explicitly discards interactions whose endpoints receive different folds.
- [EpiCLIP](https://openreview.net/pdf?id=JImubudlk9) motivates focusing contrastive
  learning on approximate antibody and antigen interface regions rather than relying
  only on whole-sequence averages.
- [PPLM](https://www.nature.com/articles/s41467-026-70457-5) and
  [PLM-interact](https://www.nature.com/articles/s41467-025-64512-w) motivate a
  pair-aware joint representation and residue-contact output without requiring the
  first experiment to fine-tune a full protein language model.
- [ConTact](https://arxiv.org/abs/2605.21600) separates contact localization from the
  downstream prediction head and shows why a positional marginal can look strong while
  remaining weakly antigen-conditioned.
- Experimental-negative research in
  [Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01089-5)
  warns that negative composition changes both generalization and learned biological
  rules. This protocol therefore uses implicit retrieval decoys for model selection and
  reserves binder/non-binder classification for external measured data.

## Data firewall and selection boundary

- Architecture, loss family, width/depth limits, hard-negative rule, and promotion gate
  are fixed before any new fixed-boundary score is read.
- All model selection occurs on authority-train-only S2 folds. Antibody clusters and
  antigen clusters are assigned independently; only same-fold interactions are
  retained.
- The fold builder must publish every discarded exact pair and source row, solver
  objective, solver status, input and source SHA-256, and a canonical fold relabeling.
- Concatenated antibody, VH, VL, and antigen views must have zero crossing edges under
  the locked identity/reciprocal-coverage definition after exhaustive exact global
  alignment. Approximate MMseqs2 closure alone is insufficient.
- Authority validation remains confirmation-only. Frozen test and external labels stay
  inaccessible throughout architecture selection.
- Frozen encoder pretraining exposure is unavoidable and must remain an explicit claim
  limitation. A raw same-encoder control is required wherever coordinates are
  comparable.

## Candidate family

Only the following ordered family may be compared in the first S2 experiment:

1. `dual_encoder_locked`: the already selected AntiBERTy antibody / ESM-2 antigen
   asymmetric dual encoder, with the fixed refit contract reproduced inside each S2
   fold.
2. `joint_reranker`: the same coarse retriever plus a small bidirectional cross-attention
   reranker operating on frozen residue embeddings for a preregistered top-K candidate
   set. The pair score is added to, rather than substituted for, the authenticated
   coarse score.
3. `joint_reranker_contact`: the identical reranker with an auxiliary native
   residue-contact matrix head. Contact labels come only from authorized training
   complexes; missing or ambiguous residues are masked, never imputed as negatives.

No fourth architecture, encoder fine-tuning variant, structure-prediction input, wider
head, deeper head, or alternative top-K may be introduced after fold-level validation
metrics are opened. A future change requires a new protocol and new train-only folds.

## Model constraints

- Protein language-model encoders remain frozen in the first comparison.
- Joint capacity is deliberately small: at most two cross-attention blocks, model width
  at most 256, at most eight heads, and dropout fixed before the first S2 run.
- Long antigens use a deterministic, score-independent chunking and masking rule. No
  validation score may select a truncation point or surface patch.
- The contact head emits an antibody-residue by antigen-residue logit matrix. Its
  marginals may be reported, but a pair matrix is the native output.
- The global pair score must be a declared, deterministic aggregation of the joint
  representation and/or contact logits. It cannot use candidate identity, family size,
  PDB identifier, library position, or source split as a feature.

## Loss and decoys

- Use symmetric multi-positive contrastive retrieval loss. Every observed cognate in
  the authorized positive graph is positive; duplicate exact pairs are collapsed.
- Mask known positives and declared high-homology ambiguous pairs from denominators.
- Mine hard candidates with one frozen, train-fold-only coarse index. The index and its
  checkpoint are hashed before reranker fitting and never refreshed from validation
  behavior.
- Hard candidates are called decoys. They may train a ranking objective but may not be
  presented as non-binders.
- For the contact variant, use a preregistered weighted contact loss with all missing
  coordinates masked. The contact-loss weight is chosen from a finite train-only grid
  declared before S2 scores are read.
- Do not add partner dropout merely because it improves a fold score. Single-side masks
  and partner permutations are required falsification tests first; a future training
  intervention requires a new locked protocol.

## Selection and stability

- Screen the three candidates at seed 42 only.
- Primary selection statistic: unweighted macro mean of fold-level bidirectional MRR.
- Tie breakers, in order: worst-fold bidirectional MRR, paired partner-permutation
  excess-over-chance collapse, fewer trainable parameters.
- A candidate advances only if its paired improvement over `dual_encoder_locked` has a
  two-sided 95% joint antibody--antigen cluster-bootstrap interval excluding zero and
  the sign is positive in every fold.
- After selection, repeat only the winner at seeds 17 and 73. Report all three seeds,
  their mean, sample standard deviation, and worst seed. Never report the best seed as
  the representative result.

## Required endpoints and controls

- Full-library Ab-to-Ag and Ag-to-Ab MRR, normalized mean/median rank, and Recall/Hit at
  1, 5, 10, and 20.
- Paired difference from exact finite-library chance and from the locked dual encoder.
- Partner permutation, composition-preserving residue shuffle, antibody-only mask, and
  antigen-only mask effects with paired intervals.
- Raw embedding, random untrained head, antibody-only unary, antigen-only unary, and
  length/composition controls under the same candidate order and positive graph.
- Native contact AUROC and AUPRC, per-complex macro contact AUPRC, and contact-map
  permutation controls for the contact variant. Contact prevalence must be reported.
- Performance stratified by both-axis nearest-train identity, antigen family/component,
  antibody family size, sequence length, missing VL, and number of positives.

## Fixed-boundary promotion gate

Only a three-seed S2-confirmed winner may be evaluated on the two corrected
authority-validation boundaries, once each, from one checkpoint family. It passes Gate
B only if:

1. both retrieval directions improve over the authenticated dual-encoder baseline on
   both boundaries;
2. the learned pair effect beats all five controls with a 95% paired lower bound above
   zero;
3. partner permutation removes at least half of the excess over exact chance and its
   paired interval excludes no change;
4. improvement is not confined to the highest-identity stratum; and
5. the contact variant, if selected, exceeds contact prevalence materially and loses
   that excess when the partner or contact map is permuted.

Failure is retained as a scientific result. Fixed-boundary failure cannot be repaired
by changing the architecture, top-K, seed, loss weight, or checkpoint epoch.

## External claim boundary

Passing Gate B supports pair-aware development retrieval and, for the contact model,
native residue-pair interface prediction on SAbDab-derived structures. It does not prove
experimental binding, affinity, therapeutic specificity, or prospective design. Those
claims require the separately frozen AVIDa/REAP/experimental-affinity protocol and,
ultimately, prospective wet-lab validation.
