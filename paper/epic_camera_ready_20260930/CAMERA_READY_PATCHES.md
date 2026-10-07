# Camera-ready textual patches

These replacements are designed for the accepted EPIC manuscript source. The source `.tex` was not present in the connected repository/GPU snapshot, so this file provides exact edits rather than pretending that the uploaded PDF was safely editable source.

## Title and footer

- Keep: `EPIC: Epitope–Paratope Interleaved Contact Learning for Antibody–Antigen Interface Prediction`.
- Replace the NeurIPS main-track footer with the AI4DD workshop camera-ready footer.
- Remove `Do not distribute`.

## Abstract replacement

> Predicting antibody–antigen interfaces from sequence inputs is difficult because residue–residue contacts are extremely sparse, partner-specific, and coupled to residue-level paratope and epitope labels. We present EPIC, a multi-task framework that combines frozen protein-language-model representations with intra-chain geometry from predicted monomer structures. EPIC uses geometry-aware self-attention, bidirectional cross-chain communication, and a shared low-rank residue-pair representation to predict contacts, paratopes, and epitopes. We evaluate EPIC on the official SAbDab2 split. Because contact positives constitute only 0.0705% of training candidate pairs, we emphasize average precision and practical ranking budgets rather than AUROC alone. EPIC provides a unified prediction profile across the three interface resolutions without requiring an experimentally determined antibody–antigen complex at inference. The results support EPIC as a sequence-input, predicted-geometry framework, while additional ablations show that the individual contributions of geometry, task coupling, and pairwise decoding are heterogeneous and should not be interpreted as universal improvements.

This removes the unsupported AsEP claim and the inaccurate implication that the method is structure-free.

## Contribution bullets

Replace the current bullets with:

1. A unified contact/paratope/epitope formulation based on a shared residue-pair representation.
2. A sequence-input inference regime augmented by predicted monomer geometry, with no native antibody–antigen complex required.
3. Controlled analyses of sparse-contact ranking, task coupling, geometry sensitivity, and pairwise-versus-marginal evidence.
4. Reproducible reporting that distinguishes same-protocol measurements from literature-context comparisons.

Do not claim that each architectural component is individually novel.

## Data/evaluation paragraph

Add after the contact-label definition:

> The contact task is extremely imbalanced: the training split contains 110,155 positive contacts among 156,171,838 candidate residue pairs (0.0705%). We therefore treat mean per-complex average precision as the primary threshold-free contact-ranking metric. We additionally report pooled prevalence, top-k precision/recall, enrichment over prevalence, Brier score, expected calibration error, and both pooled and equal-complex recall at fixed operating points. AUROC is retained as a secondary discrimination statistic.

## Input-modality clarification

Replace every unqualified `sequence-only` or `sequence-based` description of EPIC with:

> sequence-input with predicted monomer geometry

Add:

> At inference, EPIC requires antibody and antigen sequences and predicted monomer backbones. It does not require the experimentally determined antibody–antigen complex, but it is not a structure-free predictor.

## Baseline table policy

Split the current table into:

- **Same-protocol internal measurements**: only models rerun under the exact split, labels, aggregation and threshold authority.
- **Literature context (not directly comparable)**: imported values with columns for input modality, dataset/split, per-complex vs pooled aggregation, threshold source and citation.

Remove `best reported` language across the two tables.

## Contact-result paragraph replacement

> Contact AUROC is high, but the practically relevant ranking result is more modest under the 0.0705–0.1071% prevalence regime. EPIC's reported contact AUPRC is 0.0761, below the literature-reported 0.0957 value for ImaPEp in Table 1; these values are not treated as a controlled head-to-head comparison because the input modalities and evaluation protocols differ. We therefore report per-complex AP and fixed-budget ranking diagnostics in the supplement rather than interpreting AUROC or recall in isolation.

## Epitope-result paragraph replacement

> EPIC achieves high epitope recall but lower AUROC/AUPRC than MIPE and EpiFormer in the literature-context table. This is a sensitivity–ranking trade-off rather than an across-metric win. Precision, F1, calibration and matched-recall comparisons are reported in the supplement. The conditional-proportion loss may influence the operating point, but we do not assign a causal effect without a matched ablation.

## TOPK2 sentence

Delete:

> Checkpoints for the joint model and the TOPK2 expert are selected separately ...

The submitted paper does not define or report a TOPK2 expert. A later top-k readout study must remain separate.

## Metric harmonization

Replace the current final recall sentence with:

> We report two recall estimands: pooled recall, obtained by aggregating TP and FN over the cohort, and macro recall, obtained by averaging per-complex recalls. Their different weighting is stated explicitly. Primary ranking comparisons use per-complex AP/AUROC; pooled operating-point metrics are reported separately.

## Limitations

Add:

> The structure-predictor sensitivity analyses do not constitute a paired OpenDDE-versus-alternative predictor experiment on an identical frozen checkpoint. External and post-submission evidence is labelled separately. The official SAbDab2 test is historical rather than prospective, and no clinical or experimental validation is claimed.
