# External specificity validation protocol v0

Status: preregistration draft; published aggregate dataset metadata and paper-reported
benchmarks have been reviewed, but no external row-level labels, group frequencies,
model scores, or locally computed results have been inspected.

Date: 2026-07-23

## Scientific question

Does a checkpoint selected exclusively on leakage-controlled SAbDab2 development data
generalize to experimentally measured antibody--antigen specificity, including binding
changes induced by small antigen mutations?

This protocol is deliberately separate from the frozen SAbDab2 benchmark. External data
must not be used to alter the current architecture, final-refit epoch, loss, or SAbDab2
go/no-go gates.

## Locked evaluation order

1. Complete the two preregistered SAbDab2 development holdouts.
2. Complete all SAbDab2 negative controls and seed replication required by the current
   protocol.
3. Freeze one checkpoint family, source tree, and inference configuration by SHA-256.
4. Acquire and hash the external datasets without opening aggregate labels.
5. Generate all frozen embeddings and scores once.
6. Compute the analyses below from immutable score artifacts.

Failure at an earlier stage must be reported. It must not be repaired by tuning on an
external test set.

## Locked transfer mode and score

The current SAbDab2 checkpoint is evaluated **zero-shot**. No AVIDa row, label,
antigen variant, VHH lineage, threshold, calibration parameter, projection weight, or
normalization statistic may be used to fit or select the checkpoint.

For every VHH--antigen pair, the sole trained-model score is the normalized dot product
between the frozen final antibody and antigen projection vectors, using the exact
inference path embedded in the authenticated SAbDab2 checkpoint. VHHs are encoded as
heavy-chain-only antibodies only if the frozen antibody encoder's missing-light-chain
semantics pass a train-only synthetic and provenance audit. No light chain may be
invented or imputed. If that audit fails, the current checkpoint is reported as
unsupported for AVIDa rather than changing the input representation after labels are
opened. The score may be used for ranking and threshold-free discrimination; it is not
interpreted as a calibrated binding probability. Thresholded metrics are descriptive
unless a future, separately locked external calibration partition is introduced.

## Primary external datasets

### AVIDa-hIL6

- Primary paper: https://papers.neurips.cc/paper_files/paper/2023/file/8339dacd9df7ffe9623760f74169dd1e-Paper-Datasets_and_Benchmarks.pdf
- Official repository: https://github.com/cognano/AVIDa-hIL6
- Intended role: mutation-sensitive specificity stress test across wild-type human IL-6
  and 30 alanine-scanning variants.
- Dataset scale reported by the authors: 573,891 labeled VHH--antigen pairs, including
  20,980 binders and 552,911 non-binders.
- License reported by the authors: CC BY-NC 4.0.

### AVIDa-SARS-CoV-2

- Primary paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/d2a1e47f7dc635fac77fbd6e2ec799e4-Paper-Datasets_and_Benchmarks_Track.pdf
- Intended role: second experimentally labeled VHH specificity domain with viral antigen
  variants and a different immunization campaign.
- Dataset scale reported by the authors: 77,003 labeled interactions, including 22,002
  bind and 55,001 non-bind labels across 36,100 unique VHHs. The paper reports 427 VHHs
  whose binding label changes across viral variants; this subset is the strongest
  available partner-specific stress panel.

These datasets contain experimentally defined negatives, which address a central
limitation of SAbDab retrieval: an unobserved structural pairing is not necessarily a
true non-binder.

## Secondary external challenges

### REAP clinical-antibody off-target panel

- Primary paper: https://pubmed.ncbi.nlm.nih.gov/41850278/
- Reported design: 174 approved or clinical-stage antibodies screened against 6,172
  human extracellular proteins; 28% of antibodies had at least one reported off-target
  interaction.
- Intended role: proteome-scale specificity stress test after the AVIDa analysis is
  complete and frozen.
- Admission rule: antibody variable-region sequences, antigen sequences, and the
  quantitative source measurements must be recoverable with an explicit mapping ledger.
  Antibodies or antigens may not be selected or discarded using model scores.

### Synthetic Epitope Atlas

- Primary preprint: https://www.biorxiv.org/content/10.64898/2026.04.17.719295v2.full
- Intended role: affinity-ranked evaluation using experimentally measured binders and
  designed complexes that failed to bind, conditional on row-level public availability
  and licensing.
- Admission rule: predicted structures are provenance or model inputs, not experimental
  labels. No pseudo-structure-derived label may be reported as measured affinity.

Neither secondary challenge is currently authorized for row-level access. Their
published aggregate descriptions only inform this preregistration draft.

## Data firewall

- No raw label table, per-variant label frequency, locally computed model metric, or
  subgroup result may be inspected before checkpoint and inference hashes are locked.
  Aggregate counts and benchmark values already reported in the primary paper or
  official README are literature metadata, not tuning data, and must be cited as such.
- Duplicate rows must be collapsed without changing the binary label. Conflicting labels
  must be quarantined and reported, never resolved using model scores.
- Exact sequence overlap and global-alignment overlap with SAbDab2 must be audited on both
  molecular sides. Results must be reported for the full external set and for an
  overlap-excluded subset.
- VHH inputs must be processed as heavy-chain-only antibodies. No synthetic light chain
  may be imputed, and unsupported missing-light-chain behavior must fail closed.
- Every raw file, normalized table, split manifest, embedding corpus, checkpoint, score
  artifact, and report must carry a SHA-256 digest.

## Primary endpoints

The primary partner-specific analysis unit is a VHH with measurements against multiple
antigen variants, not an individual pair.

1. Within-VHH conditional pairwise accuracy/AUC: for VHHs with a measured label change,
   rank a bind-labeled variant above a non-bind-labeled variant.
2. Mutation sensitivity: paired change in score for the same VHH between wild type and
   each antigen mutant, stratified by the measured label transition.
3. Macro antigen-variant AUPRC and AUROC.
4. Multi-positive MAP, nDCG, mean reciprocal rank, and recall at 1, 5, 10, and 20 within
   each antigen-specific VHH candidate library.
5. Macro antigen-variant MCC, with thresholds selected without using the evaluated
   variant. If no independent calibration set exists, MCC is descriptive only.

Micro-averaged metrics are secondary because the datasets are highly imbalanced and can
be dominated by large antigen groups.

## Required nulls and controls

- Antibody-label permutation within each antigen variant.
- Antigen-variant permutation within each antibody, when the observed label graph permits
  it.
- Length and amino-acid-composition baseline.
- Frozen raw-embedding cosine baseline with no trained projection head.
- Randomly initialized projection-head baseline.
- Unary antibody-only and antigen-only baselines.
- Sequence-similarity nearest-neighbor baseline fitted only on an explicitly authorized
  external training partition; this baseline is omitted for strictly zero-shot tests.
- Easy-negative versus identity-matched hard-negative sensitivity.

Permutation schemes must preserve the relevant positive degree and antigen-group size.

## Uncertainty and multiplicity

- Report 95% percentile and BCa bootstrap intervals where numerically stable.
- Resample antigen variants for the primary macro analysis.
- For pair-level secondary analyses, resample experimentally defined antibody clonotype
  clusters rather than treating all sequences as independent.
- Apply Holm correction to the finite family of primary endpoint hypothesis tests.
- Report effect sizes and raw confidence intervals even when a corrected test is not
  significant.

## External success criteria

The external result is considered supportive only if all conditions hold:

1. Macro AUPRC is above the exact group-specific prevalence baseline with a 95% cluster
   interval whose lower bound is positive for the paired difference.
2. Both retrieval directions exceed their degree-aware permutation 99th percentile.
3. Partner or label permutation removes at least half of the model's excess over the
   exact random baseline.
4. The paired mutation-sensitivity effect has the expected sign and a multiplicity-
   corrected confidence interval excluding zero.
5. The trained model materially exceeds raw-embedding, unary, length/composition, and
   random-head controls.
6. Conclusions remain directionally consistent after excluding external sequences with
   high similarity to SAbDab2.

These criteria are deliberately stringent. A failure is scientifically informative and
must be reported as a limit of generalization, not hidden by pooling datasets or changing
the metric after evaluation.

## Scope limits

- AVIDa VHH data do not establish generalization to paired human heavy/light antibodies.
- IL-6 and SARS-CoV-2 are narrow antigen families and do not cover proteome-scale
  specificity.
- High-throughput AVIDa bind/non-bind labels derive from experimental selection and
  enrichment/depletion rather than an individual SPR or BLI measurement for every row;
  only explicitly reported orthogonal validation may be described as such.
- Protein-language-model pretraining exposure cannot be completely reconstructed; frozen
  raw-embedding controls and overlap audits quantify but do not eliminate this concern.
- Computational validation cannot replace prospective wet-lab confirmation.
