# FrameContact primary protocol v1

This protocol was frozen before any FrameContact pair-contact training or any
new antigen-fold test access. The machine-readable authority is
`protocols/framecontact_primary_v1.json`; any future amendment must create a
new version and preserve v1 unchanged.

## One-sentence claim

On antigen folds excluded by reciprocal structure-alignment audit, intra-chain
local-frame geometry plus an explicit contact-gated interaction path should
score the cognate antibody-antigen pair above a metadata-matched non-cognate
panel. A residue-classification gain by itself does not establish this claim.

## Why the primary metric is partner retrieval

Contact AUPRC is measured only on cognate complexes because a wrong pairing has
no aligned ground-truth contact matrix. Giving every wrong pair an all-zero
matrix would fabricate negatives. The primary endpoint is therefore the
cluster-weighted difference between a length-normalized predicted-contact score
for the cognate partner and the mean score over 31 frozen wrong partners.
Residue-pair contact AUPRC remains a required secondary endpoint.

This design follows the biological distinction exposed by antibody-antigen
cross-docking: recognizing a plausible interface is not the same as selecting
the cognate partner. It also keeps the intended interaction claim stronger than
paratope localization, which can be solved partly from CDR priors.

## Structural novelty

Foldseek is used for the exhaustive screen because it converts local tertiary
interactions into the 3Di alphabet and makes structure search fast enough for
all-versus-all auditing. Reciprocal query- and target-normalized TM scores are
retained, and a maximum reciprocal TM score of 0.50 creates a hard clustering
edge. Scores near the boundary are spot-checked with TM-align or US-align.

The split fails closed if any hard edge, exact antigen sequence, exact focal
antibody sequence, or PDB biological assembly crosses roles. A secondary
double-cold split clusters both antibody and antigen.

References:

- [Foldseek primary paper](https://www.nature.com/articles/s41587-023-01773-0)
- [Foldseek-Multimer primary paper](https://www.nature.com/articles/s41592-025-02593-7)
- [Antibody-antigen cross-docking specificity analysis](https://www.nature.com/articles/s41598-017-08414-y)

## Evidence boundary

The existing SAbDab2 self-attention experiments informed the design but are
exploratory. Existing CHIMERA residue-classification predictions have already
been accessed and are secondary only. CHIMERA-BENCH defines the canonical task
as epitope-conditioned CDR sequence-structure co-design, so the final project
must evaluate that task directly rather than relabeling its split as an
epitope/paratope benchmark.

- [CHIMERA-BENCH canonical task paper](https://openreview.net/pdf?id=PyZvVIJbSy)
- [FoldBench low-homology complex benchmark](https://www.nature.com/articles/s41467-025-67127-3)

## Required decision rule

The confirmatory claim is supported only if the 95% antigen-cluster bootstrap
confidence interval for the primary cognate compatibility margin is entirely
above zero. Top-1 retrieval and MRR, contact AUPRC, epitope/paratope metrics,
calibration, spatial-neighbor enrichment, every seed, and every intervention
must still be disclosed. No favorable secondary metric can rescue a failed
primary endpoint.
