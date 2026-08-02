# Confirmatory and external evaluation plan v1

Status: frozen execution order for work after the five-seed development matrix.
No external label is opened by this document.

## Why the current SAbDab2 test is not confirmatory

An interrupted preprocessing task read pair-label artifacts for 51 current
test-role complexes. It produced no predictions or metrics, but this prevents
a procedural never-read claim. The current test can be reported only with that
disclosure; it cannot choose a model or provide the strict confirmatory result.

## Gate 1: finish and freeze development selection

Complete all receipt-bearing binary-head, distogram, and local-triangle seeds.
Authenticate the paired aggregation and apply the pre-registered promotion
gates without manual overrides. Freeze one architecture, five checkpoints,
calibration rule, thresholds, input representation, and evaluation source
before any external test graph or label is opened.

## Gate 2: AsEP official test, with decontamination

[AsEP](https://papers.nips.cc/paper_files/paper/2024/hash/15add6732964d5b1f0954058bf3ccc88-Abstract-Datasets_and_Benchmarks_Track.html)
is an official antibody-specific epitope benchmark that represents an
antibody-antigen complex as two graphs and formulates the target as bipartite
link prediction. The official code and data authority are published in the
[AsEP repository](https://github.com/biochunan/AsEP-dataset).

The official test is not automatically independent of a newer SAbDab2
training snapshot. After model selection is frozen, materialize label-free test
metadata and remove or separately flag every member with any of:

- exact PDB/complex identity overlap;
- antibody or antigen endpoint sequence-cluster overlap;
- antigen Foldseek structural overlap above the registered threshold;
- known cognate-edge overlap.

Do not inspect epitope/contact labels while building this mask. Open the
official graph labels once, score both the complete official test and the
pre-frozen decontaminated subset in the same session, and consume the opening
token. Report contact AUPRC/P@K, epitope/paratope AUPRC, calibration,
wrong-partner drop, and same-antigen/different-antibody behavior. If the
decontaminated subset is too small, report that as a limitation rather than
relaxing the overlap rule.

## Gate 3: prospective temporal confirmation

[SAbDab2](https://sabdab.opig.stats.ox.ac.uk/) states that standardized
train/test splits are updated every six months. Preserve the current source,
model, thresholds, and all sequence/structure cluster authorities, then admit
only complexes first appearing in a future release. Exclude structural or
endpoint overlap with every current train/validation complex. This future
release is the strongest available procedural never-seen test for the present
model.

## Gate 4: CHIMERA canonical co-design

[CHIMERA-Bench](https://arxiv.org/abs/2603.13431) defines the canonical task as
epitope-conditioned CDR sequence-structure co-design and supplies unseen
epitope, unseen antigen-fold, and prospective temporal splits. A residue
classification score on those splits is diagnostic only and must not be called
a CHIMERA benchmark result.

After the contact model is frozen, use its contact representation as the
epitope/partner-conditioning interface to a CDR generator. Evaluate the
official co-design task and metric groups without redefining its targets. The
key causal comparison is whether contact-gated conditioning improves the
correct partner/epitope design objective over unary surface or CDR priors.

## Claims allowed at each stage

| Evidence | Allowed statement |
|---|---|
| five-seed train/validation matrix | development ablation only |
| current SAbDab2 test | disclosed conventional test, non-confirmatory |
| decontaminated AsEP official test | external antibody-specific generalization |
| future SAbDab2 release | prospective temporal generalization |
| CHIMERA official task | epitope-conditioned CDR co-design result |

No stage authorizes a state-of-the-art claim without matching the comparator's
inputs, split, checkpoint provenance, and metric implementation.
