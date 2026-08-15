# Antibody pretraining-to-FrameContact transfer research lock — 2026-08-15

## Decision

The current 634,513,920-parameter OAS encoder remains a systems- and
reproducibility-focused foundation run. It is not presented as the largest
antibody language model. Its downstream value must be established by matched
development-only transfer into contact prediction and partner retrieval, not by
parameter count, pretraining loss, or GPU utilization alone.

Production pretraining may begin only after the frozen four-RTX-5090 efficiency
gate passes. Downstream transfer remains separately locked until exact-overlap
and near-identity decontamination audits are complete.

## Primary-source scale and objective comparison

- [AbLingua](https://www.nature.com/articles/s42003-026-10283-z) reports an
  encoder family up to 1.7B parameters trained on 1.4B antibody sequences. It
  combines scaling with an enlarged, motif-oriented vocabulary. FrameContact's
  650M residue-token MLM is therefore not a model- or data-scale record.
- [IgBERT and IgT5](https://pmc.ncbi.nlm.nih.gov/articles/PMC11654935/) were
  pretrained on more than two billion unpaired OAS sequences and continued on
  about two million native heavy-light pairs. This supports a future paired
  continuation, but an unpaired OAS run cannot claim to learn native H-L or
  antibody-antigen specificity.
- [S²ALM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12364524/) uses hierarchical
  sequence-structure pretraining with sequence-structure matching and
  cross-level reconstruction over 75M sequences and 11.7M structures. It is the
  relevant objective-strength upper bound; plain MLM must not be described as
  structure-aware pretraining.
- [CALM](https://www.biorxiv.org/content/10.64898/2026.02.25.707916v1) directly
  tests leakage-controlled bidirectional antibody-antigen retrieval. This
  reinforces FrameContact's requirement that a claimed specificity gain must
  improve partner retrieval, not only residue/contact metrics.

These works set different upper bounds. Published scores are not inserted into
the AsEP same-protocol result table unless a local adapter uses the same split,
inputs, selection rule, and metric implementation.

## What the present run can establish

The authenticated corpus contains 224,887,833 globally deduplicated training
records and 25,259,993,406 unique training residue tokens. The frozen campaign
trains to 30B consumed residue tokens, exposing the final 4,740,006,594 tokens
as an explicit repeat rather than relabeling them as unique data.

The systems evidence is meaningful only if all of the following remain true:

1. four RTX 5090 GPUs are scheduler-authenticated;
2. approximate `6NT` MFU is reported with the 209.5 TFLOP/s/GPU dense-BF16
   denominator and is not called profiler-exact;
3. compiled and eager runs have identical membership, work, precision, masking,
   initial state, and validation panel;
4. exact checkpoint/restart reproduces model, optimizer, trainer coordinates,
   and scientific trace;
5. every training segment is create-only, hash-sealed, predecessor-bound, and
   independently audited.

High GPU utilization or MFU alone does not establish biological value. The
scientific result is the later matched transfer delta per GPU-hour.

## Transfer gate

The transfer sequence is frozen as follows:

1. Run `audit_oas_asep_development_overlap.py` on the authenticated OAS train
   and validation shards and the AsEP train/validation cache. Search heavy and
   light chains separately. Reconstruct every matching OAS row from token arrays
   and re-prove its sequence SHA-256.
2. If exact matches exist, report both the full development panel and the
   pre-flagged exact-decontaminated subset. Do not retrain or select exclusions
   from metric outcomes.
3. Run a separately frozen MMseqs2 audit at 90% sequence identity and registered
   coverage before calling the panel cluster-decontaminated. A zero exact-match
   count is insufficient.
4. Freeze a checkpoint-to-residue-encoder adapter. Compare random initialization,
   frozen pretrained features, and end-to-end fine-tuning under matched seeds,
   updates, data, heads, checkpoint rule, and optimizer budget.
5. Select no architecture, threshold, adapter, or checkpoint on frozen,
   official, or external test data.

## Required downstream evidence

Report native pair-contact AUPRC and P@K, epitope/paratope AUPRC, bidirectional
partner top-1/MRR/margin, paired seed deltas, complex-cluster bootstrap
intervals, wall-clock GPU-hours, and transfer quality per GPU-hour. A transfer
is promoted only if contact ranking and partner retrieval improve without a
material regression in residue marginals. MLM validation loss is supporting
evidence, never the promotion criterion for FrameContact.

## Current boundary

The exact-overlap auditor and negative-control tests are implemented, but no
real AsEP/OAS overlap receipt has yet been produced. The 90% identity audit and
checkpoint adapter are not yet complete. No downstream transfer, official-test,
SOTA, or world-leading claim is authorized by this lock.
