# ABF-PairContact next-scale research plan v1

Status: v20 failed its frozen gate. The endpoint-zero-sum head-only mechanism
line is closed; the large-data sequence-only contact-supervision route is the
active research direction. This document does not by itself authorize model
selection, label access, or a scientific claim.

## Decision point

The frozen v20 precision-confirmation canary failed because contact and ranking
train-loss reductions were 0.009102 and 0.014496, below the preregistered 0.01
and 0.02 gates. All four independent holdout improvement gates passed, but the
complete gate did not. The cross-specific adapter comparison is therefore not
authorized.

- Stop the endpoint-zero-sum mechanism line. Do not relax the thresholds or
  inspect the failed holdout to design v21.
- Move directly to the large-data contact-supervision and benchmark route
  below.

## Closed mechanism line

The current pair-aware ESM2 implementation shares `q_proj` and `k_proj`
between intra-chain rotary attention and position-free cross-chain attention.
Conventional LoRA on those shared modules would not isolate partner exchange,
and the v20 gate did not authorize a cross-specific adapter experiment. No
adapter rank, optimizer, holdout, or threshold may now be selected from v20.
This mechanism line remains closed unless a genuinely new, independently
motivated protocol is preregistered without inspecting v20 holdout examples.

## Large-data sequence-only contact supervision

The scale route should use structural complexes only to derive training labels;
model inputs remain amino-acid sequences. The first target is a leakage-audited
mixture of PINDER/PPInterface interfaces, followed by antibody-focused
adaptation. Candidate records must carry source, release date, endpoint
sequence-cluster IDs, interface-cluster IDs, structure provenance, and exact
contact-label hashes.

Required data controls:

- remove exact pair and reversed-pair duplicates across sources;
- cluster each endpoint independently and cluster interfaces separately;
- create train/development splits that are cold in both endpoints and in
  interface similarity;
- balance native noncontacts by row and column endpoint degree before random
  fill;
- include partner-swap and same-family hard negatives, with adaptive weighting
  confined to training;
- keep experimental or predicted coordinates out of model inputs unless a
  separately named structure-input arm is being evaluated;
- reserve all post-cutoff or third-party sealed complexes before label
  materialization.

Recommended staged evidence:

1. 32k-scale PPInterface/Handshake-sized reproduction gate;
2. 500k contact-supervised scale gate, matching the order of magnitude used by
   LC-SEPLM;
3. 1M+ interface-pair training from the leakage-audited PINDER mixture;
4. antibody-focused adaptation without exposing the final external cohort;
5. three-seed evaluation plus a third-party sealed, post-cutoff antibody-and-
   antigen double-cold test.

## Benchmark contribution

The official PINDER roadmap still lists an antibody-focused `pinder-ab`
benchmark as future work. A high-value project contribution is therefore a
versioned antibody-interface benchmark rather than only another model score.
Its public development component can be built from pre-cutoff PINDER and
SAbDab2 records, while its definitive test component must remain a separately
collected post-cutoff cohort with third-party sealed labels.

The benchmark should publish sequence and interface cluster inventories,
release-date provenance, duplicate and contamination reports, eligibility
rules, contact-label definitions, zero-positive handling, frozen metric code,
and baseline predictions. Training examples must be removed transitively by
both antibody-side and antigen-side similarity and by interface similarity.
The benchmark may be released even if the proposed model does not win every
task; its value comes from making leakage and input modalities auditable.

## Metrics needed for a world-class claim

Residue marginals and pair contacts must be separated.

- Pair contact: pooled AP, mean-per-complex AP, P@L/5, P@L/10, P@K,
  calibration error, and matched interaction accuracy.
- Interface residues: AP, MCC, F1, AUROC, and per-complex distributions.
- Partner specificity: cognate-versus-swap ranking and performance on examples
  with identical query chains but different partners.
- Robustness: endpoint sequence identity bins, interface-similarity bins,
  predicted-versus-experimental monomer geometry when applicable, and
  homomer/heteromer strata.
- Systems: pairs per second, residue tokens per second, peak memory, trainable
  parameters, and full-proteome retrieval cost.

No single development median authorizes a SOTA claim. The final claim requires
the frozen external cohort, complete metric definitions, and comparison under
matched input modalities and split constraints.

## Primary research basis

- Handshake (2026): <https://doi.org/10.64898/2026.06.04.730144>
- LC-SEPLM (2026): <https://arxiv.org/abs/2607.22777>
- PPLM (2026): <https://doi.org/10.1038/s41467-026-70457-5>
- RaftPPI (ICLR 2026): <https://openreview.net/pdf?id=Dp1RM3gPg8>
- MSA Pairformer (Cell 2026): <https://doi.org/10.1016/j.cell.2026.06.029>
- PINDER: <https://doi.org/10.1101/2024.07.17.603980>
- DataSAIL: <https://doi.org/10.1038/s41467-025-58606-8>
- PLM leakage audit: <https://doi.org/10.1038/s42256-025-01176-7>
