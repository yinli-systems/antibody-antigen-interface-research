# Prospective low-N epitope-targeted antibody study

This directory is the computational and preregistration scaffold for a future
prospective wet-lab study. The scientific question is narrow:

> At the same experimental budget, does partner-specific ranking, with or
> without uncertainty/diversity-aware acquisition, improve the hit rate of a
> fixed antibody-design generator relative to that generator's native ranking?

FrameContact/ESM-C is a candidate ranker in this study, not the generator and
not an experimental result. No target, assay threshold, or candidate budget is
invented in this repository. The draft protocol remains blocked until those
items and an independent safety review are supplied by the experimental
collaborator.

## Fixed three-arm design

All arms draw from the same frozen proposal pool and use the same eligibility
rules. Each policy selects exactly `K` candidates; policy overlap is retained
and reported rather than silently replaced.

1. `generator_baseline`: the generator/oracle's native score.
2. `partner_specific`: a frozen partner-specific FrameContact/ESM-C score.
3. `uncertainty_diversity`: a frozen acquisition score with deterministic
   first-pass cluster coverage.

The laboratory receives a shuffled list of blinded IDs and payload hashes. It
does not receive arm membership or model scores. One physical candidate may
belong to more than one policy slate and is assayed once. The private mapping
is Git-ignored.

## Hard gates

- Use a non-pathogenic, non-toxin research target with a documented safety
  review, available antigen, positive and negative controls, and a feasible
  orthogonal confirmation assay.
- Freeze the target epitope, proposal pool, eligibility rules, score versions,
  per-arm budget, primary assay, hit threshold, and analysis before any
  candidate-level outcome is available.
- Keep candidate generation fixed across arms. Only the selection policy may
  differ.
- Blind the assay operator to policy membership. Preserve all failures and all
  assayed candidates in the analysis.
- Treat hit-rate comparisons as conditional on this target, proposal pool,
  assay, and fixed candidate slate. A positive low-N study is not a universal
  antibody-design claim.

## Workflow

1. Copy `protocol/PROTOCOL_DRAFT_V1.json`, fill every blocked field, and obtain
   the recorded safety review. Bind the exact generator, proposal-pool
   manifest, FrameContact/ESM-C checkpoint, acquisition-score implementation,
   and diversity-clustering implementation by SHA-256.
2. Freeze it before candidate outcomes exist:

   ```bash
   python3 prospective_study.py freeze-protocol \
     --input completed_protocol.json \
     --output authority/PROTOCOL_V1.json \
     --receipt authority/PROTOCOL_V1_RECEIPT.json
   ```

3. Produce a TSV with exactly these columns:
   `candidate_id`, `payload_sha256`, `eligible`, `baseline_score`,
   `partner_score`, `acquisition_score`, and `diversity_cluster`.
4. Create a 32-byte-or-longer local blinding key outside version control and
   freeze the three policy slates:

   ```bash
   python3 prospective_study.py freeze-slate \
     --protocol authority/PROTOCOL_V1.json \
     --protocol-sha256 <sha256> \
     --candidates scored_candidates.tsv \
     --blinding-key-file private/blinding.key \
     --public-manifest blinded/CANDIDATE_MANIFEST.json \
     --private-mapping private/CANDIDATE_ARM_MAPPING.json \
     --receipt blinded/CANDIDATE_MANIFEST_RECEIPT.json
   ```

5. The laboratory returns exactly one row per blinded ID with columns
   `blinded_id`, `primary_hit`, `primary_value`, `confirmatory_hit`, and
   `expression_pass`, plus a separate two-row control file. Empty values are
   allowed only for the last two candidate fields. The analyzer recomputes
   `primary_hit` from the frozen direction and threshold and rejects a mismatch.
6. Analyze once, without changing the slate or threshold:

   ```bash
   python3 prospective_study.py analyze \
     --protocol authority/PROTOCOL_V1.json \
     --protocol-sha256 <sha256> \
     --public-manifest blinded/CANDIDATE_MANIFEST.json \
     --private-mapping private/CANDIDATE_ARM_MAPPING.json \
     --candidate-receipt blinded/CANDIDATE_MANIFEST_RECEIPT.json \
     --assay-results assay_results.tsv \
     --control-results control_results.tsv \
     --output results/BLINDED_ASSAY_RESULT.json \
     --receipt results/BLINDED_ASSAY_RESULT_RECEIPT.json
   ```

   The control file has exactly `control_id`, `control_role`, `primary_value`,
   and `control_pass`. The positive control must satisfy the frozen hit rule and
   the negative control must not satisfy it.

The result JSON is deliberately data-only. It reports arm hit rates, Wilson
intervals, pairwise risk differences, and candidate-level Bayesian-bootstrap
intervals while preserving overlap covariance. It does not write a superiority
claim.

## Evidence motivating this design

- BindCraft demonstrated that structure-conditioned binder generation can
  achieve experimentally verified binders, while success varies materially by
  target and workflow: <https://www.nature.com/articles/s41586-025-09429-6>
- Germinal showed prospective, epitope-targeted antibody design with tens of
  experimentally tested designs per antigen: <https://www.nature.com/articles/s41587-026-03187-0>
- RFdiffusion-based antibody design highlighted that atomic-accuracy designs
  can still have low wet-lab success, making selection/oracle quality a central
  bottleneck: <https://www.nature.com/articles/s41586-025-09721-5>
- CAFA established the value of prospective evaluation for separating model
  development from later experimental evidence:
  <https://www.nature.com/articles/nmeth.2340>

These sources motivate the study design; they do not determine its target,
assay, or expected result.
