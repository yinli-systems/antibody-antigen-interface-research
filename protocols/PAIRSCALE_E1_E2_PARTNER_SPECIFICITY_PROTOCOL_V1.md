# PAIRScale E1/E2 Partner-Specificity Protocol V1

## Status and scope

This protocol authorizes a development-only repair of the failed PAIRScale
partner-specificity gate. It does not authorize confirmatory evaluation,
checkpoint promotion, large-scale training, or a comparative-superiority
claim.

The current backbone already contains two interleaved bidirectional residue
fusion blocks and an explicit pair state. This protocol therefore does not
describe that implementation as late fusion. E1 and E2 isolate the missing
scientific mechanisms:

1. bidirectional cognate retrieval and matched-decoy ranking; and
2. an auditable contact-logit decomposition into antibody unary, antigen
   unary, and pair-conditioned interaction residual terms.

The previously used endpoint-disjoint 192/64 assignment is development data.
It is not an untouched final test after prior model comparisons. No new final
test may be read, materialized, or scored under this protocol.

## Frozen model semantics

For every candidate antibody residue `i` and antigen residue `j`, the contact
logit is

```text
z[i,j] = u_ab[i] + u_ag[j] + r_contact[i,j]
```

The unary terms are computed from projected monomer residue streams before
any cross-partner fusion. The interaction residual is computed from the fused
explicit pair state. Padding and pairs outside the authenticated candidate
mask are exactly zero in every published tensor.

The specificity projection is separate from the contact projection. It reads
the same pair state but cannot read either unary logit. Its residue-pair output
is double centered before aggregation. All retrieval and cognate-margin losses
must consume only this centered specificity residual score.

The distance-mass contact decoder remains available as an auxiliary diagnostic
and for exact backward compatibility. It cannot be silently substituted for
the decomposed contact logit in an E2 run.

## Score-free bidirectional decoy policy

For every observed cognate pair, both directions are constructed:

- antibody anchor to candidate antigens; and
- antigen anchor to candidate antibodies.

Candidate selection may read only canonical endpoint identifiers, instance
identifiers, endpoint lengths, and the fixed seed. It must exclude every pair
present in the observed incidence relation so that a known positive is never
treated as a decoy. Unknown cross-pairs remain unlabeled ranking controls, not
measured non-binders.

Candidates are ordered by absolute replaced-endpoint length difference, then
by a domain-separated SHA-256 tie break. Distinct replaced endpoint identifiers
are required. Contact labels, contact density, positive count, affinity,
embedding values, predictions, losses, ranks, epitope size, and paratope size
are forbidden selection inputs.

## Objectives

The contact objective retains per-complex masked BCE and within-complex hard
negative ranking. The partner objective contains both:

- a multi-positive bidirectional softmax retrieval loss; and
- a pairwise cognate-vs-decoy softplus margin in both directions.

The retrieval implementation accepts an explicit positive mask. This prevents
known cross-reactive or repeated observed pairs from becoming false negatives.
The frozen score-free selector excludes known positives from the decoy panel,
but the loss remains multi-positive by construction.

No binary non-binding label is assigned to an unknown decoy. Retrieval and
margin weights are reported separately from the contact losses. A result is
invalid if the specificity loss backpropagates into either unary head.

## First ablation sequence

- B0: completed centered multi-decoy reference.
- B1: current backbone plus bidirectional retrieval and margin, with centered
  legacy pair evidence as the specificity input.
- B2: B1 plus explicit unary/interaction decomposition and an independent
  specificity residual projection.
- B3: only after B2, compare two and four interleaved fusion blocks at matched
  non-embedding parameter count and identical objectives.

Runs use seeds 17, 42, and 73 for screening. A candidate may be extended to
five seeds only after the frozen screening rule is applied. No early stopping
or development-holdout checkpoint selection is allowed.

## Development gates

Gate S requires all of the following on the frozen development panels:

1. raw and centered matched-minus-decoy paired component-bootstrap 95% lower
   bounds above zero;
2. antibody-to-antigen and antigen-to-antibody conclusions in the same
   direction;
3. at least four of five seeds positive after an authorized extension; and
4. homolog-matched retrieval enrichment at least three times its declared
   random baseline.

Gate C requires at least `+0.01` absolute macro pair AUPRC over the strongest
separable baseline, a paired component-bootstrap lower bound above zero, and
directionally consistent P@K and prevalence-normalized AP lift.

Gate R requires five seeds, component-aware paired bootstrap, a fixed
permutation test, and prespecified strata. Scaling remains blocked until Gates
S, C, and R all pass. Thresholds cannot be weakened after results are observed.

## Required tests and evidence

Before any cluster run, the implementation must prove:

- exact masked `u_ab + u_ag + r_contact` reconstruction;
- monomer unary invariance to the opposite partner;
- additive row/column invariance of the centered specificity score;
- no unary-head gradient from retrieval or margin objectives;
- deterministic symmetric decoy selection and observed-incidence exclusion;
- multi-positive loss behavior and permutation symmetry;
- finite forward, backward, optimizer, and exact-resume paths; and
- unchanged legacy distance-mass behavior when decomposition is disabled.

Every independently verified evidence unit is committed and pushed. Failures
and null results are retained. No identical-benchmark evidence means no SOTA
or world-leading claim.
