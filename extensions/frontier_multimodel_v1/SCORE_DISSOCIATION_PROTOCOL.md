# Partner-score dissociation protocol v1 — frozen 2026-09-30

## Question

Why did contact-map AP deteriorate when residue correspondence was corrupted while
the existing partner-ranking MRR improved? The primary hypothesis is that the
current score — double-centred interaction logits followed by a length-normalised
log-mean-exp — can reward interaction dispersion or an upper tail that is not
spatially aligned with the true contacts. Candidate length, chain-local packing,
node-head priors and duplicate-sequence panels are competing explanations.

## Scope and firewall

Development validation only: the existing 217-instance SAbDab2 validation split,
ESM-2 + ESM-IF1 validation-best checkpoints for training seeds 1, 2 and 3. No
new training, test contact labels, OOD labels or model selection. Conditions are
fixed before scoring: aligned identity, within-chain cyclic shift +8, and
within-chain deterministic global permutation, using the already audited
correspondence-v2 implementation and perturbation seed 20260930.

The candidate universe is the Cartesian product of the 122 unique validation
antibody sequences and 119 unique validation antigen sequences: 14,518 distinct
sequence pairs per seed and condition. Repeated instances do not create repeated
candidates. Every project-observed sequence pair is a known positive; all other
pairs are *unlabelled distractors*, not non-binders. Multi-positive anchors are
retained. Exact score ties use an ID-order-independent hypergeometric expectation.

## Frozen score decomposition

For every sequence pair report, without fitting or choosing among them:

1. the checkpoint's current double-centred log-mean-exp score;
2. uncentred interaction-logit log-mean-exp;
3. double-centred top-1%-with-minimum-16 mean;
4. centred and uncentred interaction mean, standard deviation, RMS, maximum and
   top-16 mean;
5. pair-probability mean, maximum and top-16 mean;
6. epitope/paratope head mean, maximum and top-4 mean;
7. explicit antigen/antibody decoder-bias mean and standard deviation;
8. chain lengths, pair count, and a label-free chain-local C-alpha packing proxy
   from the development-only independently predicted structures.

The packing proxy is the mean number of same-chain C-alpha neighbours within 10
angstrom and the fraction with at most 12 neighbours. Heavy and light chains are
computed separately and residue-weighted; no artificial peptide bond or shared
inter-chain frame is introduced. It is not solvent-accessible surface area.

## Endpoints

For each direction, seed, condition and frozen score: expected MRR of the first
known positive, expected Recall@1/5/10 under random ordering inside exact ties,
and known-positive AP with every unobserved pair explicitly described as
unlabelled. Also report per-anchor Spearman correlations with candidate length,
packing proxy, node-head prior and interaction dispersion.

Primary diagnostic comparisons pair corruption against identity within the same
seed and unique anchor. Crossed bootstrap resamples the three fixed seeds and
unique anchor groups. Intervals are pointwise exploratory, not corrected for the
large score family. Contact-link analysis joins the prior 217 matched-complex AP
records and tests whether score changes track contact-AP changes; it does not
reuse contact labels during all-vs-all inference.

## Decision boundaries

This stage may identify a score failure or bias. It cannot promote a replacement
score, because all alternatives are inspected on validation. A replacement must
be frozen and evaluated on a newly admitted cohort. No ranking result is called
binding specificity because distractors are not experimentally confirmed
non-binders. No result here changes the closed evidence freeze.

## Primary-source constraints

- SaProt/ProSST work remains separate: real 3Di/quantised structure tokens are
  required; AA-only input is not accepted as structure-aware.
- RePaRank is a later same-protocol baseline, not evidence for choosing a score
  in this diagnostic.
- The 106x106 VHH and CALM studies motivate rigorous retrieval controls, but
  manually recombined or unobserved pairs remain unverified non-binders.

## Amendment after pre-inference canary failure

Canary job `1639957_0` failed before model inference because the initial code
incorrectly required every candidate chain to have a packing structure. Eighteen
unique validation antibodies and nineteen unique validation antigens lack at
least one chain-level structure authority; this is consistent with the historical
zero-filled representation contract. The candidate universe, endpoints and score
family remain unchanged. Packing is now an optional covariate with an explicit
presence flag; no sequence is dropped, imputed or assigned a fabricated value,
and proxy correlations use finite available pairs only. Attempt 1 remains a
preserved failed plan. A new plan/hash is required before retrying. No scientific
score output was produced or inspected before this amendment.
