# FrameContact / ABPairFM world-class research roadmap v1

## Decision

The project should pursue a strict pair-specific contact benchmark and a small
contact-guided CDR sequence-design proof of concept. It should not present
itself as an AlphaFold 3 or RFantibody competitor: those systems generate or
predict full atomic structures and, in RFantibody's case, include experimental
screening. The immediate contribution remains residue-pair ranking under a
strong leakage firewall.

## Corrections to the motivating comparison

- AlphaFold 3 reports protein--antibody ranking over 1,000 model seeds, but this
  is a specialized high-sampling evaluation, not a general 60% success claim.
  The relevant evidence is that antibody--antigen docking remains difficult
  even for a large diffusion-based structure model.
- The 2025 AF3 benchmark's antibody set removes antigens above 40% identity to
  antigens in AF3's training set and clusters antigens at 40%. That is useful,
  but weaker than this repository's intended four-view rule: VH, non-empty VL,
  concatenated antibody, and antigen must all be cold.
- RFantibody is experimentally validated full-backbone/pose design conditioned
  on framework and epitope information. A contact-guided sequence sampler is a
  hypothesis generator, not a replacement.
- Chai-1 has public code, weights, and a technical report. No primary public
  Chai-2 technical artifact was found during this review, so Chai-2 claims are
  excluded until an authoritative source is available.
- Hiring and scholarship advice from personal blogs is not scientific evidence
  and is not used to define model gates.

Primary sources: [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w),
[independent AF3 benchmark](https://academic.oup.com/bib/article/26/6/bbaf616/8351050),
[RFantibody](https://www.nature.com/articles/s41586-025-09721-5), and
[Chai-1](https://github.com/chaidiscovery/chai-lab).

## Track A: four-view low-homology evidence

The existing exact S2-80 evidence must remain intact. Its complete relation
proved that the attempted full-closure binary split was infeasible, and all
provisional assignments remain unauthorized. An infeasibility proof is not
permission to train a leaking split.

The new S40 track is a separate, create-only train-sequence stage:

1. materialize 40%-identity / 80%-reciprocal-coverage relations independently
   for VH, VL, concatenated antibody, and antigen;
2. merge each related component without splitting PDBs or exact physical
   systems;
3. allow an explicit discard/vertex-separator set when balance is infeasible;
4. run an independent all-boundary audit;
5. authorize adaptation only if all four views have zero crossings.

The relation and strict no-discard feasibility stages are now complete. The
four relations contain `2,471,824 / 1,586,364 / 2,508,590 / 56,069` canonical
rows for VH / VL / concatenated antibody / antigen. On `2,718` unique exact
pairs, the largest antibody component has degree `2,716`; only three antibody
components remain, so no 15--25% validation subset exists when the largest is
kept in train. The antigen axis remains feasible with 605 components. This
means that copying FoldBench's antigen-side 40% rule to complete antibody
variable domains collapses the antibody pool because of shared immunoglobulin
frameworks.

The generic gate is implemented in
`src/antibody_pretraining/four_view_coldness.py`. The authenticated PINDER-train
relation campaign is under
`remote_tools/abf_paircontact_v1/pinder_scale_v1/`, with compact evidence under
`results/pinder_antibody_s40_v1/`. The uniform four-view-40 track is therefore
a documented infeasibility result, not the default benchmark. Any replacement
must state its asymmetric thresholds explicitly; a defensible candidate is
antigen-40 plus a separately justified antibody/CDR coldness rule. This stage
cannot create a new blind test. Final evidence still needs a post-cutoff cohort
frozen before label access or a third-party sealed executor.

## Track B: pair-contact model

The active B32K run remains the scientific priority. It already uses explicit
pair labels, paratope/epitope auxiliary supervision, legal partner swaps,
adaptive hard-negative weighting, and multiplicity weighting. Two matched
canaries were correctly rejected:

- bipartite edge refinement: contact-AP difference-in-differences
  `-0.0000424 [-0.0002954, 0.0001756]`;
- class-balanced focal substitution: contact-AP difference-in-differences
  `-0.0001504 [-0.0003098, -0.0000154]`.

Neither candidate is promoted or tuned against the canary. A main-paper pair
claim requires B32K to pass, then one frozen larger PINDER stage and an external
same-definition pair benchmark.

## Track C: contact-guided CDR generation

`src/antibody_pretraining/contact_guided_cdr_generation.py` implements the
smallest defensible generative extension:

- freeze framework residues;
- iteratively sample canonical amino acids only at a supplied CDR mask;
- condition proposals on an antigen sequence through a caller-supplied
  denoiser;
- rerank with a frozen residue-pair contact model over a predeclared antigen
  hotspot, with an off-target contact penalty;
- preserve deterministic seeds and reject malformed/non-finite inputs.

This is a sequence-proposal PoC. It generates neither backbone coordinates nor
rigid-body poses. Promotion is blocked until B32K passes and a train-only CDR
denoiser exists. The first experiment must compare guided and unguided sampling
at identical seeds and proposal budgets; contact score is a surrogate outcome,
not binding evidence. Functional claims require experimental validation.

## Publication package

The fastest credible paper is not "negative results alone." It is a benchmark
and methodology paper whose contribution is:

1. exact four-view leakage accounting and explicit infeasibility evidence;
2. a scalable pair-contact curriculum with physical-system weighting;
3. preregistered canary rejection and append-only receipts;
4. one passing pair-ranking result on a leakage-aware cohort;
5. a small generation integration result reported only as a surrogate study.

The current negative evidence is summarized in
`paper/neurips_2026_framecontact/PAIR_RANKING_NEGATIVE_EVIDENCE_V1.md`. It is
appendix/workshop-ready evidence, not yet a complete NeurIPS main-track claim.

## Stop/go table

| Gate | Current state | Promotion condition |
|---|---|---|
| B32K pair-contact | running, frozen | pair AP above prevalence, positive matched-minus-swap, all audits pass |
| S40 four-view relation | passed, job 1533008 | four authenticated relation receipts |
| Uniform S40 split | failed strict no-discard feasibility | do not train; redesign and name thresholds explicitly |
| CDR generator G0 | local implementation passed | framework/validity/determinism tests |
| CDR generator G1 | blocked | B32K pass plus train-only denoiser and matched control |
| blind generalization | unavailable | post-cutoff frozen cohort or third-party sealed labels |
| functional design | unavailable | orthogonal filtering and wet-lab binding validation |
