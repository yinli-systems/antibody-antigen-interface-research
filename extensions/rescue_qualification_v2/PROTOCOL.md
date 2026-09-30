# Rescue qualification v2 — independent supplement, 2026-09-30

No existing inference, training or aggregation job is modified. No model is
retrained here. The parent rescue-v1 protocol and its results remain immutable.
This supplement was specified before any trained rescue contact metric is read.

## Why this stage is necessary

1. Pure permutations of extra-feature rows do not introduce independent learning
problems for the current rowwise two-tower matcher. CE and its parameter gradients
are permutation equivariant; this is proven in RESEARCH.md and tested against
actual frozen feature inputs with an untrained copy of the original matcher.
2. Original rescue training admits complete ESM-IF1 sources only; original contact
evaluation instead requests every validation sequence and unconditionally appends
a one-valued presence flag. Missingness must preserve historical behavior, not
remove complexes or pretend a missing source is present.
3. An unconditional assignment can damage an already correctly aligned input.
A clean-input-through-rescue control is essential; a positive corrupted-input
rescue effect alone does not establish a deployable improvement.

## Scope / interventions

Primary matcher is the already registered v1 model and original assignment rule;
its training seeds, labels, learning rate, epochs and selected checkpoint are not
changed. New canonical-content assignment is a secondary engineering candidate:
candidate vectors, not original residue IDs, define tie ordering. It addresses
column-order tie instability, not a claim of higher contact accuracy. Neither
solver receives the synthetic true permutation at inference.

The same 217 previously exposed development-validation complexes are retained.
For every existing rescue/downstream seed pair and each registered corruption,
report aligned, corrupted, rescued and clean-rescued AP. Additionally record an
untrained-matcher baseline and the canonical-solver candidate without selecting
between them using contact metrics. All treatments keep ESM-2 and presence bits
unchanged. Completely absent ESM-IF1 sources remain all-zero with presence=0;
partial-residue missingness or cross-chain feature movement fails closed.

## Gates

- Verify frozen training-plan hash, source hashes, seed and selected epoch.
- Require a terminal history and best checkpoint for each of all three seeds.
- Bind evaluation input files before contact inference; pin absent features too.
- Identity AP must agree with the historical per-seed validation receipt within
  1e-6; otherwise retain failure and prohibit promotion.
- Require 217 IDs x 3 seed pairs x 3 corruptions, no duplicate or missing rows.
- Report absolute effects and equal-component/equal-seed bootstrap intervals;
  do not treat corruption conditions as independent biological samples.
- Gap recovered is a ratio of mean paired differences, not a mean of unstable
  per-case ratios. Nonpositive denominators yield null, not a clipped success.
- Keep negative clean-input changes and untrained-matcher outcomes visible.

## Biological limits

Same-chain synthetic reassignment is not antibody-antigen contact matching.
One-to-one assignment does NOT model biological many-to-many contacts. Exact
sequence separation is not homology decontamination. Matching-accuracy selection
on the existing validation chains does not produce a blind contact test. There
is no new OOD, affinity, co-folder or experimental-binding result in this stage.

No new GPU job is submitted by the preflight/probe. Future execution of the
supplemental evaluator requires the complete already-running training results.
