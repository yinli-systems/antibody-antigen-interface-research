# Post-freeze correspondence extension — 2026-09-30

This is a NEW exploratory development campaign. Evidence Freeze v2 and all
historical protocols, test closures, checkpoints and claims remain unchanged.
No new test access, model selection, training or superiority claim is authorized.
The existing validation set is reused and explicitly not independent confirmation.

## Frozen design

Three existing encoders (ESM-IF1, AntiBERTy, ProteinMPNN), fixed training seeds
1, 2, 3, validation-best checkpoints already on disk, all 217 development
validation complexes if the existing manifest still resolves to that count.
The machine-readable PLAN.json is written before scoring and externally SHA-bound.
Every selected contact, embedding, present representation, source file and
checkpoint is hashed; missing representation paths are recorded and must remain
missing. Test metadata is used only for disjointness/known-pair exclusion.
No test contact or feature artifact is read.

Twelve conditions: identity; chain-preserving shifts +1, -1, +2, +4, +8, +16;
within-chain local block shuffles 4 and 8; within-chain same-amino-acid shuffle;
within-chain global shuffle; shift-8 followed by its KNOWN inverse.
The last condition checks implementation recovery, not learned realignment.
ESM-2 columns and presence flags never move. Every permutation is label-free,
bijective and deterministic; VH/VL boundaries must not be crossed.
Wraparound and actual linear displacements are explicitly reported. These are
residue-index interventions, not claimed measurements in angstroms.

## Outcomes and analysis

Primary: mean-per-complex non-interpolated pair AP on the pinned candidate grid.
Also report model-derived marginal outer-product AP, epitope AP and paratope AP.
Zero-positive cases are not silently averaged as zero; eligible counts are
reported and paired comparisons use identical cases in every paired seed.
Equal-cluster/equal-seed crossed bootstrap is the primary descriptive estimand;
complex-weighted effects and cluster-only intervals conditional on fixed seeds
are separately reported. All intervals are pointwise and exploratory, with no
promotion rule, equivalence claim or multiple-comparison significance claim.

A reciprocal partner-retrieval diagnostic uses 16 anchors selected by an ID hash,
15 length-matched, distinct partner-sequence alternatives, excluding known
positives and the same authority component. Identity, shift-8 and global-shuffle
conditions are evaluated in both directions. Unobserved pairings are UNKNOWN,
not confirmed non-binders. Scores use the checkpoint's existing complex-score
function; no validation/test search chooses a new score. Exact ties are handled
by expectation over uniformly random tie orders, not an ID-sort advantage.
This is a matched-panel diagnostic, NOT the promised large all-vs-all benchmark.

## Engineering gates

CPU contract suite must pass before launch. GPU smoke must pass before formal
submission. Identity re-evaluation and inverse restoration must agree within
absolute probability error 1e-6. Source, split, checkpoints, selected input bytes,
model state and presence masks must stay unchanged. Missing result/task/seed/case
or mismatched digest blocks aggregation, never silently reduces the denominator.
Dense-candidate support is required; another support needs its own contract.
Original failures remain on disk, retries use a new attempt root/plan.

Budget: 9 tasks, at most 3 concurrent single-4090 jobs, 45-minute walltime each
(maximum formal allocation 6.75 GPU-hours), plus a bounded engineering smoke.
Outputs are scalar per-case metrics/diagnostics and partner scores, not copies of
weights, full embeddings or dense pair-prediction stores. No existing job is
cancelled, no old artifact deleted, no sealed set opened.

## What this does not establish

Inference corruption is not training corruption. Known-inverse restoration does
not establish a learned rescue mechanism. Three seeds are not ten new independent
biological samples. Preserved alignment is not established as mathematically
necessary for all possible predictors. Missing structural features are retained
with the original presence flags; their load counts are reported. Encoder widths
are inherited, not a new parameter-matched training experiment.
