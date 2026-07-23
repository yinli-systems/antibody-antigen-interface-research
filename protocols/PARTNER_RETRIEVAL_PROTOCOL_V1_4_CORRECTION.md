# Partner Retrieval Protocol v1.4 — Authority-Boundary Correction

Status: locked before any AntiBERTy model is trained on, selected against, or scored on the corrected boundaries.

Date: 2026-07-23 CST

## Correction

The authoritative development split is the immutable artifact with SHA-256
`e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b`:

- authority train: 3,229 instances;
- authority validation: 216 instances;
- sealed test: 218 identifiers, used only for exclusion.

The previously generated exact-NW component splits are not clean post-selection
holdouts:

- concatenated-antibody plus antigen split, file SHA-256
  `c586fbd1ec83effe1208746d64950feddef280a95a938cf241293dda0b396b64`,
  contains 112 authority-train instances among its 245 validation instances;
- concatenated-antibody plus VH, VL, and antigen split, file SHA-256
  `5d92edd03e3b96edfa727823b2a5bbdf49e80614ef759fe0c6b14cf6c2b6b92b`,
  contains 8 authority-train instances among its 52 validation instances.

Those artifacts remain useful only as development-reuse or sequence-isolation
stress tests. They must not be described as clean holdouts and must not be used
for post-selection confirmation claims.

## Corrected boundary construction

Two versioned boundaries will be derived without reading labels, predictions,
checkpoints, or sealed-test sequences:

1. concatenated VH+VL antibody and antigen constraints;
2. concatenated VH+VL antibody, separate VH, separate non-empty VL, and antigen
   constraints.

For each mode, the components are the pre-existing authoritative graph
components assigned wholly to authority validation. The exhaustive exact global
Needleman–Wunsch authority-train-by-authority-validation edge inventory is then
applied at 80% aligned-region identity and 80% reciprocal coverage in every
declared view. A validation component is eligible if and only if none of its
members has a threshold-qualified exact-NW edge to any authority-train sequence
in any declared view.

The assignment rule is deterministic and has no size target:

- retain all 3,229 authority-train identifiers in `splits.train`;
- assign every eligible pure authority-validation component to
  `splits.validation`;
- assign every remaining authority-validation component to
  `splits.excluded_validation`;
- never move an authority-train identifier to validation;
- never add an excluded authority-validation identifier to training;
- require the three lists to be pairwise disjoint and to account exactly for
  authority train plus authority validation;
- require zero sealed-test intersection;
- independently rerun exhaustive exact global alignment across the final
  train-validation boundary in every declared view.

No model score, label, prediction, or checkpoint may influence eligibility,
component assignment, constraint-mode selection, or boundary size.

## Chronology and allowed interpretation

The 216-instance authority validation pool was previously inspected by the
exploratory ESM2/ESM2 retrieval experiment recorded in Protocol v1. Therefore,
the corrected subsets are clean relative to the subsequent AntiBERTy three-fold
architecture and epoch-selection procedure, but they are not globally untouched
data. They are reused development confirmation subsets, not independent test
sets.

Passing these boundaries can support a narrowly stated robustness result under
fixed authority and exact-NW constraints. It cannot support an external,
prospective, temporal, or world-leading claim. Independent confirmation requires
a separately locked external benchmark such as a versioned AVIDa protocol or
future temporal SAbDab structures.

## Frozen-test firewall

The sealed test remains closed. Its identifiers may be read only to prove
exclusion. Its rows, sequences, labels, embeddings, predictions, and metrics
must not be loaded, aligned, materialized, trained on, or scored during boundary
construction, verification, model selection, or development confirmation.
