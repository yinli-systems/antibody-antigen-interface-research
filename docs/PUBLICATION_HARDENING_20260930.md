# Publication hardening — 30 September 2026

This is an additive follow-up to the first main consolidation at
`4fc0613fefea83b194e88e2cf80d6e48157de4d5`.

## Delivered

- Generated, shorter main README with separate project results, external
  checkpoint diagnostics, counterevidence and incomplete experiments.
- An absolute-AP data chart from the same four-arm, ten-seed experiment.
- A 59-metric machine-readable registry: exact source SHA-256, selector, cohort,
  aggregation, evidence class and interpretation limit.
- A standard-library verifier for numeric drift, generated-file drift, local
  links and complete archived Git trees; optional explicit remote tag checking.
- A navigable directory of all 747 indexed evidence-facing branch paths, each
  tied to its actual source blob and full commit ID.
- Complete tracked-tree inventory for all 17 original snapshots: 55,630 entries
  including repetitions, 4,526 distinct path/blob versions, 4,394 unique blobs.
- Offline bundle created and restored in a separate bare repository; all 17
  tags/trees matched and `git fsck --full` exited successfully.
- Seven exact-source worktree routes; the multimodel route was actually
  exercised. The helper never submits training or changes an existing worktree.
- Nineteen publication-layer tests, including source tampering, missing/ambiguous
  selectors, non-finite numbers, negative-evidence guards and wrong destinations.

## Corrections, not cosmetic omissions

“Correspondence is necessary” was too general for the available experiments.
Same-AA shuffling preserves amino-acid identity, not residue correspondence.
Both issues are corrected in current presentation, with prior README bytes
retained under the archive and unchanged historical commits.

The original main core lacks the newer composite-embedding source required by
copied extensions. The exact-source helper prevents readers from accidentally
running a mixed revision. No claim is made that all historical code was merged
into one tested runtime, or that publication checks replace scientific reruns.

The official RePaRank measurement remains explicitly a baseline diagnostic,
with training overlap unknown. Pooled, macro and equal-component metrics remain
separate, and incorrect earlier conversational figures are documented rather
than presented as alternative measurements.

## Non-destructive scope

No original research branch, frozen source, historical result artifact, dataset,
checkpoint, embedding bank or GPU job was deleted or modified by this pass.
The only edits to existing first-consolidation files are reader-facing README
and result-index text. New artifacts, test tools, reports and snapshots are
additive. The backup covers Git objects/history; it does not pretend to be a
copy of every non-Git cluster tensor or account record.

Read the [publication-verification receipt](../archive/PUBLICATION_VERIFICATION_20260930.json),
[restore receipt](../archive/PRESERVATION_VERIFICATION_20260930.json),
[test transcript](../archive/PUBLICATION_TESTS_20260930.txt),
and [claim boundaries](CLAIM_BOUNDARIES.md).
