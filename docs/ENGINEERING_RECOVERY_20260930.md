# Execution engineering — 30 September 2026

Actual recovery source/evidence branch: `kevin/execution-recovery-20260930`,
commit `ecc826ee0394ce864aef872d146cda303206ff94`.
This is additive to publication main `e8ffb2a`; it does not rewrite archived
branches, scientific results, frozen training source, model weights or datasets.

## Engineering delivered

- Correct AF3 input staging from exact original bytes, without changing old paths.
- Explicit original SIF runtime, with FUSE and no missing shared-sandbox fallback.
- Software 3.0.1 and parameter-directory 3.0.0 recorded separately.
- Complete 25-sample model/sequence/backbone/output validation, not exit-code-only admission.
- Persistent submission intents, ambiguous-acknowledgement holds and duplicate-run protection.
- Schema-only adapter for the real `mean_reciprocal_rank` field, outside the source tree
  bound by active training. Original AP/bootstrap/layer selection remain unchanged.
- Complete 40-chain assembly with exact 217-case coverage, preserving base PDBs.
- 49 individual regression tests plus one full synthetic aggregation integration test.

## Actual jobs, not implied completions

At the imported 08:09:57 UTC snapshot, original array `1640503` has 12/36
completed tasks, six running and eighteen pending; no formal failure.
AF3 recovery task 0 `1641607` is pending Priority; tasks 1–39 have not been
submitted. Corrected analysis `1641627` is pending on
`afterok:1640503,afterany:1640514`. Old analyzer `1640514` is preserved, not patched
or replaced. The schema KeyError was reproduced separately on a real completed
receipt; the corrected pipeline passed a synthetic complete-campaign test.

No AF3 prediction success, layer winner, new blind/OOD measurement or accuracy
improvement is claimed in this engineering release. The full RePaRank structure,
learned rescue, co-folder and affinity endpoints remain unfinished.

## Evidence and exact execution revisions

[Recovery report](../extensions/execution_recovery_v1/README.md) ·
[Primary research](../extensions/execution_recovery_v1/RESEARCH.md) ·
[Actual job snapshot](../extensions/execution_recovery_v1/evidence/cluster/FINAL_SNAPSHOT.json) ·
[Reproduced schema defect](../extensions/execution_recovery_v1/evidence/cluster/REAL_SCHEMA_DEFECT.json) ·
[Resume safely](../extensions/execution_recovery_v1/HANDOFF.md)

The live AF3 job uses commit `0bd7366`; the scheduled analysis uses `b72e6bf`.
Their source hashes and exact paths are in the committed execution plans. The
new `execution-recovery` source route prepares the complete extension at the
release commit without submitting any jobs. Do not change pinned live files to
make their hashes pass.
