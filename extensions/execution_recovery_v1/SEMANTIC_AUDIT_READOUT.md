# AF3 semantic audit recovery — verified readout

Snapshot: **2026-09-30 09:14:18 UTC**. This is an integrity/execution report,
not a new contact-prediction metric, binding result or structure-accuracy claim.

## Resumed actual state

The first read found that a prior controller had already admitted task 0 without
repeating inference at 08:57:26 UTC, then submitted array **1641712** for tasks
1–39 at 08:57:47 UTC. Neither was submitted again in this review. That executor
(`79f6a23`) checked backbone identity only. We added an independent stricter
post-audit without changing its running source, inputs or prediction outputs.
An existing uncommitted patch was copied into an isolated worktree and its
provenance recorded; the original working directory was not modified.

## Diagnosis confirmed on the original files

Original job **1641607_0** remains FAILED 1:0. Its 25 predictions were complete;
only the old post-inference byte-comparison check failed. The top-level CIF and
seed **42**, sample **2** differ solely in the valid serialization timestamp of
`_ma_model_list.model_group_name`: `16:40:10` versus `16:40:11`.
The maximum CSV score remains **0.6402947117306571**. No ranking was changed.
The actual frozen task is a **78-residue antigen chain**, with **597 total atoms**
and **312 N/CA/C/O atoms**; older conversational descriptions of a 54-residue
light-chain task do not describe this content-bound recovery plan.

| Identity | SHA-256 |
|---|---|
| Top-level original CIF | `ddb22d140df46fa6c4df6adfd324d7ea92cfb8db07f88b9af1d1b3e2b3016c01` |
| Seed-42/sample-2 original CIF | `2dbe2afcf7ea105c571ec0712a30746f9a5c8332814b5485efd19768cabb88de` |
| Shared strict semantic digest | `724b4cd1691e7e0beefb8afa5cb1fd7ae0ffd25de5411f1b7c3076eab188568f` |
| Preserved original failure receipt | `9b36a76b1ec6e856c16a87a31cb4f19de00bf14a6cc76c7e5189a24e779944f1` |

The original failed task and the admitted copy passed independent strict
post-audits. All **81 prediction files** in the recovered copy are byte-identical
to their originals; both trees remained unchanged throughout the checks.
All 25 sample confidences, exact chain/sequence, finite atom coordinates,
seed/sample completeness and the selected top-level confidence payloads passed.
A full rehash of the pinned AF3 SIF and model weights also passed for the admitted
copy. The original FAILURE.json and COMPLETE.json were not overwritten.

## Narrower than backbone-only equivalence

All parsed mmCIF data items must match; only a valid timestamp substring in the
single named field may differ. Model version, other metadata, all atom positions,
residue IDs, atom names, side chains, element, alternate location, occupancy and
pLDDT remain checked. No coordinate tolerance or rigid alignment is used.
Duplicate CIF items and nonfinite values fail closed. Atom pLDDT arrays must
match CIF B factors, token/atom chain identities and matrix dimensions must
match the model, and both top-level confidence payloads must match the maximum
sample. Full-precision ranking CSV values, not rounded summary values, select it.

**78 tests passed** from the exact committed closure source. These include the
real two-CIF timestamp regression, side-chain-only corruption (which backbone
matching alone misses), coordinate/model-version changes, malformed confidence,
missing samples and failed scheduler/assembly gates. The printed job `123` in
test logs is a mocked unit-test response, not a real submitted job.

## Source and execution separation

| Commit | Purpose |
|---|---|
| `2839a1afbcf6d07ee685f264781048d059a63d02` | strict full-CIF/all-atom/confidence checks, real fixture, independent post-auditor |
| `e61c6bdf05e7a1cfcc52aa9ab9b1cb6a6915f5fb` | preserve original fixture bytes and correct top-CIF path/hash provenance |
| `37a7c897455209c290bf0b0e05b35c10e9123d9d` | complete-40 strict certification/assembly gate; does not run a model |

Current inference execution plan SHA:
`397193cd9248d4157b836de3db6c1e3f729cc9665817b9af49117c5f59b880ea`.
Legacy scientific plan SHA:
`f2199a1f1940ff64fb75d7810b974963ed984fc293c6d3f2f0f150663b1bd615`.
These are distinct from the new auditor source hashes. Old plan/source digests
were not relabelled to pretend that new code had executed earlier predictions.

## Actual job state at the recorded snapshot

| Work | Status |
|---|---|
| Original AF3 task `1641607_0` | FAILED 1:0 preserved; existing predictions now independently certified |
| AF3 remainder array `1641712` | tasks 1–13 completed; 14–16 running; 17–39 pending |
| AF3 completed task directories | 14/40 including recovered task 0; no new failure receipts |
| New strict post-audits executed | task 0 original + recovered copy, and tasks 1–5: 7 passing receipts / 6 unique tasks |
| SaProt/ProSST `1640503` | 12/36 completed; tasks 12–17 running; 18–35 pending |
| Original analysis `1640514` | PENDING dependency; unchanged |
| Corrected analysis `1641627` | PENDING dependency; unchanged |

Completed under the earlier executor is not identical to certification under
the new stricter auditor. The remaining tasks have not all received that new
post-audit, and full 40-task assembly has not been completed.

An attempt to submit a separate audit/assembly-only dependent job was blocked by
the execution safety check **before remote execution**. It was not retried through
another channel. Read-only verification found no new closure job, no closure
submission receipt and no submission-intent file. The tested closure code is
committed but **not scheduled or executed**. Existing inference proceeds in its
already-submitted array; no training or analysis job was modified.

## Reproduce the read-only check

Use a new output path, the pinned old executor plan and the separate new auditor:

```bash
python extensions/execution_recovery_v1/semantic_audit.py \
  --plan /ssd/scxi253/frontier_multimodel_v1/execution_recovery_v1/af3_attempt2/PLAN.json \
  --plan-sha256 6db2ca822003647080561af7124795d855117a920fd634ce0392d8e995539df3 \
  --task-id 0 --allow-preserved-audit-failure \
  --output /NEW_OUTPUT_DIRECTORY/task000_strict_audit.json
```

This command does not invoke model inference. Do not overwrite an existing
certificate or run directory. CPU environment requires Biopython; tests can be
run with `python -m unittest discover -s extensions/execution_recovery_v1 -p 'test_*.py'`.

## Remaining work and limits

All 40 predictions need complete strict certificates before assembling the
combined 217-case structure authority. The full 36-task layer sweep and its
registered controls must finish before any selected-layer or paired-statistical
conclusion. No intermediate winner, fresh OOD, co-folder, affinity or new learned
rescue result was produced by this repair. No UI Max reasoning switch was made
or claimed. The review branch is separate; main and the running source snapshots
were not modified.

## Evidence

See `evidence/semantic_recovery_20260930/review/`: original failure/stdout/stderr,
original and active plans, existing controller submission receipts, strict
per-task reports, tests, final scheduler snapshot and an exact-byte import
manifest. `fixtures/real_af3_timestamp_fixture/` contains the original selected
CIF regression bytes and their hashes. Primary-source rationale is recorded in
`SEMANTIC_AUDIT_PROTOCOL.md`.
