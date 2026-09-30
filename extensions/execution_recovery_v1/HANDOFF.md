# Resume safely from actual execution state

Read README.md and the timestamped final snapshot first. Do not resubmit jobs
1640503, 1640514, 1641607 or 1641627. Inspect squeue, sacct and existing receipts.

## AF3 recovery

Root `/ssd/scxi253/frontier_multimodel_v1/execution_recovery_v1`.
Source `source_0bd7366/extensions/execution_recovery_v1/af3.py`.
Plan `af3_attempt2/PLAN.json`, SHA
`6db2ca822003647080561af7124795d855117a920fd634ce0392d8e995539df3`.
Task 0 job is 1641607; its submitted intent is already present. Pending Priority
is not a failed job and must not be duplicated. On failure preserve the source,
plan, output directory and failure receipt; do not edit a pinned source in place.
Only after task_000/COMPLETE.json passes and its scheduler allocation terminates
may task IDs 1–39 be submitted through the controller. The controller itself
rejects reused tasks and ambiguous earlier submission intents.

Do not move or rewrite the original 40-chain inputs. Do not silently reduce
seeds, samples, chain coverage or runtime checks to get a PASS.

## Layer-sweep aggregation

Original source and training array are unchanged. Independently scheduled job
1641627 waits on afterok:1640503,afterany:1640514. It imports the original
aggregator but reads the actual mean_reciprocal_rank field. The old analyzer is
expected to expose its schema error; preserve that record. The corrected job
must still admit all 36 tasks and 217 cases; incomplete arrays remain HOLD.

Original plan SHA:
`24d6c6cfb2cee44b9cc289ac8721b3ad183321db11a059c14b29c878dd7cea12`.
Corrected analysis authority SHA:
`146222211e5a88d789ba61174103db3a88e34e6ef944978295d94a149d1ed505`.

## New evidence versus implementation

49 individual tests and one synthetic full pipeline passed. These are engineering
checks, not additional independent biological samples. The latest preserved
snapshot contains 12/36 completed formal training tasks. Do not select a layer
without all registered controls, call an unknown pairing a non-binder, or reopen
closed test/OOD data. No autonomous assistant follow-up is configured.
