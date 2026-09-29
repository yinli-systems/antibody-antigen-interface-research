# Execution — 2026-09-30

New branch `kevin/correspondence-upgrade-20260930` at implementation commit
`3d9cb70`, based on frozen `55cfd137`. No frozen source, protocol, result or
checkpoint was changed. This is a post-freeze exploratory extension.

## Verified completed

- Standalone extension contract suite: **34 tests passed** on the cluster.
- Slurm smoke **1639862**: **COMPLETED, exit 0:0**, RTX 4090.
- Three validation complexes evaluated across all 12 perturbation conditions.
- Identity repeat maximum probability difference: **0**.
- Known-inverse restoration maximum probability difference: **0**.
- Input plan: **217 validation complexes, 1,045 SHA-bound input files, 9 tasks**.
- Plan SHA-256: `98aa7753dba0cbcb4c4f769cc2ed630cb7c0a43d2e77f5813644947c962f868f`.

## Submitted, not yet a completed scientific result

Array **1639865**, task indices 0–8, max 3 concurrent single-4090 allocations,
45-minute wall limit per task. At the first post-submission check, tasks 0–2
were RUNNING and 3–8 were PENDING at the concurrency limit.

The full design is three existing encoders × three existing training seeds ×
12 evaluation-time conditions, on 217 development-validation complexes.
No new training, new test read, SOTA claim or metric improvement is asserted.
The retrieval component is a bounded matched-panel diagnostic, not a completed
large all-vs-all specificity benchmark and not a verified non-binder assay.

## Evidence

`evidence/attempt1/` contains the frozen plan, its external digest, job IDs,
standalone test transcript, engineering smoke result and timestamped start
snapshot. Later task results must be matched to the exact plan before aggregation.

## Remaining program

See the explicit NOT-completed matrix in `README.md`. Modern encoder extraction,
learned rescue, new OOD admission, new baseline reproduction, co-folder comparison
and affinity transfer have not been run by this extension. Existing PCGrad,
GradNorm and the older corruption ladder were not counted as new implementations.
