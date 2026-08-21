# Cluster activity snapshot — 2026-08-21

This is a point-in-time scheduler snapshot taken at
`2026-08-21T18:17:16+08:00`. It is not a terminal training result or a guarantee
that the same jobs remain active after the timestamp.

| Workstream | Running | Pending | Requested GPUs currently active | Interpretation |
|---|---:|---:|---:|---|
| MIPE Appendix `lr=1e-4` | 5 fold jobs | 5 continuation jobs + 1 final evaluation | 5 | Formal five-fold training is in progress; no terminal metrics exist |
| AsEP/SAbDab2 partner-negative ablation | 10 array tasks | 0 | 10 | Five no-negative seeds per dataset are running concurrently; test remains sealed |
| Antibody/OAS pretraining | 1 segment | 0 | 4 | Exact 4-GPU recovery passed and the first audited 30B-token campaign segment is running |

Each running MIPE fold requests one `gpu_4090`; its continuation is
dependency-gated, and the final evaluation is dependency-gated on all five
continuations.

The negative-sample array uses one RTX 4090 and six CPUs per task. At the
snapshot, the five AsEP tasks had completed six or seven epochs without stderr;
the larger SAbDab2 tasks had not yet completed their first long epoch.

The pretraining segment uses four RTX 4090 GPUs and 24 CPUs. Its prerequisite
continuous-versus-resumed recovery audit passed before production submission.
The frozen chain performs checkpoint evaluation and an immutable segment audit
before authorizing a successor. At this snapshot, 65 of 248 `gpu_4090` devices
were reported available cluster-wide.

Scheduler IDs, account names, host names, and absolute cluster paths are
intentionally omitted. The machine-readable snapshot is
[`CLUSTER_ACTIVITY_SNAPSHOT_20260821.json`](CLUSTER_ACTIVITY_SNAPSHOT_20260821.json).
