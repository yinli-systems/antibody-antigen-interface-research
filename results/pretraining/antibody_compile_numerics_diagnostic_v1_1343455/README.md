# Compile numerics diagnostic job 1343455

Job `1343455` completed all six registered 600-update runs on one node with
eight NVIDIA GeForce RTX 5090 GPUs. Job `1343456` then preserved the original
v1 audit failure caused by an incorrect interpretation of explicit
`gc.collect()` counters. The create-only v2 audit replayed the immutable run
with corrected Python cyclic-GC semantics and passed its evidence-integrity
checks.

The v2 audit found no configuration eligible for strict confirmation. The
fastest stable configuration was `compile_gc_off` at 219,955 useful residue
tokens/s, 0.4996 approximate 6N MFU, 1.3824x pooled-eager throughput, and a
1.0202 p95/p50 step-time ratio. Its evaluation cross-entropy increased by
2.03% relative to eager, with paired 95% bootstrap upper bounds of 3.78% and
3.76% against the two identical eager references. This exceeded the frozen
1% noninferiority limit. The accuracy-first compiled configuration increased
cross-entropy by 3.46% and did not repair the numerical gate.

No production, scale, downstream, biological, or state-of-the-art claim is
authorized by these artifacts. The held-out SAbDab2 data was not accessed.

Evidence files:

- `independent-audit-v2.json`: corrected self-hashed independent audit.
- `parent-audit-v1.err`: preserved traceback from the original GC-semantic
  interpretation.
- `parent-audit-v1.out`: runtime-manifest verification emitted by the original
  audit job.
