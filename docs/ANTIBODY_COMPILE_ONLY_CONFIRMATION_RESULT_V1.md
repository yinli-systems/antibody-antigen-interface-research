# Antibody compile-only confirmation result v1

**Executed:** 2026-07-31 Asia/Shanghai

**Training job:** `1343024`

**Independent audit job:** `1343025`

**Scope:** eight RTX 5090 GPUs, one allocation, two-repetition crossover,
600 updates per run, 100 excluded warmup updates, dense training-only pilot

## Decision

The regional compile-only candidate is not promoted. It passes throughput,
approximate MFU, memory, membership, replicate-stability, artifact-integrity,
and short-horizon numerical gates. It fails the preregistered long-horizon
evaluation-noninferiority and step-stability gates. Production pretraining and
16/32-GPU scaling remain unauthorized.

The audit job's exit code is `1` by design because the independent auditor writes
the create-only result and then returns nonzero whenever `gate_passed` is false.
Runtime-lock verification passed before the scientific audit ran.

## Measured result

| Mode | Useful residue tokens/s | Approximate 6N MFU | p95/p50 range | Peak allocation fraction |
|---|---:|---:|---:|---:|
| Safe eager, pooled | 150,991.14 | 0.3430 | 1.064--1.124 | 0.6661 |
| Regional compile, pooled | 202,795.10 | 0.4607 | 1.128--1.137 | 0.5993--0.6179 |

Regional compilation improved pooled throughput by `1.3431x` while reducing
peak allocated memory. Both compiled repetitions were effectively identical:
their throughput relative range was `0.0000868`, and their final model hashes
matched exactly.

## Failed gates

### Evaluation noninferiority

The frozen training-excluded panel reported a compiled-versus-eager masked
cross-entropy relative increase of `0.02030`. The paired 95% bootstrap upper
bounds were `0.03756` and `0.03734`, both above the frozen `0.01` maximum.
Masked accuracy did not degrade, but the gate requires both metrics to pass.

### Step stability

The two compiled p95/p50 step-time ratios were `1.13728` and `1.12842`, above
the frozen `1.10` maximum. One eager repetition also exceeded the threshold;
therefore the next diagnostic must attribute deterministic runtime stalls rather
than treating compilation as their sole cause.

## Integrity and claim boundary

- Audit file SHA-256:
  `06cb74083bd79eec8a5f5abefc99ebc4c2d90634d9d4cba96d1d5dd973028c4d`
- Audit self-hash:
  `9887b28baedac3ea465f4add59a5e340e4c01d4402877c0c4b0d6be65a65d317`
- Frozen protocol SHA-256:
  `1878fc468fe66543a718d81b9bbbb3c476f20e40d50230a5e402be55f4876046`
- Artifact manifest verified: `true`
- Writable paths in frozen evidence: none
- SAbDab2 held-out access: `false`
- Downstream-label access: `false`

This is a systems pilot on a dense training-only panel. It does not establish a
production-corpus loss curve, a biological transfer result, multi-node scaling,
or a state-of-the-art result.
