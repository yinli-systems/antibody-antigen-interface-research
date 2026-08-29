# Residue-head and distogram development runbook

Status: two frozen development-only arrays are running on ParaCloud RTX 4090
GPUs. This document does not authorize a SAbDab2 frozen-test, external
generalization, benchmark-improvement, or A100 claim.

## Registered questions

1. Compare independent intra-chain unary residue prediction, direct noisy-OR
   marginalization of the contact matrix, and their zero-gated hybrid.
2. Compare an independent binary contact head with one categorical distogram
   whose near bins exactly define contact probability.
3. Test whether a sparse CDR-by-isolated-antigen-surface reverse-triangle
   constraint improves precision/P@K while retaining at least 95% of recall.

All comparisons use seeds `17/42/73/101/137`, the antigen-fold v3 train and
validation roles, the same reciprocal 31-partner panels, and validation-only
checkpoint selection. Every epoch records the validation-selected operating
point and the separately named raw-logit-zero operating point.

## Authorities

| Artifact | SHA-256 |
|---|---|
| structure manifest | `edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0` |
| contact manifest | `d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33` |
| ESM manifest | `9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545` |
| antigen-fold split | `fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6` |
| development selection | `2d71d43e666965b9e69cd107ff72c0e46a65b9f7651e83cf5e344bcb5b335fa8` |
| pair geometry manifest | `5924697eaffb6ae023834cd72bd179aefd95a6d8890bfca5ce7e2dd9feba41b2` |
| distogram protocol | `ef564c5329a525455536e47adfd6270762421c211a2078ce5e4a79ba85472aad` |
| distogram protocol receipt | `7ca2933dca175af2f49463d05fe4be8a90ae791edddf6da9062a9bb9fc7b944e` |
| v4 source manifest | `7a8f335baca281e0daf4020708221252cb02adf09b4a090e761907e075178195` |
| v4 source tree | `81491c299fa22337b59a22897e131e39a7e2d3faae70a3db7b51799d4692ba00` |

The pair-geometry cache contains exactly 3,114 train/validation complexes and
162,076,965 observed residue pairs. It excludes all 549 test-role complexes.
The 121 cutoff reconciliations are bounded by the registered 0.002-Angstrom
coordinate-quantization tolerance; the largest adjustment is 0.001941205
Angstrom.

## Running jobs

- Slurm `1345042`: unary-only, contact-marginal-only, and gated-hybrid;
  15 total runs, maximum five concurrent.
- Slurm `1345108`: distogram and distogram plus local triangle constraint;
  10 total runs, maximum five concurrent.
- Slurm `1345107`: completed v4 RTX 4090 BF16 smoke. Receipt SHA-256:
  `4b6a7f9daf6ff7c27b25aa60314188b5118af5e957cbb16e5be4d6b2e236d900`.

The smoke used one authenticated train complex, 6,549 candidate pairs and 44
contacts. It completed forward/backward/optimizer update with 203 finite
gradient tensors, and the distogram-to-contact projection maximum error was
exactly zero. It is engineering evidence only.

## Monitor without opening test data

```bash
squeue -j 1345042,1345108 -o "%.18i %.4t %.10M %.20R"
sacct -j 1345042,1345108 --format=JobID,State,Elapsed,ExitCode -n -P
```

Do not pass a test split, external result path, or test ensemble to either
training command. Partial output directories without a complete run receipt
are fatal and must be investigated, not silently overwritten.

For read-only, synchronized multi-seed metric inspection, use
`scripts/summarize_framecontact_live_array.py`. Version 2 reports both the
validation-selected operating point and raw-logit-zero precision, recall, F1,
MCC and predicted-positive ratio; contact P@K/Recall@K; epitope/paratope
metrics; both partner-retrieval directions; and descriptive epoch slopes. The
`--compact` view omits per-seed metric bodies while retaining epoch coverage,
the latest synchronized epoch and trajectory diagnostics. It never chooses a
checkpoint.

```bash
python3 scripts/summarize_framecontact_live_array.py \
  --log-root /path/to/slurm-logs \
  --log-prefix fc-res-head \
  --job-id 1345042 \
  --conditions framecontact_unary_only framecontact_contact_marginal \
    framecontact_contact_hybrid \
  --seeds 17 42 73 101 137 \
  --minimum-seeds 5 \
  --compact
```

The “best epoch” fields in this live view are explicitly descriptive. Only the
registered post-run selector may freeze a checkpoint.

When two conditions live in different arrays or progress at different rates,
use `scripts/compare_framecontact_live_conditions.py`. It chooses the latest
epoch present for every declared seed on both sides and reports per-seed paired
`right - left` deltas. This prevents an epoch-12 versus epoch-9 snapshot from
being mistaken for an ablation result. The current unary-versus-distogram
epoch-10 receipt is
`receipts/FRAMECONTACT_UNARY_VS_DISTOGRAM_MATCHED_EPOCH10.json`, SHA256
`43a4d4904e9fa32a61ae7a86bd216966a8bc548be6f0eb478dba2eb357f22c4e`.

## Aggregate only after all 25 receipts exist

```bash
base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source="$base/residue-head-source-v4"
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3

PYTHONPATH="$source/src" "$python_bin" \
  "$source/scripts/aggregate_framecontact_head_distogram_multiseed.py" \
  --binary-run-root "$base/runs/residue-head-exploratory-v1" \
  --distogram-run-root "$base/runs/distogram-exploratory-v1" \
  --binary-protocol "$source/protocols/framecontact_residue_head_exploratory_v1.json" \
  --binary-protocol-receipt "$source/protocols/FRAMECONTACT_RESIDUE_HEAD_EXPLORATORY_V1_RECEIPT.json" \
  --distogram-protocol "$source/protocols/framecontact_distogram_exploratory_v1.json" \
  --distogram-protocol-receipt "$source/protocols/FRAMECONTACT_DISTOGRAM_EXPLORATORY_V1_RECEIPT.json" \
  --output "$base/runs/FRAMECONTACT_HEAD_DISTOGRAM_5SEED_SUMMARY.json"
```

The aggregator authenticates every result and checkpoint digest, enforces
shared data authorities, reports paired seed deltas and complete five-seed
epoch trajectories, and evaluates the pre-registered promotion gates. A failed
gate must be reported as a negative result.

## Current evidence boundary

An interrupted all-cohort preprocessing attempt read pair-label artifacts for
51 current test-role complexes before cancellation. It published no cache and
read no model prediction or test metric. Consequently, the current SAbDab2 test
may be described only with that disclosed preprocessing access; it cannot
support a procedural never-read or confirmatory claim. Strict confirmation
requires a newly isolated external/temporal holdout.
