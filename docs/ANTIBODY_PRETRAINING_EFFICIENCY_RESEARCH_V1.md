# Antibody pretraining efficiency research v1

**Frozen:** 2026-07-29 Asia/Shanghai

**Scope:** systems research and preregistered scale gates; no downstream test
access and no production or state-of-the-art authorization

## Decision

The next engineering target is a safe 650M eight-GPU reference, followed by
regional compilation and a measured 16-GPU strong-scaling run. A 32-GPU run is
allowed only after the 16-GPU gate passes and the scheduler actually provides
four eight-GPU nodes. A 1.2B--1.4B systems pilot remains locked until the 650M
correctness, recovery, and efficiency gates pass.

This ordering separates three questions that must not be conflated:

1. Does an optimization preserve the same training computation?
2. Does it increase useful residue-token throughput with adequate memory
   headroom?
3. Does the resulting representation improve a preregistered scientific
   benchmark without downstream-test selection?

The answer to one question does not imply the others.

## Evidence from the existing 650M pilot

The historical fixed-work benchmark used eight RTX 5090 GPUs, 500 optimizer
updates, BF16 compute, and a per-GPU micro-batch of two. DDP was the fastest of
the four audited backends at 20,948.86 non-padding residue tokens/s, but median
GPU utilization was only 35%. FSDP2 used much less memory but was slower because
the small per-rank matrices did not amortize sharding communication.

An exploratory micro-batch scan was completed before this v1 gate was frozen.
It used the immutable historical source, the same seed, and the same 150 update
memberships for every candidate. Excluding 25 warmup updates, the observations
were:

| Per-GPU micro-batch | Residue tokens/s | p50 step (s) | p95/p50 | Peak memory (MiB) |
|---:|---:|---:|---:|---:|
| 4 | 39,269.69 | 0.7037 | 1.0663 | 16,491 |
| 8 | 66,462.05 | 0.4156 | 1.0691 | 17,723 |
| 16 | 103,023.82 | 0.2690 | 1.0125 | 21,419 |
| 32 | 132,873.24 | 0.2088 | 1.0210 | 29,099 |

All four scans processed a mean 27,862.84 residue tokens per update. The
micro-batch-32 scan is the fastest engineering observation, but it consumes
about 89.2% of reported GPU memory and therefore fails the frozen 25% headroom
rule. Micro-batch 16 reaches about 4.92 times the historical throughput while
using about 65.7% of memory, so it is the safe reference for the next controlled
benchmark. These 150-update runs are repeated-shard systems evidence, not OAS
validation or biological-quality evidence.

The old pilot receipt did not record the micro-batch argument inside its own
payload. The exploratory scan therefore cannot authorize production. The new
runner must bind all launch arguments and optimization switches into its
create-only receipt before any candidate can pass.

## Highest-impact implementation sequence

### 1. Batch geometry

Use the largest micro-batch that stays below 75% peak memory under checkpoint
and telemetry load. Larger per-rank matrices reduce Python launch overhead and
increase Tensor Core occupancy. Gradient accumulation still enforces the frozen
global residue-token target in production; changing the global target is not a
strong-scaling optimization.

### 2. Regional `torch.compile`

Compile each repeated Transformer block with `fullgraph=True` before applying
FSDP2, matching the ordering used by TorchTitan. Regional compilation avoids a
single large graph, allows compilation reuse across identical blocks, and keeps
FSDP boundaries outside compiled regions. PyTorch reports 10--23% MFU gains for
FSDP training from compilation in a 7B study, but this repository requires its
own measured parity and throughput result.

### 3. Kernel and optimizer path

Continue to use PyTorch scaled-dot-product attention so the runtime selects an
eligible CUDA backend. Probe fused AdamW rather than assuming compatibility
with DTensor optimizer states. Stage host tensors in pinned memory and use
nonblocking host-to-device copies, while retaining transfer time inside the
measured training step.

### 4. FSDP2 and multi-node geometry

Use DDP when the model and the frozen headroom fit on one GPU. Use FSDP2 when
capacity requires sharding. For multiple eight-GPU nodes, compare full FSDP2
with hybrid sharding that shards within a node and replicates between nodes;
the choice is made only from identical-work measurements. The short maximum
sequence length of 194 does not justify context parallelism. Tensor and
pipeline parallelism are reserved for a model that cannot meet the memory gate
with FSDP2 because they add communication and scheduling complexity.

### 5. FP8 as a separate numerical candidate

Transformer Engine supports FP8 on Ada and Blackwell hardware, and PyTorch has
documented FSDP2 plus float8 speedups. This cluster image does not currently
contain Transformer Engine or torchao, so FP8 is not silently enabled. It needs
an isolated, hash-pinned environment, an identical-work BF16 comparison, loss
and gradient parity, checkpoint recovery, and validation-curve parity before
promotion.

### 6. Data path and scientific quality

The production corpus must finish authenticated OAS admission before the run is
called real pretraining. Stable length bucketing may reorder examples within an
already frozen update but may not change update membership. Boundary-safe
packing is useful only when the attention implementation proves that examples
cannot attend across sequence boundaries.

Model quality is improved through data and objectives rather than parameter
count alone:

- phase A uses study-balanced, deduplicated unpaired OAS MLM;
- phase B continues on native heavy--light pairs with paired MLM and
  bidirectional retrieval controls;
- phase C may add train-only structure contrastive supervision and the native
  antibody--antigen residue-pair head under the independently frozen split;
- germline and region imbalance are audited, with focal or weighted objectives
  treated as preregistered ablations;
- matched partner shuffle, single-partner controls, degree-aware random
  controls, retrieval, calibration, and native contact AUPRC are required.

A larger model is promoted only when validation loss or frozen transfer per
GPU-hour improves. Repeating a limited corpus for very many iterations is not
automatically beneficial and may overfit.

## Scale and claim gates

- 8 GPUs: same-membership eager/optimized correctness, at least 500 updates,
  exact recovery, p95/p50 at most 1.10, and peak memory at most 75%.
- 16 GPUs: fixed-global-work strong-scaling efficiency at least 0.85.
- 32 GPUs: run only after the 16-GPU gate, with efficiency at least 0.80.
- 64 GPUs: locked until 32 GPUs pass; efficiency must be at least 0.70.
- 1.2B--1.4B: systems pilot only after the safe 650M candidate passes.
- production: requires complete corpus admission and the scientific promotion
  gate. No held-out SAbDab2 result is used for pretraining selection.

The current VIP allocation exposes two eight-GPU RTX 5090 nodes, not a verified
32-GPU allocation. A queued or projected 32-GPU run is never reported as a
measured 32-GPU result.

The phrase "SIR model" did not resolve to a relevant antibody foundation-model
benchmark in the literature search. The comparison target must be identified
by exact paper, checkpoint, dataset split, and metric before an "exceeds SIR"
claim can be evaluated. Until then, the comparator set is CurrAb, IgBert/IgT5,
BALM-paired, AntiBERTa2, and other methods only where identical evaluation is
possible.

## Primary sources

- PyTorch FSDP2: <https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html>
- PyTorch FSDP plus compile throughput study:
  <https://pytorch.org/blog/maximizing-training-throughput/>
- TorchTitan parallelization implementation:
  <https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize.py>
- PyTorch FSDP2 plus float8:
  <https://pytorch.org/blog/training-using-float8-fsdp2/>
- Transformer Engine user guide:
  <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/>
- Megatron Core parallelism guide:
  <https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html>
- Sequence packing without cross-example attention:
  <https://arxiv.org/abs/2107.02027>
- Protein language-model compute scaling:
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/8066ae1446b2bbccb5159587cc3b3bcc-Abstract-Conference.html>
- IgBert and IgT5:
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012646>
- Native paired-antibody language modeling:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11117052/>
- CurrAb:
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013473>
- AntiBERTa2 structure contrastive pretraining:
  <https://www.biorxiv.org/content/10.1101/2023.12.12.569610v1>
