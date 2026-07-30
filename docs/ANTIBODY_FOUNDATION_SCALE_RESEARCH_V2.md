# Antibody foundation-model scale research v2

**Frozen:** 2026-07-31 Asia/Shanghai

**Scope:** quality-preserving OAS pretraining and measured systems scaling. This
document does not authorize downstream test access or a state-of-the-art claim.

## Evidence that changes the next step

The authenticated OAS download contains 265,834,820 source rows across 6,373
data units and 120,714,440,652 compressed bytes. The complete production corpus
has not yet passed row filtering, global exact deduplication, validation
precedence, independent replay, or a loader/recovery gate. Large-scale training
must therefore begin with corpus materialization rather than a longer repeat of
the 8,112-sequence dense pilot.

The preregistered eight-RTX-5090 parity experiment completed nine 500-update
runs for seeds 17, 42, and 73. Regional compilation, fused AdamW, pinned staging,
and stable padding buckets were effective. BF16 gradient compression reached
156,227 residue tokens/s, 1.505 times eager, but failed the frozen final-outcome
non-inferiority gate. The PyTorch BF16 DDP hook is documented as experimental.
It is excluded from the quality-preserving production candidate rather than
being promoted from throughput alone.

## Frozen systems direction

The production candidate keeps BF16 model compute with FP32 optimizer state and
FP32 gradient reduction. It uses PyTorch SDPA, regional full-graph compilation
of repeated Transformer blocks, fused AdamW, pinned nonblocking staging, stable
length buckets, gradient-as-bucket-view, and DDP accumulation with `no_sync` on
all non-final microsteps. The production token target amortizes one FP32
all-reduce over many microsteps, so the failed BF16 communication shortcut is
not required to achieve high throughput.

DDP remains the single-node 650M reference because the model fits with at least
25 percent memory headroom. FSDP2 is the capacity path for the later 1.3B pilot
and the multi-node comparison. Context parallelism is excluded because the
maximum sequence length is 194. Tensor and pipeline parallelism are excluded
until a model cannot fit under FSDP2; adding them to a short-sequence 650M model
would increase communication without solving a capacity problem.

## MFU and efficiency semantics

NVIDIA specifies 209.5 dense BF16 Tensor TFLOPS with FP32 accumulation for one
GeForce RTX 5090. The primary hardware denominator is therefore 209.5e12 times
the observed GPU count. Approximate model FLOPs use the conventional `6 * N * T`
training estimate, where `N` is the exact parameter count and `T` is the number
of non-padding residue tokens. The resulting value is named `approximate_6n_mfu`
and must never be presented as a profiler-exact FLOP count.

Every performance receipt also reports useful residue tokens/s, sequence
tokens/s/GPU, p50 and p95 step time, p95/p50, peak memory fraction, GPU power,
GPU utilization, data-preparation time, communication time where observable,
and GPU-hours. A speedup is valid only for identical membership, global work,
precision semantics, and hardware.

## Production-batch crossover correction

Job `1342810` acquired one eight-RTX-5090 node and failed before its first
optimizer update. The v1 crossover requested 600 updates while the reused
scientific-parity v2 protocol allowed exactly 500; the runner correctly
rejected the disagreement. Its immutable artifact manifest and training log
remain bound in the v2 crossover protocol.

The corrected execution does not weaken either protocol. A separate
non-authorizing parity extension preserves the frozen split and seed while
allowing 600 systems-only updates, which yields 100 excluded warmup updates and
500 measured updates. A source-bound wrapper delegates all model, optimizer,
data, masking, and evaluation work to the unchanged historical runner and
publishes a create-only sidecar receipt. The auditor independently binds that
sidecar to the base receipt, trainer state, extension protocol, wrapper, and
unchanged base runner.

## Data and scientific quality

Phase A trains on the globally deduplicated, study-audited OAS unpaired corpus.
Validation studies have unconditional precedence: any exact validation sequence
removes all identical training rows. Sampling and learning-rate schedules are
indexed by consumed residue tokens rather than epochs.

Phase B may continue from the frozen Phase-A checkpoint on independently
admitted native heavy-light pairs. Candidate objectives are paired MLM and
bidirectional heavy-light retrieval. A paired objective is promoted only when
matched-partner shuffling degrades retrieval and the gain is consistent across
predeclared seeds.

Downstream antibody-antigen work remains separate. SAbDab2 outer/test labels are
not model-selection inputs for foundation pretraining. Native pair-contact AUPRC,
partner shuffle, single-partner controls, degree-aware random controls, and
bidirectional retrieval are evaluated only under their independently frozen
split protocol.

## Scale ladder

1. Complete OAS materialization, independent replay, and one-GPU loader/recovery.
2. Run a fresh three-seed eight-GPU 650M gate with production batch geometry,
   FP32 gradient reduction, at least 500 measured updates, and a training-excluded
   OAS validation panel.
3. Prove exact checkpoint/restart trace replay at the selected world size.
4. Measure fixed-global-work 8-to-16 strong scaling. Require efficiency at least
   0.85 and no quality or recovery regression.
5. Measure 16-to-32 scaling only after stage 4. Require 32-GPU efficiency at
   least 0.80, p95/p50 at most 1.20, and identical scientific direction.
6. Train the 650M production run to the frozen consumed-token budget only after
   all preceding gates pass.
7. Admit a 1.2B--1.4B FSDP2 pilot only if validation loss or frozen transfer
   quality per GPU-hour improves over 650M. Parameter count alone is not a gate.

The 64-GPU stage remains locked until 32 GPUs pass with efficiency at least
0.70. A queued allocation, scheduler estimate, synthetic input, or repeated
pilot shard is never reported as measured production training.

## Primary sources

- NVIDIA RTX Blackwell architecture and RTX 5090 dense BF16 peak:
  <https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf>
- PyTorch FSDP and `torch.compile` MFU study:
  <https://pytorch.org/blog/maximizing-training-throughput/>
- PyTorch DDP communication hooks:
  <https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html>
- PyTorch TorchTitan pretraining platform:
  <https://github.com/pytorch/torchtitan>
- PyTorch float8 pretraining guidance:
  <https://docs.pytorch.org/ao/stable/eager_tutorials/pretraining.html>
- Megatron Core parallelism guide:
  <https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html>
- Large-scale paired antibody language models:
  <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012646>
