# Antibody foundation-model scale research v2

**Frozen:** 2026-07-31 Asia/Shanghai

**Scope:** quality-preserving OAS pretraining and measured systems scaling. This
document does not authorize downstream test access or a state-of-the-art claim.

## Evidence that changes the next step

The authenticated OAS download contains 265,834,820 source rows across 6,373
data units and 120,714,440,652 compressed bytes. Full materialization and an
independent replay have now passed. Exact filtering, global deduplication, and
validation precedence produced 224,887,833 unique training records and
1,743,380 unique validation records.

The exact read-only scale audit in job `1343027` counted 25,259,993,406 unique
training residue tokens and 196,927,742 validation residue tokens across all
512 train/validation shards. Its receipt self-hash is
`be2ec8831d231a53285459b5ddf2173d7ce3133406aa9e996d2ee83906c70c99`.
The preregistered 1.3B data gate required 30,000,000,000 unique training residue
tokens and therefore failed. Repeating examples does not satisfy this gate;
the 1.3B pilot remains locked unless a separately authenticated corpus extends
the unique-token inventory.

The first loader/recovery attempt, job `1342989`, failed before training because
the original loader retained every shard array as an open memory map and
exceeded the process file-descriptor limit. This is a loader implementation
failure, not evidence of corpus corruption or model instability. The failure
stderr is preserved, and the replacement loader uses a bounded read-only mmap
cache while retaining complete array hashing and schema validation.

Replacement job `1343030` completed the uninterrupted four-update run, the
two-update checkpoint run, and the two-update resumed run on all 224,887,833
training records. Its original auditor failed because it compared the DCP
optimizer payload hash, which uses canonical parameter names, directly against
the applied runtime optimizer hash, which uses optimizer parameter IDs. A
corrected independent re-audit keeps both checks separate: saved DCP payload
equals loaded DCP payload, and applied optimizer state equals the stage-one
runtime optimizer state. It also proves identical final model, final optimizer,
scientific trace, and trainer coordinates. The corrected receipt self-hash is
`7347f5c3101f631200f09b175e574ae197a579eac2e9a479fd6790ab1e547f9f`.
This remains a non-authorizing retrospective result. A newly frozen replay was
therefore required before the recovery implementation could be promoted.

The same run measured 19.19 seconds of data preparation versus 4.05 seconds of
GPU update time across the four uninterrupted updates. The next source revision
therefore gathers each rank's complete update by shard once and then restores
the original microbatch order. This preserves membership and masking semantics
but must pass a new dynamic recovery/performance replay before promotion.

Newly frozen job `1343046` completed in 7 minutes 19 seconds with exit code
zero. Its source-bound independent auditor passed all four exact-recovery
conditions: final model equality, final optimizer equality, scientific-trace
equality excluding timing, and terminal trainer-coordinate equality. The
create-only audit file SHA-256 is
`89dd12ca7999de230642d4e7f0b6c0d0ee168286bdccc56236a089f7ac79df4b` and its
verified receipt self-hash is
`de7914b594dff66031ebd26f60a39ba747afa49b5da91d39816eb1e8b52b844e`.
The uninterrupted and resumed terminal model digest is
`d3a97e0d20bdd534fb45afd64876df8466e2b0f6a1d2928c9d619d2f507187c7`;
the corresponding optimizer digest is
`20329fb916ac11361a5b9452c496d551d1f35ffc24bc543fbea95cc0de5242ae`.

The grouped read reduced uninterrupted four-update data preparation from 19.19
to 13.82 seconds, a 27.96 percent reduction, while preserving every scientific
trace value and terminal state. The gate still spent 13.82 seconds preparing
data versus 4.05 seconds in GPU updates, so it does not authorize production.
The next frozen replay raises the soft file-descriptor limit to 4,096 and keeps
up to 1,024 authenticated training-field maps in a deterministic read-only LRU.
This is intended to remove map reopen churn; the complete array hash and schema
checks remain unchanged.

Job `1343051` executed that frozen configuration and passed the same four exact
recovery conditions. The audit file SHA-256 is
`6dac4ab3f5a1de3f9c70abdc217cdf3913aa8db46ee48ee6e74f78547f265538` and the
verified receipt self-hash is
`42ef213ad569aa8874a057613351a587787edf996d4c1b3c7a9df82854dc8b5f`.
The observed soft and hard descriptor limits were both 131,072, while the
receipt continued to bind the protocol minimum of 4,096 and cache capacity of
1,024. Four-update data preparation fell again, from 13.82 to 3.97 seconds, a
71.26 percent reduction; compared with the original 19.19-second path, the
cumulative reduction is 79.30 percent. End-to-end measured preparation plus GPU
update time fell from 23.23 to 7.93 seconds without changing the final model,
optimizer, memberships, masks, losses, gradient norms, learning rates, or
trainer coordinates. This is the promoted loader implementation for the next
real-OAS eight-GPU gate; it still does not by itself authorize production.

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

Jobs `1342810` and `1342862` each acquired one eight-RTX-5090 node and failed
before their first optimizer update. The v1 crossover requested 600 updates
while the reused scientific-parity v2 protocol allowed exactly 500. The first
extension retained independent seed 97 while the parent registered only seeds
17, 42, and 73. In both cases the runner correctly rejected the disagreement.
The immutable artifact manifests and training logs remain bound through the v2
and v3 crossover protocol chain.

The corrected execution does not weaken either protocol. A separate
non-authorizing parity extension uses registered seed 17, preserves the frozen
split, and allows 600 systems-only updates, which yields 100 excluded warmup
updates and 500 measured updates. A source-bound wrapper delegates all model,
optimizer, data, masking, and evaluation work to the unchanged historical
runner and publishes a create-only sidecar receipt. The auditor independently
binds that sidecar to the base receipt, trainer state, extension protocol,
wrapper, and unchanged base runner.

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

1. Complete the replacement one-GPU loader/recovery gate. Newly frozen jobs
   `1343046` and `1343051` passed exact recovery; the latter also passed the
   protocol-bound descriptor-cache optimization. Job `1342989` remains the
   preserved failed loader attempt, and job `1343030` remains the corrected
   retrospective audit.
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
7. Admit a 1.2B--1.4B FSDP2 pilot only after an independently authenticated
   corpus reaches the frozen 30B unique-residue-token threshold and validation
   loss or frozen transfer quality per GPU-hour improves over 650M. Parameter
   count or repeated exposure alone is not a gate.

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
