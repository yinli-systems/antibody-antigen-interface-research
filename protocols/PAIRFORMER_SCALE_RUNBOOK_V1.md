# PAIRFormer scale runbook v1

**Status:** prospective and blocked at Gate 0

**Date:** 2026-07-25

**Maximum planned synchronous ceiling under the current QoS:** 8 nodes / 64 GPUs

This runbook defines how large-scale training may be measured and authorized.
It does not authorize contact training, model selection, outer-fold inference,
or external-test access. Metric thresholds in this document are prospective
go/no-go rules, not achieved results.

## Workload semantics

The data layers must remain distinct:

- OAS contains more than one billion antibody repertoire sequences according
  to its current official landing page and is distributed under CC BY 4.0.
  It may support sequence pretraining, but it is not an antibody--antigen
  interaction dataset. The official headline count is not an admitted or
  consumed training count.
- AVIDa-hIL6 and AVIDa-SARS-CoV-2 contain experimentally labeled interaction
  pairs and may support binding and mutation-specificity objectives. Their
  dataset licenses are CC BY-NC 4.0, independently of the MIT licenses on the
  associated code repositories. AVIDa-hIL6 reports 573,891 labeled pairs and
  AVIDa-SARS-CoV-2 reports 77,003; they are distinct assay systems and may not
  be represented as one homogeneous benchmark.
- Structural coordinates and derived geometry require separate provenance and
  license receipts. No explicit dataset license was located on the reviewed
  SAbDab2 About or Privacy pages, so SAbDab2-added annotations and packaged
  downloads remain blocked until an explicit reuse receipt is obtained.
  Re-derived annotations from wwPDB CC0 archive files must preserve their own
  code, assembly, chain-mapping, and geometry manifests.
- Unknown or synthetically mismatched partners are ranking controls. They may
  not be represented as measured non-binders.

Every reported scale must include raw, QC-passing, deduplicated, admitted, and
actually consumed counts, plus admitted residue tokens and candidate residue
pairs. A public source's headline size is not a training count.

## Preconditions

The scale ladder may begin only after externally anchored receipts pass for all
six Gate 0 domains:

1. data and license admission;
2. the exact joint three-outer/nine-inner split;
3. an independent leakage audit;
4. fold-local cache equality;
5. the frozen runtime and software lock;
6. immutable manifests.

The Gate 0 decision may authorize train-only engineering and inner-fold model
selection. It can never authorize outer-fold or external-test evaluation.

Before each scale stage, freeze and hash:

- source, container, Python, PyTorch, CUDA, NCCL, driver, and kernel versions;
- model architecture, parameter count, optimizer, scheduler, precision policy,
  global effective batch, accumulation, clipping, and checkpoint policy;
- source registry, admission receipt, split assignment, fold allowlist,
  sampler, feature cache, hard-negative registry, and label schema;
- all random seeds and the parent checkpoint;
- Slurm allocation, process group, rank mapping, GPU--NIC topology, and
  collective policy.

## Correct global objective

For a variable-size effective batch, let rank \(r\) contribute a local
numerator \(N_r\) and valid-complex denominator \(D_r\). The intended global
mean is

\[
L = \frac{\sum_r N_r}{\sum_r D_r}.
\]

Because DistributedDataParallel averages gradients across its initial world
size \(W\), each rank must backpropagate

\[
L_r^{\mathrm{backward}} =
W \frac{N_r}{\sum_j D_j}.
\]

The denominator uses a SUM collective. The authority record must bind the
world size, every rank's local packet, process group, backend, reduction
operation, DDP averaging policy, and absence or exact identity of a custom
communication hook. Missing, duplicate, replayed, empty, or wrong-world
packets fail closed.

This derivation assumes static process-group membership and DDP's default
division by the initial world size. A membership change invalidates the
in-progress optimizer step and requires recovery from the preceding verified
checkpoint. A custom communication hook or a different reduction policy
requires a new parity proof; the DDP formula must not be assumed for FSDP2.

For DDP gradient accumulation, non-final microbatches use `no_sync()` around
both forward and backward. When a `GradScaler` is enabled, its scale remains
constant for the complete effective-batch window; `scaler.step(optimizer)` and
`scaler.update()` occur only at the window boundary. BF16 autocast does not by
itself imply that a `GradScaler` is present.

The fixed global correctness batch must contain enough valid supervised units
for every rank under the registered sampler. If it cannot do so, that world
size fails the strong-scaling design; fake supervision or duplicated samples
may not be added to keep ranks nonempty.

## Stage 0 -- one-node correctness

Run on 8 GPUs with a fixed global effective batch:

- capture `nvidia-smi topo -m`, PCIe peer-access, GPU--NIC affinity, and Slurm
  rank placement;
- run all-reduce at the actual DDP gradient-bucket sizes; if FSDP2 is proposed,
  also run all-gather and reduce-scatter at its actual parameter- and
  gradient-shard sizes;
- compare one FP32 distributed optimizer step with a deterministic
  single-process reference on a reduced but code-path-equivalent fixture if the
  production configuration cannot fit that reference; a reduced fixture
  validates the objective and communication mechanics, not full-model
  numerical equivalence;
- run at least 500 warmed optimizer steps;
- interrupt after a frozen step, restore the complete checkpoint, and compare
  the next step with an uninterrupted reference.

Proceed only if:

- three repeated collective runs have no mismatch, hang, or silent fallback;
- every rank has the intended source, split, batch, and model hashes;
- loss and gradients satisfy the preregistered numerical tolerance;
- there is no NaN, Inf, data duplication, or data omission;
- checkpoint continuation reproduces the required next-step state;
- peak GPU memory leaves at least 15% project-defined safety headroom.

## Stage 1 -- four-node scale

Run on 32 GPUs with the same global effective batch and learning-rate policy as
the 8-GPU correctness run. Compare the frozen common optimizer-step prefix,
then measure at least 1,000 warmed steps.

Report:

- useful admitted residue tokens and candidate pairs per second;
- median and p95 step time;
- compute, collective, data-wait, and checkpoint fractions;
- GPU and host peak memory;
- checkpoint write and restore latency;
- failed or retried steps and jobs.

Define strong-scaling efficiency relative to 8 GPUs as

\[
E_N =
\frac{\mathrm{throughput}_N}
{(N/8)\,\mathrm{throughput}_8}.
\]

The prospective 32-GPU go targets are:

- \(E_{32} \ge 0.80\);
- p95 / median step time no greater than 1.15;
- no rank divergence or collective error;
- loss and gradient parity with the 8-GPU reference within the frozen
  tolerance.

These values are project targets, not observed measurements or vendor
guarantees.

## Stage 2 -- eight-node canary

Run on 64 GPUs only after Stage 1 passes:

1. repeat all three NCCL collectives with 64 ranks;
2. verify the intended PCIe/RDMA/network route and record any fallback;
3. execute at least 2,000 warmed optimizer steps, including one complete
   checkpoint/interruption/restart cycle;
4. compare the preregistered train-only canary objective with the matched
   8- and 32-GPU step prefixes.

The prospective 64-GPU go targets are:

- \(E_{64} \ge 0.70\);
- p95 / median step time no greater than 1.20;
- zero hang, NaN, OOM, sample duplication, or omission;
- exact recovery of sampler cursor, RNG, optimizer, scheduler, model,
  checkpoint lineage, and scaler state when a scaler is present;
- no systematic train-only canary regression relative to the smaller
  correctness runs.

If the 64-GPU stage fails, do not force a synchronous run for presentation
value. Route the allocation into independent 8-GPU pods for folds, seeds,
hard-negative mining, data processing, and ablations, and report the measured
reason for that decision.

## Stage 3 -- production training

Production may start only after the systems canary and the scientific 8-GPU
pilot pass. The pilot must show, on inner development data only:

- pair macro AUPRC above the strongest separable marginal baseline;
- a positive matched-partner-shuffle effect, defined as
  \(\Delta=\mathrm{AUPRC}_{cognate}-\mathrm{AUPRC}_{shuffled}\), with its
  paired 95% confidence-interval lower bound above zero;
- better cognate retrieval than the marginal model;
- no confusion between measured non-binders and unknown controls;
- gains that are not explained only by pooled micro aggregation or a small
  number of large complexes.

Use one preregistered production recipe. For nested cross-validation, train an
independent model for each outer training partition, and select it using only
that partition's associated inner folds. "One recipe" does not mean one model
trained across an outer holdout. Training logs may inspect only train and
permitted inner-validation material. After all required fold-specific models
are frozen:

1. freeze checkpoint selection and calibration;
2. freeze all prediction and evaluation code;
3. request a separate, externally anchored one-shot outer-evaluation
   authorization;
4. if authorized, score each outer fold once under the registered policy and
   make the resulting generalization estimate immutable;
5. do not use outer-fold results to select, tune, or retrain the configuration;
6. train any final all-development-data model only from a recipe frozen by the
   nested process, without using outer performance to alter that recipe;
7. access an allowed external or chronological split only under another
   separate one-shot authorization.

## DDP versus FSDP2

Use BF16 DDP when parameters, gradients, optimizer state, activations, and
temporary buffers fit with the required memory headroom. DDP does not combine
device memory into a shared pool.

Use FSDP2 only after a controlled memory measurement shows that DDP cannot meet
the frozen headroom, or as a separately preregistered scale study. An OOM is
not required. Apply `fully_shard` bottom-up and initialize the optimizer after
sharding. Use a distributed-checkpoint path whose model and optimizer state
has passed same-world-size restart and any claimed cross-world-size resharding
test. Before promotion, verify:

- identical global-objective semantics;
- optimizer and checkpoint parity;
- all-gather and reduce-scatter performance at real message sizes;
- final inner-development metric and calibration parity;
- a net throughput or model-capacity benefit after communication cost.

FP8 is not a default. It requires a separate loss, gradient, calibration, and
scientific-metric parity gate against BF16.

## Required immutable artifacts

Each stage must publish:

- source/runtime/container/data/split/cache/model/optimizer hashes;
- allocation, host, GPU, rank, topology, and environment manifests;
- exact command and Slurm script;
- per-rank source and batch receipts;
- raw and summarized communication measurements;
- step-time, throughput, memory, and utilization traces;
- checkpoint generation manifests and continuation receipts;
- all failures, retries, amendments, and decisions;
- an explicit statement that no outer or external test was accessed.

No efficiency or scientific claim is valid without these artifacts.

## Current evidence boundary

The cluster inventory contains 12 eight-GPU RTX 5090 nodes, but the current
single-job QoS limit is eight nodes, so the designed synchronous ceiling is
64 GPUs. RTX 5090 has 32 GB GDDR7 and no NVLink. The project-local evidence
record reports that diagnostic jobs `1313384` through `1313389` completed
correct 64-rank all-reduce. It records a 64 MiB median latency of 38.44 ms with
3.4367 GB/s bus bandwidth, and a 512 MiB median of 861.49 ms, p95 of
1,309.65 ms, and bus bandwidth of 1.2269 GB/s. These are topology diagnostics
with material large-message jitter, not a completed 64-GPU model-training run.
The record is `protocols/pairformer_gate_manifest_v1.json`; it must itself be
externally anchored before it can authorize a later action.

The project has not yet completed Gate 0, the 8-GPU PAIRFormer scientific
pilot, the 32/64-GPU canaries, or a production PAIRFormer run. It therefore has
no new large-scale training efficiency, GPU-hour, pair-AUPRC, docking, or
world-leading result to report.

## Primary references

- OAS: <https://opig.stats.ox.ac.uk/webapps/oas/>
- OAS bulk-download documentation:
  <https://opig.stats.ox.ac.uk/webapps/oas/documentation/>
- AVIDa-hIL6: <https://github.com/cognano/AVIDa-hIL6>
- AVIDa-SARS-CoV-2: <https://github.com/cognano/AVIDa-SARS-CoV-2>
- AVIDa-SARS-CoV-2 dataset page:
  <https://cognanous.com/datasets/avida-sars-cov-2>
- SAbDab2 About: <https://sabdab.opig.stats.ox.ac.uk/about>
- SAbDab2 Privacy: <https://sabdab.opig.stats.ox.ac.uk/privacy>
- wwPDB usage policy: <https://www.wwpdb.org/about/usage-policies>
- PyTorch DistributedDataParallel:
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>
- PyTorch AMP accumulation:
  <https://docs.pytorch.org/docs/stable/notes/amp_examples.html>
- PyTorch FSDP2:
  <https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html>
- PyTorch Distributed Checkpoint:
  <https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html>
- NVIDIA NCCL tests: <https://github.com/NVIDIA/nccl-tests>
- NVIDIA NCCL troubleshooting:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html>
- NVIDIA RTX 5090 specifications:
  <https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/>
