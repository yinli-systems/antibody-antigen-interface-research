# Antibody foundation pretraining protocol v1

**Status:** authorized, data admission and systems validation in progress

**Authorization date:** 2026-07-28

**Track:** independent of the frozen SAbDab2 E1/E2 and V6 split tracks

## Purpose and claim boundary

This protocol authorizes reproducible antibody language-model pretraining from
public OAS repertoire sequences and the engineering needed to run it with DDP,
FSDP2, and DeepSpeed ZeRO. It does not authorize reading a SAbDab2 held-out
split, changing a frozen downstream split, selecting a model on downstream
test performance, or claiming partner specificity, biological superiority, or
state of the art.

OAS is an antibody repertoire resource, not an antibody--antigen interaction
dataset. Pretraining on OAS can teach antibody sequence and native heavy--light
chain regularities. It cannot by itself prove that a model recognizes the
correct antigen partner.

The existing `frontier_model_contract_v1` remains historically correct: it did
not authorize billion-scale pretraining and its finite-cache sampler does not
support that workload. This new numbered protocol supplies a separate user
authorization and requires a new sharded streaming data path. It does not
rewrite the old contract.

## Source admission

The only admitted production source in phase A is the official Observed
Antibody Space service under CC BY 4.0:

- <https://opig.stats.ox.ac.uk/webapps/oas/>
- <https://opig.stats.ox.ac.uk/webapps/oas/documentation/>
- <https://opig.stats.ox.ac.uk/webapps/oas/documentation_paired/>

Every source data unit must be discovered from a preserved raw response from
the official service. The authority manifest records the request method and
URL, response SHA-256, canonical URL, study, filename, source class, and any
metadata present in the response. Downloaded bytes are admitted only after
gzip integrity validation and SHA-256 calculation. Headline database counts
must never be reported as downloaded, admitted, or consumed counts.

Raw OAS files, derived sequence corpora, and model weights remain outside Git.
Only manifests, code, aggregate receipts, and small synthetic fixtures may be
committed. Attribution and the source license must accompany any derived model
release.

## Data quality and isolation

The unpaired phase admits amino-acid variable-region records only when all of
the following hold:

1. the record is productive and in frame;
2. it has no stop codon and no failed ANARCI status;
3. its chain is heavy, kappa, or lambda and agrees with the sequence locus;
4. the canonical amino-acid sequence contains only the 20 standard residues
   plus an explicitly counted unknown residue;
5. its length is within the frozen model boundary;
6. exact duplicates are assigned deterministically to one canonical record.

Near-duplicate clustering, study, subject, species, chain, germline, disease,
and vaccine metadata are retained for audits. Validation is held out by whole
study and subject where identifiers exist; a sequence from a held-out group
may not enter training through another data unit. Validation membership is
selected from metadata and sequence hashes before training loss is observed.

No SAbDab2 held-out sequence, label, prediction, or metric may be opened to
construct this pretraining corpus. Consequently, overlap with unrevealed
downstream test material cannot be claimed absent. A later downstream report
must disclose pretraining exposure and use an independently frozen
decontamination or chronological confirmatory track for a clean claim.

## Model family

The primary model is a bidirectional encoder because masked language modeling
has the closest semantics to residue representation and contact-oriented
downstream transfer. The frozen architectural family contains:

- residue-level tokenization with no learned subword vocabulary;
- separate heavy, kappa, lambda, and paired-chain type embeddings;
- pre-normalized Transformer blocks;
- rotary sequence positions with an independent position origin for each
  chain in a paired example;
- exact scaled-dot-product attention using the best audited PyTorch CUDA
  kernel available at runtime;
- SwiGLU feed-forward blocks and RMSNorm;
- a tied token embedding and masked-token output projection;
- zero training dropout by default for the billion-sequence regime;
- no SAbDab2 label, antigen identity, affinity, contact, or downstream score as
  a pretraining input.

The correctness model is approximately 35M parameters. The first production
candidate targets approximately 650M parameters. An exact source-generated
parameter receipt is mandatory; approximate names are never reported as exact
counts. A 150M or 300M candidate may be promoted instead when a preregistered
compute-frontier pilot shows better validation loss or transfer per GPU-hour.
Increasing parameter count is not itself a promotion criterion.

## Objectives

### Phase A: unpaired repertoire pretraining

The primary objective is token-level masked language modeling. Mask selection,
replacement probabilities, valid-token denominator, and random stream are
fixed in the machine-readable protocol. Loss is the global sum of masked-token
negative log likelihood divided by the global number of selected valid
targets. Padding, separators, and metadata tokens never enter the denominator.

An antibody-region-aware mask is an ablation, not an undisclosed replacement
for uniform MLM. It may be promoted only by held-out OAS validation and frozen
sequence-only probes, never by SAbDab2 test performance.

### Phase B: native heavy--light continuation

Phase B begins from the frozen phase-A checkpoint and uses official native
paired OAS records. It combines MLM with bidirectional heavy--light retrieval.
Negative chains are matched within study and coarse germline/length strata
when possible to reduce study and composition shortcuts. Randomly shuffled
chains are controls, not measured non-pairs. Cross-chain objectives must be
reported separately from unpaired MLM.

### Phase C: downstream interaction modeling

Antibody--antigen contact or retrieval training is not foundation pretraining
under this protocol. It remains governed by its own frozen split and evaluation
authorization. Phase-C results may compare frozen and fine-tuned phase-A/B
encoders only after the downstream gate permits that comparison.

## Optimization contract

- optimizer: AdamW;
- precision: BF16 parameters for compute, FP32 optimizer state unless a
  separately verified optimizer is promoted;
- schedule: token-count-based warmup followed by cosine decay;
- batching: a fixed global count of admitted residue tokens, with
  length-bucketing or boundary-safe packing;
- attention: examples must never attend across a packed sequence boundary;
- accumulation: one optimizer update only after the frozen global token
  target; partial final windows are rejected or explicitly masked;
- clipping: global gradient norm with distributed semantics verified against a
  single-process reference;
- randomness: counter-derived streams for source order, record order, masking,
  replacement, and model stochasticity;
- monitoring: train loss, held-out loss, perplexity, masked-token accuracy,
  useful tokens/s, padding fraction, data wait, step-time quantiles, GPU memory,
  GPU utilization, communication time, and checkpoint latency.

Training-token count, rather than epochs over a mutable upstream service, is
the authoritative progress coordinate. Repeated tokens and repeated source
records are counted explicitly.

## Distributed backends

DDP is the numerical reference when the correctness model fits. FSDP2 is the
primary production backend. `fully_shard` is applied bottom-up to Transformer
blocks and then to the root module; the optimizer is created after sharding.
The checkpoint path uses PyTorch Distributed Checkpoint and must pass
same-world-size recovery before any cross-world-size claim.

DeepSpeed ZeRO-2 and ZeRO-3 form an independent backend comparison. ZeRO-3 is
not combined with FSDP2 in one process. CPU or NVMe offload is disabled unless
an in-memory candidate fails the frozen headroom gate, because capacity without
throughput is not an efficiency result.

Both backends must consume identical canonical batches and match the reference
loss and update within frozen BF16/FP32 tolerances. A backend may be selected
for production only from correctness, throughput, memory, stability, and
recovery evidence; downstream biological metrics are not used for that choice.

## Scale ladder

1. **One GPU:** FP32 reduced-model reference, BF16 parity, deterministic data
   and masking replay, and interrupt/resume equivalence.
2. **Eight GPUs:** at least 500 warmed optimizer steps for DDP where it fits,
   FSDP2, and DeepSpeed; record real token throughput, memory, communication,
   data wait, and checkpoint recovery.
3. **Thirty-two GPUs:** proceed only when the selected eight-GPU backend has no
   correctness failure and fixed-work efficiency projects at least 25% resource
   headroom. Require strong-scaling efficiency of at least 0.80 relative to the
   eight-GPU result.
4. **Sixty-four GPUs:** proceed only after the 32-GPU gate. Require efficiency
   of at least 0.70, p95/median step time no greater than 1.20, a complete
   interruption/recovery cycle, and no sample omission or duplication.
5. **Production:** use the largest scale that passed. If synchronous scaling
   fails, use independent eight-GPU data-parallel pods and report the measured
   limitation instead of claiming a failed large run.

Every scale uses the same fixed global work for efficiency measurement. A
larger global batch is a separate optimization experiment and cannot be called
strong scaling.

## Promotion and stopping

The production architecture is selected from OAS held-out loss, masked-token
accuracy, paired retrieval in phase B, stability, and GPU-hour efficiency.
Selection never reads SAbDab2 held-out performance. The token budget and
validation cadence are frozen before a production run. A run may stop early
only for a preregistered numerical, data-integrity, infrastructure, or
compute-frontier condition; all stopped runs remain in the ledger.

No model is described as best, leading, or state of the art without an
identical benchmark and uncertainty-aware comparison. The strongest valid
outcome may be a negative scaling or transfer result.

## Required artifacts

Each admitted run preserves:

- protocol, source, environment, container, model, optimizer, scheduler, data,
  and tokenizer hashes;
- source discovery response and canonical data-unit manifest;
- raw/admitted/deduplicated/consumed record and token counts;
- held-out group membership digest;
- exact launch command, Slurm allocation, host/rank/GPU/NIC topology;
- per-rank sampler cursors and batch digests;
- raw and summarized performance traces;
- immutable checkpoint generations and recovery receipts;
- failure, retry, and amendment history;
- an explicit statement of downstream/test access.

## Research anchors

- OAS contains public antibody repertoire data under CC BY 4.0.
- IgBert and IgT5 establish large unpaired OAS MLM followed by native-pair
  continuation as a strong antibody-specific baseline.
- BALM-paired provides independent evidence that native heavy--light pairing
  can improve cross-chain representations.
- Compute-optimal protein language-model work warns that repeating a limited
  MLM corpus can overfit and that model and token budgets should be scaled
  jointly.
- PyTorch FSDP2 shards parameters, gradients, and optimizer states; DeepSpeed
  ZeRO-3 provides the corresponding full model-state partitioning backend.

Primary references:

- <https://pmc.ncbi.nlm.nih.gov/articles/PMC11654935/>
- <https://zenodo.org/records/10684811>
- <https://arxiv.org/abs/2411.02142>
- <https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html>
- <https://www.deepspeed.ai/tutorials/zero/>

