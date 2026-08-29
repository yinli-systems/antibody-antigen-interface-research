# FrameContact frontier engineering evidence — 2026-08-01

## Outcome

Two post-registration research paths now have authenticated engineering
evidence:

1. a zero-gated paired-sequence conditioner plus local EDM objective runs
   end-to-end under CUDA/BF16; and
2. the official PPLM 650M checkpoint can produce safe, sequence-bound antibody--
   antigen features that the FrameContact loader accepts.

Neither result is trained-performance, frozen-test, external-generalization,
benchmark, novelty, commercial-use, or production evidence.

## Paired conditioner plus EDM smoke

### Successful run

- Slurm: `1345879`
- GPU: NVIDIA GeForce RTX 4090
- source manifest SHA256:
  `a74684585b16dea6c3ed1669046759c476d14bd518a839846b3a1092ae24e930`
- source tree SHA256:
  `a7ee6c42498b9fa9ad49a5cbeba9c5d74659f2927a8773aea3a5361c32079100`
- receipt:
  `receipts/frontier-paired-conditioning-v3-4090-1345879.json`
- receipt SHA256:
  `bd72d1befe500ef51766482ab70ba5afee0ae632c6cbf3cdcef037262783e2e5`

Measured checks:

| Check | Result |
|---|---:|
| zero-gated baseline max difference | `0` |
| activated paired-path contact change | `0.6106104851` |
| exact distogram-to-contact projection max error | `0` |
| independent-monomer SE(3) max error, FP32 | `0` |
| independent-monomer SE(3) max error, BF16 | `0` |
| local EDM loss | `0.0063204966` |
| paired prior gate gradient absolute value | `0.0005289220` |
| distogram-head gradient L1 | `11.4256687164` |

The active-change magnitude is not an accuracy metric. It only proves that the
new branch is connected after its gate is deliberately activated.

### Preserved failure trail

- `1345856` failed the initial BF16 invariance assertion. The test reused one
  autocast context and did not separate algebraic FP32 invariance from mixed-
  precision numerical drift. The replacement uses fresh contexts and reports
  both FP32 and BF16 errors.
- `1345874` passed the revised invariance gate but failed because CUDA
  `eigvalsh` does not support BF16. The local Gram/eigenvalue calculation is now
  explicitly FP32 with autocast disabled, while the rest of the model remains
  BF16-eligible.

These failures are part of the evidence. They are not deleted or rewritten as
successful jobs.

## Official PPLM checkpoint smoke

### Provenance

- paper: <https://www.nature.com/articles/s41467-026-70457-5>
- official source: <https://github.com/junliu621/PPLM>
- source revision:
  `1267db4592703095b889cc22da6cf1796ce038b5`
- official checkpoint size: `2,604,388,437` bytes
- checkpoint SHA256:
  `c8ce3b75dfd7b5890dfa7d96de7f698b522a5cbf59d7ee2732cb80ad3e5df495`
- license boundary: the paper describes MIT code while the current repository
  README states PolyForm Noncommercial. The stricter noncommercial boundary is
  retained and neither code nor weights are redistributed.

The extractor uses `torch.load(..., weights_only=True)`, requires an exact
33-layer/1280-width/20-head architecture and exact state-dict compatibility,
and refuses modified tracked source.

### Input authority

- instance: `pdb_00008c3k_B_+`
- role: SAbDab2 antigen-fold **train**
- antibody length: `111`
- antigen length: `59`
- both sequence hashes match the authenticated structure-cache record
- no validation/test labels, predictions, or metrics are used

### Successful run

- Slurm: `1345959`
- GPU: NVIDIA GeForce RTX 4090
- source manifest SHA256:
  `54dd6eb7c963df71c2d498cdf4bbfdaca09927c6512513f2124fc9a6dfe3f353`
- source tree SHA256:
  `cc9bf8342701e175ec551310a9d8f2d028396edddccff07863e748b5a08c680d`
- elapsed time: `11.7651 s`
- peak allocated CUDA memory: `2,870,809,088` bytes
- artifact SHA256:
  `dd59e2033d625b5ad56109faf18c3c4a6bee53bbd2e6a1e77ff2d5195285fff1`
- receipt SHA256:
  `e62d5e10b34ac358aeed50d01c7f741693a89c3e471a7a7388afd14b8138e1df`

Output contract:

| Array | Shape | dtype |
|---|---:|---|
| antibody embedding | `[111, 1280]` | float32 |
| antigen embedding | `[59, 1280]` | float32 |
| symmetric grouped inter-attention | `[4, 111, 59]` | float32 |

All 660 source attention maps are not cached. A frozen reduction first averages
the two directions, then averages all heads inside four layer groups
`[0,8), [8,16), [16,24), [24,33)`. This keeps coarse depth information while
reducing storage by 165x relative to 33x20 maps. The reduction is an explicit
ablation choice, not a claim that four maps preserve every PPLM signal.

The generated NPZ has exactly three finite floating-point arrays and is read
with `allow_pickle=False`. Its receipt binds source, checkpoint, license,
sequence order, sequence hashes, reduction, shapes, dtypes and artifact hash.

## Current development-training snapshot

The existing frozen arrays are unchanged by this work:

- `1345042`, unary-only first condition: all five seeds have reached epoch 12.
  At synchronized epoch 12, mean contact macro/pooled AUPRC is
  `0.0232966/0.0116354`; validation-selected precision/recall/F1/MCC is
  `0.0209907/0.0854411/0.0331819/0.0396953`; P@10 is `0.0366120`; epitope
  precision/recall/F1/MCC is `0.125443/0.555118/0.204066/0.0939018`.
  The pre-registered worst-direction selection margin is `-0.00274424`.
- `1345108`, distogram first condition: all five seeds have synchronized at
  epoch 10. At synchronized epoch 10, mean
  contact macro/pooled AUPRC is `0.0230196/0.0119389`;
  validation-selected precision/recall/F1/MCC is
  `0.0216276/0.0898042/0.0339083/0.0410793`; P@10 is `0.0320219`; epitope
  precision/recall/F1/MCC is `0.137677/0.481281/0.213204/0.107160`.
  The pre-registered worst-direction selection margin is `0.00151922`.

The two `selection_margin` values above are the mean across seeds of each
seed's minimum directional margin. They are deliberately not replaced by the
larger mean-directional margins (`0.00749686` and `0.0107711` respectively),
because doing so would weaken the registered worst-direction criterion.

The new read-only trajectory parser also answers the requested epoch trend
question. For unary-only, epoch 1 to 12 validation-selected precision changed
`0.0226084 -> 0.0209907` and recall `0.138204 -> 0.0854411`; for distogram,
epoch 1 to 9 precision changed `0.0222086 -> 0.0216571` and recall
`0.132823 -> 0.100716`. At the fixed raw-logit-zero operating point, precision
increased (`0.00708205 -> 0.0113224` unary; `0.00866691 -> 0.0140740`
distogram) while recall decreased (`0.908730 -> 0.434615`; `0.838175 ->
0.316149`). This is a score-sharpening/operating-point trade-off, not monotonic
joint improvement. In contrast, distogram epitope precision increased from
`0.123761` to `0.137677` while recall decreased from `0.656648` to `0.481281`.

These cross-epoch values are descriptive development diagnostics. They do not
select a checkpoint, and the epoch-12 versus epoch-10 snapshot is not a fair
condition winner comparison. The contact-marginal, hybrid, and triangle
conditions remain queued behind the five-task concurrency limit.

At the latest epoch present for every seed on both sides, epoch 10, the paired
`distogram - unary` development deltas are more informative:

| metric | mean paired delta | seeds above zero |
|---|---:|---:|
| contact macro AUPRC | +0.000174 | 3/5 |
| contact pooled AUPRC | +0.001181 | 5/5 |
| selected precision | +0.002359 | 5/5 |
| selected recall | +0.015358 | 3/5 |
| selected F1 | +0.003399 | 5/5 |
| selected MCC | +0.005364 | 3/5 |
| fixed-threshold precision | +0.003079 | 5/5 |
| fixed-threshold recall | -0.084299 | 0/5 |
| P@10 | -0.002696 | 3/5 |
| P@Lmin/5 | -0.001998 | 2/5 |
| epitope pooled AUPRC | +0.009342 | 4/5 |
| epitope precision | +0.016621 | 5/5 |
| epitope recall | -0.091436 | 0/5 |
| paratope pooled AUPRC | -0.013043 | 1/5 |
| worst-direction selection margin | +0.019331 | 4/5 |

The margin delta has sample SD `0.0363602`; moreover directional top-1
retrieval changed by `-0.001093` and `-0.019308` on average. Therefore the
present evidence is consistent with a sharper/more conservative score
distribution, but does not yet establish better ranked contacts or more robust
partner retrieval. The authenticated matched-epoch JSON has SHA256
`43a4d4904e9fa32a61ae7a86bd216966a8bc548be6f0eb478dba2eb357f22c4e`.

These are in-flight development values, not checkpoint selections or final
comparisons. Other array conditions are still queued behind the five-task
concurrency limits.

## Critical scalability boundary before PPLM training

PPLM outputs are pair-conditioned. A wrong-partner control therefore requires a
separately extracted feature artifact for that exact antibody--antigen pair.
Reusing a cognate pair's PPLM embedding after swapping one endpoint would be a
scientific error and could fabricate partner specificity.

The present evidence authorizes only:

- single-pair extraction;
- safe cache ingestion;
- model-path and gradient smokes.

The code now provides an ordered pair cache key and a fail-closed
`PPLMPairFeatureStore`: the key hashes both endpoint sequence digests with
explicit antibody/antigen roles, and a swapped endpoint resolves to a different
artifact. This prevents accidental reuse of cognate paired features in a
wrong-partner example. A complete cohort manifest and negative-panel cache have
not yet been materialized.

It does **not** authorize a full training launch until the following are built:

1. a cache key bound to both endpoint sequence hashes;
2. explicit artifacts for every cognate and preregistered wrong pair;
3. length/OOM stress tests, especially for combined lengths above 1,024;
4. cutoff and SAbDab2/AsEP overlap audit for PPLM pretraining;
5. a noncommercial-use acknowledgement;
6. identical five-seed selection and calibration rules.

## Next controlled gates

1. Let the frozen binary/distogram arrays finish and seal their development
   selection.
2. Stress the real PPLM extractor over short/median/long train pairs and record
   failure/throughput distributions.
3. Build exact pair-keyed caches for a small train/validation hard-negative
   pilot, not the frozen test.
4. Compare paired embedding only, grouped attention only, both, and equal-
   parameter random/frozen controls.
5. Run the local EDM loss only after the current distogram selection is locked;
   require P@K/precision improvement, recall retention, calibration and partner
   margin.
6. Open no held-out external label until architecture, seed, checkpoint,
   threshold, temperature and partner panel are frozen.

## Optional SDPA execution path: measured and evidence-gated

The explicit reference attention path uses `einsum -> additive pair bias ->
masked softmax -> einsum`.  It is required whenever attention weights or
content-only diagnostics are requested.  A separate, default-off
`attention_kernel=sdpa` path now uses PyTorch scaled dot-product attention only
when diagnostics are not requested, while preserving the same materialized
per-head geometry/contact bias and candidate mask.

The change is systems-only:

- no parameter or state-dict key was added;
- `reference` remains the default;
- the current Slurm arrays `1345042` and `1345108` remain on their immutable
  reference source trees;
- no model-quality, validation, frozen-test, or external metric was computed by
  the kernel benchmark.

### Correctness and regression evidence

- primitive FP32 maximum output/gradient differences:
  `6.109476e-7 / 7.152557e-7`;
- primitive BF16 maximum output/gradient differences:
  `0.01171875 / 0.005859375`;
- focused current-source tests: `69 passed`;
- full authenticated V12 regression: `257 passed, 5 skipped` in `75.61 s`;
- the five skips are optional external Boltz baseline tests, not FrameContact or
  SDPA tests.

The V12 source manifest SHA256 is
`1312349b1e93b64affa64d3aa557aa0abcc8b74d60b32bb38ad8d12512fd4b35`;
its tree SHA256 is
`f8319a3dd0cb148a2e3a3921df909efae77522cc601a995516244e2e5f8e899a`.

### RTX 4090 synthetic integrated benchmark

Slurm `1346024`, PyTorch `2.7.1+cu128`, CUDA `12.8`, BF16, batch size one:

| Ab x Ag length | train-step speedup | train peak-memory ratio | inference speedup |
|---:|---:|---:|---:|
| 220 x 128 | 1.0719x | 0.9390 | 1.0588x |
| 220 x 512 | 1.0611x | 0.9189 | 0.9657x |
| 220 x 1000 | 1.0424x | 0.9149 | 1.0118x |

Thus SDPA produces a modest `4.2-7.2%` synthetic full-training-step speedup and
about `6.1-8.5%` lower peak allocated memory in these three shapes.  Inference
is mixed, including a small regression at 220 x 512, so this does not support a
blanket “faster inference” claim.  It is also not a custom fused-kernel claim:
the profiler identified PyTorch's memory-efficient CUTLASS attention backend.

The authenticated benchmark receipt SHA256 is
`1a214289d80c5fd404839877d675b97e79a5b5ba501f5cf97ff4a2971026f60c`.
The governing protocol is
`protocols/framecontact_attention_kernel_amendment_v1.json`.

### AB03 A100 synthetic replication

The same numerical and integrated checks were independently rerun on AB03 GPU0
using an NVIDIA A100-PCIE-40GB, PyTorch `2.5.1`, CUDA `12.4` and cuDNN `90100`.
The container network was disabled and the exact image ID, uploaded package,
source manifest, benchmark and run receipt were cryptographically bound.

- FP32 maximum output/gradient differences: `7.450581e-7 / 5.960464e-7`;
- BF16 maximum output/gradient differences: `0.0078125 / 0.0078125`;
- both passed the frozen `5e-5 / 0.03` tolerances;
- the profiler again selected PyTorch memory-efficient CUTLASS attention for
  SDPA, not a project-authored fused kernel.

| Ab x Ag length | pair count | train-step speedup | train peak-memory ratio | inference speedup |
|---:|---:|---:|---:|---:|
| 220 x 128 | 28,160 | 1.0632x | 0.9390 | 1.0461x |
| 220 x 512 | 112,640 | 1.0679x | 0.9189 | 1.0191x |
| 231 x 2363 | 545,853 | 1.0072x | 0.9136 | 1.0203x |

On this A100 runtime, SDPA therefore reduced allocated training peak memory by
about `6.1-8.6%`.  Median synthetic train-step time improved by `0.7-6.8%`;
the longest observed-size boundary was essentially latency-neutral but retained
an `8.6%` allocator reduction.  This is shape-specific single-run systems
evidence, not measured epoch throughput or a biological benchmark improvement.

The first AB03 attempt (`v2`) completed the benchmark but failed receipt
finalization because the root-run container produced a mode-`0600` JSON that
the host account could not read.  It was not promoted.  The runner was amended
to use the invoking UID/GID and the complete benchmark was rerun in a new
create-only `v3` result directory.

Authenticated local copies:

- benchmark SHA256:
  `8f853bee439630d2e9784359b0a8869fa9556c4366415334e69ef046c29e9687`;
- run-receipt SHA256:
  `71a0bde2894afac74d9eacd4b4de6dfa06dacbe701e8eafb1d720167194f9243`;
- source-manifest SHA256:
  `50195990a8d1d9eb3ca5a03a433daf7ed51056b627d3a84e1a0695200ad855b9`;
- package SHA256:
  `2760235d7cde987d86d5b465641a1ddd054b2fddba9acf55de12f436b5492fc5`.

The A100 addendum is
`protocols/framecontact_attention_kernel_a100_addendum_v1.json`.  It authorizes
only A100 numerical-equivalence and recorded-shape latency/allocator claims;
model quality, frozen test, external generalization, custom fused-kernel and
SOTA claims remain unauthorized.

### Updated in-flight development snapshot

At the latest synchronized five-seed epoch:

- unary-only epoch 12: contact macro/pooled AUPRC
  `0.02329661 / 0.01163539`, selected precision/recall/F1/MCC
  `0.02099070 / 0.08544107 / 0.03318192 / 0.03969532`, paratope AUPRC
  `0.69862894`, epitope AUPRC `0.12519582`, worst-direction selection margin
  `-0.00274424` (mean-directional margin `0.00749686`);
- distogram synchronized epoch 10: contact macro/pooled AUPRC
  `0.02301956 / 0.01193890`, selected precision/recall/F1/MCC
  `0.02162759 / 0.08980420 / 0.03390827 / 0.04107928`, paratope AUPRC
  `0.67956956`, epitope AUPRC `0.13111566`, worst-direction selection margin
  `0.00151922` (mean-directional margin `0.00720204`).

These remain in-flight train/validation observations.  Distogram currently has
higher pooled contact AUPRC, selected precision, recall, F1, MCC, epitope AUPRC
and worst-direction margin, but essentially tied/slightly lower macro contact
AUPRC, lower P@10 and lower paratope AUPRC. They are neither synchronized at
the same epoch nor checkpoint selections, so this is not evidence that
distogram is superior: later epochs, other conditions and all five paired-seed
promotion rules remain outstanding.

### Live-monitor regression evidence

The live parser was upgraded without changing the frozen training source or
jobs. It now exposes fixed and validation-selected precision/recall/F1/MCC,
P@K/Recall@K, residue metrics, both retrieval directions, and descriptive
full/recent epoch slopes. The parser SHA256 is
`131c3c6e225628adb9a166835c0d730696fcec5fa33e7c2831007b9c9740be24`;
the matched-epoch comparator SHA256 is
`1ef38311f49082180912a40b5c071269aeff091afc6e7c595af561a9450db67d`.

- focused local and authenticated ParaCloud tests: `6 passed`;
- current-source authenticated ParaCloud regression at a location with the
  bound sibling `external/` and `recovery/` artifacts: `261 passed, 5 skipped`
  in `78.24 s`;
- the same five skips are optional pinned-Boltz-source tests.

An earlier temporary copy at the wrong directory depth produced `257 passed`
plus two path-binding failures because its required sibling evidence roots did
not exist. That run was not promoted; moving the identical files under the
correct evidence root resolved both failures without changing code.
