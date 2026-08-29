# FrameContact / Geometry-aware Antibody–Antigen Project Master Ledger

**Cutoff:** 2026-08-02 00:32 CST (2026-08-01 16:32 UTC remote audit)
**Purpose:** exhaustive, evidence-bounded account of work completed, results obtained, current live state, failures, and remaining gates.
**Primary rule:** development validation, external zero-shot evaluation, engineering smoke, and frozen-test evidence are never conflated.

## 1. Executive truth

This project progressed through four connected stages:

1. Reproduced and audited MIPE and SPPIDER-seq, including strict splits, confidence intervals, partner-shuffle controls, deployment images, and A100 smoke tests.
2. Built and tested three families of intra-chain geometric self-attention encodings on real SAbDab2 N/CA/C coordinates: distance, distance plus local direction, and full relative pose.
3. Identified the central limitation of independent epitope/paratope heads: they predict intrinsic interface propensity and are mathematically unable to condition on the opposite partner.
4. Designed and implemented **FrameContact**, a partner-conditioned, contact-first architecture with separate antibody/antigen encoders, cross-chain pair representations, contact/distogram supervision, contact-derived residue marginals, specificity controls, strict structural novelty splits, and reproducibility infrastructure.

The evidence does **not** support a final SOTA or frozen-test claim. The strongest defensible conclusions are:

- Local direction improved the short real-structure intra-chain pilot over the spatial-topology control, but that experiment had one seed, three epochs, and only 36 validation complexes.
- On AsEP development validation, simple distance encoding was stronger than direction/full-pose variants for contact and epitope prediction; “more geometry channels always helps” was falsified.
- A single-seed topology-routed pilot improved contact and residue metrics over dense attention, but reduced the primary partner-specificity margin. It is exploratory, not confirmatory.
- CHIMERA zero-shot evaluation showed a statistically positive temporal mean-per-complex epitope AUPRC delta for spatial topology, especially on long and low-prevalence antigens, but mapping coverage failed the predeclared 95% gate and pooled epitope performance fell. It cannot be promoted as a confirmatory external win.
- The live SAbDab2 distogram branch is not currently winning: at matched epoch 19 it slightly improved contact/epitope pooled AUPRC and precision, but worsened recall, partner specificity, and paratope AUPRC across all five seeds.
- No SAbDab2 frozen test or AsEP official test has been opened for FrameContact model selection.

## 2. What was built

### 2.1 Research and literature synthesis

The research covered SE(3)-aware Transformers, protein inverse folding, equivariant graph networks, pair trunks, antibody-specific epitope models, protein-pair language models, and docking/contact prediction. The local catalog contains the exact paper/method ledger in [`docs/literature_catalog.json`](../docs/literature_catalog.json).

The practical conclusions used in the design were:

- Raw global xyz coordinates are unsuitable as positional encodings.
- Intra-chain geometry is naturally represented through SE(3)-invariant relative quantities in residue-local frames.
- Before estimating a relative complex pose, cross-chain attention should exchange invariant scalar features rather than subtract coordinates from unrelated monomer frames.
- Partner specificity should be supervised through an explicit bipartite residue-pair/contact representation, not inferred from two independent residue classifiers.
- A distogram provides a coherent contact probability by summing near-distance bins and avoids two independently contradictory distance/contact heads.

### 2.2 Real SAbDab2 structure adapter

Implemented a real-structure batch adapter that supplies:

- N, C-alpha, and C backbone coordinates;
- chain IDs and residue IDs;
- residue-valid, atom-valid, frame-valid, and missing-frame masks;
- frozen ESM residue embeddings;
- contact, epitope, and paratope labels;
- batching/cropping rules that preserve chain and label alignment.

Local frames are right-handed and constructed from N/C-alpha/C. Missing or degenerate frames are masked rather than imputed as valid geometry.

### 2.3 Intra-chain positional/geometric encodings

Implemented and tested:

- true dense sequence-only attention;
- sequence-local and spatial-neighbor graphs;
- C-alpha distance RBF bias;
- distance plus local-frame direction;
- full relative pose using local direction and relative rotation;
- IPA-style geometric attention;
- multi-basis local-pose bias;
- pair-conditioned local-pose rotary score;
- topology-routed multiscale attention at 6/8/12 Angstrom radii with separately gated RBF, direction, and rotation channels;
- geometry-ablated same-code controls.

For residues i and j, the invariant local direction and rotation are based on

`R_i^T (x_j - x_i)` and `R_i^T R_j`.

### 2.4 Partner-conditioned FrameContact

Implemented a model family with:

- parameter-unshared antibody and antigen intra-chain encoders;
- late or every-layer interleaved gated bidirectional cross-attention;
- low-rank bipartite pair/contact head;
- binary-contact or single-distogram output;
- exact contact probability as the sum of near-distance distogram bins;
- residue prediction modes:
  - unary self-attention head;
  - contact-only marginal/noisy-OR;
  - zero-gated hybrid of unary and contact-derived score;
- same-antigen/different-antibody partner panels;
- wrong-partner and partner-shuffle evaluation;
- geometry shuffle, local-frame corruption, graph rewiring, and coordinate perturbation controls;
- calibration, Brier/ECE, P@K/Recall@K, bootstrap confidence intervals, and directional partner-retrieval margins.

### 2.5 Geometry-consistency innovations

Implemented or prototyped:

- CDR-by-antigen-surface-patch sparse triangle consistency;
- triangle-inequality regularization using known intra-chain distances and predicted inter-chain distances;
- local Euclidean distance matrix feasibility using a Gram matrix PSD/rank-at-most-three criterion;
- contact-bottleneck specificity score based on positive counterfactual contact lift;
- listwise K-partner training aligned with 31-partner evaluation panels;
- paired-protein-language-model conditioning with pair-keyed caching;
- latent-pose/pose-ensemble interfaces as a later-stage design boundary, without claiming completed biological validation.

## 3. Completed benchmark results

### 3.1 MIPE reproduction and audit

Scope: official five-fold held-out predictions, 560 complexes (112 per fold), 10,000 complex-level bootstrap samples. Checkpoint selection was frozen before held-out predictions were loaded. The provided external test split was not accessed.

| Checkpoint rule | Task | AUROC | AUPRC | MCC |
|---|---:|---:|---:|---:|
| epoch 800 | Paratope | 0.843347 | 0.621497 | 0.462123 |
| epoch 800 | Epitope | 0.791853 | 0.251433 | 0.255619 |
| validation-best | Paratope | 0.881067 | 0.689221 | 0.534632 |
| validation-best | Epitope | 0.807419 | 0.280407 | 0.280765 |

Native bidirectional attention as a residue-pair interaction predictor was poor:

- pair AUROC: 0.493253;
- pair AUPRC: 0.003424;
- partner-specificity mean AUPRC difference: -0.036887;
- 95% CI: [-0.040148, -0.033782].

Interpretation: MIPE works as a residue propensity predictor, especially for paratopes, but its attention matrix is not a credible partner-specific contact map.

Evidence: [`METRICS_VERIFIED.json`](../../../mipe-sppider-handoff/mipe-sppider-handoff-20260729-v2/METRICS_VERIFIED.json).

### 3.2 SPPIDER-seq strict SAbDab2 evaluation

Split: 2,727 train / 481 validation / 455 frozen test instances; zero canonical-cluster and focal-query sequence overlap; seeds 17/42/73; 84 test clusters; 1,000 cluster bootstrap samples.

| Task | AUROC | AUPRC | F1 | MCC | AUPRC 95% CI |
|---|---:|---:|---:|---:|---:|
| Paratope | 0.948622 | 0.651089 | 0.640178 | 0.602334 | [0.626906, 0.678340] |
| Epitope | 0.671765 | 0.145445 | 0.210433 | 0.126885 | [0.121332, 0.166848] |

Improvement over the old version:

- paratope AUPRC +0.037670, paired CI [0.017025, 0.058822];
- epitope AUPRC +0.030356, paired CI [0.012954, 0.043886].

Partner shuffle:

- paratope AUPRC drop +0.011850, CI [0.000212, 0.026145];
- epitope AUPRC drop +0.002915, CI [-0.000489, 0.006323].

Calibration:

- paratope Brier/ECE: 0.04955 / 0.00418;
- epitope Brier/ECE: 0.07399 / 0.02350.

Interpretation: strong paratope and modest epitope propensity prediction, but limited evidence of antigen-antibody partner conditioning. The native model has no pair-contact output.

Evidence: [`FINAL_RESEARCH_REPORT.md`](../../../mipe-sppider-handoff/mipe-sppider-handoff-20260729-v2/SPPIDER/results/final_v3_ensemble_seed42/reports/FINAL_RESEARCH_REPORT.md).

### 3.3 Real SAbDab2 four-mode self-attention pilot

Scope: locked development split, 2,676 train / 36 validation complexes, seed 20260728, three epochs, real N/C-alpha/C structures and frozen ESM features. This was not a frozen-test experiment. The `none` condition retained the same spatial graph and is therefore a spatial-topology control, not a true dense baseline.

| Encoding | Joint macro AUPRC | Epitope AUPRC | Paratope AUPRC |
|---|---:|---:|---:|
| spatial-topology control | 0.440398 | 0.253331 | 0.627464 |
| distance RBF | 0.443554 | 0.256434 | 0.630673 |
| distance + direction | **0.455351** | 0.255474 | **0.655229** |
| full relative pose | 0.454978 | **0.257624** | 0.652331 |

Interpretation: local direction was worth retaining, while full relative rotation added no clear short-run benefit. The experiment is too small to establish a general benchmark win.

Evidence: [`SABDAB2_SELF_ATTENTION_ABLATION.json`](../../../outputs/sabdab2_structure_self_attention_ablation/SABDAB2_SELF_ATTENTION_ABLATION.json).

### 3.4 AsEP five-seed development matrix

Scope: 1,383 train / 170 validation, development-only; official test remained unopened. Metrics are validation summaries available in the research update.

| Intra-chain condition | Partner margin | Contact macro AUPRC | Contact pooled AUPRC | Contact P@10 | Epitope macro AUPRC | Epitope pooled AUPRC | Paratope macro AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| distance | **0.001171** | **0.07039** | **0.03627** | **0.10306** | **0.46873** | **0.34506** | 0.70932 |
| spatial topology | 0.001167 | 0.06728 | 0.03499 | 0.09518 | 0.45513 | 0.33023 | 0.71425 |
| distance + direction | 0.001081 | 0.06628 | 0.03577 | 0.09529 | 0.44859 | 0.32788 | **0.72272** |
| full relative pose | 0.001101 | 0.06223 | 0.03068 | 0.09541 | 0.42888 | 0.29082 | 0.71779 |

At the report cutoff, the dense sequence-only condition had four completed seeds and was stronger than the geometry conditions on partner margin (0.002239), contact macro AUPRC (0.08609), and epitope macro AUPRC (0.48971). It is not reported as a final five-seed row.

Interpretation: distance alone was the best of the four geometric variants, while direction/full pose did not generalize from the tiny SAbDab2 pilot to this AsEP development setting. This is a useful negative result.

Evidence: [`FRAMECONTACT_RESEARCH_UPDATE_20260731.md`](FRAMECONTACT_RESEARCH_UPDATE_20260731.md).

### 3.5 Topology-routed single-seed AsEP pilot

Scope: seed 17, 170 validation complexes, validation-selected checkpoints. Exploratory only.

| Model | Best epoch | Primary worst-direction partner margin | Contact pooled AUPRC | Contact macro AUPRC | Epitope pooled / macro AUPRC | Paratope pooled / macro AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| dense | 22 | **0.001430** | 0.047764 | 0.084220 | 0.393419 / 0.496511 | 0.689254 / 0.700860 |
| topology-routed | 30 | 0.000871 | **0.065979** | **0.107915** | **0.426230 / 0.527848** | **0.737759 / 0.751405** |

The routed model improved contact and residue metrics but worsened the primary partner-specificity margin. Attention diagnostics showed materially lower entropy and nonzero geometry-counterfactual changes, demonstrating that the geometry paths were actually used; this does not prove improved generalization.

Evidence: [`evaluation/topology-routing-seed17`](../evaluation/topology-routing-seed17).

### 3.6 CHIMERA real A100 zero-shot external evaluation

Hardware: NVIDIA A100-PCIE-40GB. Six frozen epoch-20 checkpoints: dense and spatial-topology, seeds 17/42/73.

Coverage failed the predeclared 95% mapping gate:

- antigen-fold: 192/292 complexes = 65.7534%;
- temporal: 196/293 complexes = 66.8942%.

Epitope results:

| Split | Model | mean-complex AUPRC | pooled AUPRC | AUROC | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| antigen-fold | dense | 0.572380 | 0.420891 | 0.809266 | 0.382552 | 0.496937 | 0.432306 | 0.379664 |
| antigen-fold | spatial | 0.576836 | 0.252825 | 0.724627 | 0.238583 | 0.312021 | 0.270405 | 0.199659 |
| temporal | dense | 0.307776 | 0.273784 | 0.748427 | 0.222397 | 0.417810 | 0.290280 | 0.239567 |
| temporal | spatial | 0.324708 | 0.255077 | 0.746921 | 0.199975 | 0.435566 | 0.274105 | 0.223944 |

Primary mean-complex AUPRC deltas:

- antigen-fold +0.004456, CI [-0.008832, 0.017795];
- temporal +0.016933, CI [0.003391, 0.030752], raw p=0.013999, Holm p=0.027997.

Because coverage failed, the statistically positive temporal result is not a confirmatory external success.

Paratope results:

| Split | Model | mean-complex AUPRC | pooled AUPRC | AUROC | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| antigen-fold | dense | 0.659569 | 0.612733 | 0.956902 | 0.377995 | 0.887269 | 0.530139 | 0.549002 |
| antigen-fold | spatial | 0.680742 | 0.638800 | 0.958715 | 0.464944 | 0.847696 | 0.600516 | 0.603161 |
| temporal | dense | 0.730485 | 0.698137 | 0.966357 | 0.444912 | 0.898786 | 0.595194 | 0.600230 |
| temporal | spatial | 0.755189 | 0.735117 | 0.972192 | 0.511609 | 0.898544 | 0.651990 | 0.651241 |

Post-hoc case study:

- antigen-fold length >=50: spatial epitope mean-AUPRC delta +0.0219, positive CI;
- antigen-fold low prevalence: +0.0300, positive CI;
- temporal length >=50: +0.0195, positive CI;
- temporal low prevalence: +0.0194, positive CI;
- longest quartile dense/spatial epitope mean AUPRC: 0.1205 / 0.1467;
- clear failures included 4qww (antigen length 529, discontinuous epitope, delta -0.183) and several short/high-prevalence cases.

Interpretation: geometry can help difficult long/low-prevalence cases, but per-complex gains coexist with worse pooled calibration/ranking. The analysis is diagnostic, not proof of universal benefit or canonical CHIMERA co-design performance.

Evidence: [`AB03_CHIMERA_EXTERNAL_RESULT_20260730.md`](../evaluation/AB03_CHIMERA_EXTERNAL_RESULT_20260730.md) and [`AB03_CHIMERA_EPITOPE_CASE_STUDY_20260730.md`](../evaluation/AB03_CHIMERA_EPITOPE_CASE_STUDY_20260730.md).

## 4. Current live SAbDab2 experiment

### 4.1 Matrix design

The running development-only experiment compares:

- residue output: unary self-attention head, contact-only marginal, hybrid unary plus contact;
- pair output: binary contact versus distogram-derived contact;
- geometry-consistency: triangle constraint off/on;
- five seeds;
- model selection using validation-only partner-specificity and contact tie-break metrics.

### 4.2 Live scheduler state

At 2026-08-01 16:32 UTC:

- Slurm job 1345042: five unary tasks running; contact-marginal/hybrid tasks queued by `JobArrayTaskLimit`;
- Slurm job 1345108: five distogram tasks running; triangle tasks queued by `JobArrayTaskLimit`;
- current jobs are on RTX 4090 nodes, not A100;
- no completed run receipt existed for these arrays at the audit cutoff;
- the latest synchronized comparisons available were residue-head epoch 22 and distogram epoch 19.

### 4.3 Matched epoch-19 distogram minus unary result

Across five seeds:

| Metric delta | Mean delta | Seeds positive | Exact sign-test p |
|---|---:|---:|---:|
| worst-direction partner margin | **-0.008344** | 2/5 | 0.3750 |
| contact pooled AUPRC | +0.0000218 | 3/5 | 1.0000 |
| contact fixed-threshold precision | +0.001185 | 4/5 | 0.1875 |
| contact fixed-threshold recall | **-0.029880** | 2/5 | 0.5000 |
| contact fixed-threshold F1 | +0.001998 | 4/5 | 0.1875 |
| epitope pooled AUPRC | +0.001223 | 3/5 | 0.8750 |
| epitope precision | +0.008387 | 3/5 | 0.3750 |
| epitope recall | **-0.108218** | 1/5 | 0.1250 |
| paratope pooled AUPRC | **-0.034064** | **0/5** | 0.0625 |
| paratope precision | **-0.056923** | **0/5** | 0.0625 |
| paratope recall | +0.014879 | 3/5 | 0.4375 |

Current interpretation:

- the distogram makes predictions somewhat sparser and can improve precision;
- it has not improved pair AUPRC meaningfully;
- it reduces epitope recall sharply;
- it worsens paratope AUPRC and precision in every seed;
- most importantly, it worsens the primary partner-specificity margin.

Therefore it is **not currently a promotion candidate**. The experiment remains development-only and can still change with later validation-selected epochs, but no test should be opened on the basis of the present trajectory.

Evidence: [`TEACHER_UPDATE_20260801_MATCHED_EPOCH18.md`](../docs/TEACHER_UPDATE_20260801_MATCHED_EPOCH18.md), live audit receipts under [`receipts/live-array-audit-*`](../receipts), and the 2026-08-02 scheduler audit recorded in this ledger.

## 5. Split and leakage work

### 5.1 Strict double-cold SAbDab2 split

Built an auditable split from the official SAbDab2 v0.1 Zenodo release (DOI 10.5281/zenodo.20083995):

- cohort: 3,663 complexes;
- strict antigen-fold roles: 2,584 train / 540 validation / 539 test;
- zero cross-role antibody clusters;
- zero cross-role antigen cluster tokens;
- zero cross-role exact antibody/antigen sequences or PDB IDs;
- zero crossings among 725,687 Foldseek hard edges;
- hard structural edge: reciprocal-min TM-score >=0.50;
- 184 connected components;
- largest structural component: 1,039;
- 1,482 instances moved relative to the earlier development v3 assignment.

The audit also found that the earlier v3.1 antigen-fold split was not truly double-cold:

- eight antigen tokens affecting 240 rows crossed roles;
- eleven antibody clusters affecting 64 rows crossed roles.

The temporal split is 1,499 / 397 / 1,767 with exact isolation. An older 2,727 / 481 / 455 temporal-style split had one PDB overlap and is not the final strict split.

No labels or predictions from the strict test partition have been read. The test panel has not yet been materialized for model evaluation.

Evidence: [`receipts/double-cold-split-v1-1346323`](../receipts/double-cold-split-v1-1346323).

## 6. Engineering and systems results

### 6.1 Test and source receipts

The sealed v17 source receipt contains 528 records and passed:

- local sealed run: 298 tests passed in 17.77 seconds;
- remote run: 293 passed, five optional Boltz tests skipped, zero failures, in 78 seconds.

Current workstation recheck on 2026-08-02:

- bare `pytest` failed collection because the repo is not installed editable and `PYTHONPATH=src` was absent;
- with `PYTHONPATH=src`, 292 tests passed and six failed;
- five failures were optional Boltz tests because local `fairscale` is absent;
- one USalign boundary test was blocked by the managed sandbox denying `SC_SEM_NSEMS_MAX`/process semaphore access;
- this does not supersede the clean sealed v17 remote receipt, but it documents the exact local reproduction prerequisites.

Evidence: [`receipts/double-cold-source-v17-final/ENGINEERING_RECEIPT.json`](../receipts/double-cold-source-v17-final/ENGINEERING_RECEIPT.json).

### 6.2 Real Boltz Pairformer checkpoint adapter

Completed a strict adapter smoke with official weights:

- 2,544 tensors strictly loaded;
- 147,400,704 frozen parameters;
- 48 Pairformer blocks;
- pair output shape [1,2,2,128];
- single output shape [1,2,24];
- CPU synthetic forward 0.049 seconds after 5.567-second load;
- adapter trainable parameters: 24,344.

Boundary: this was not the native Boltz end-to-end data pipeline and did not produce a biological benchmark result.

Evidence: [`BOLTZ1_PAIRFORMER_LOCAL_SMOKE_20260731.json`](../receipts/BOLTZ1_PAIRFORMER_LOCAL_SMOKE_20260731.json).

### 6.3 Real paired-protein language model extraction

Completed a real PPLM checkpoint extraction smoke:

- checkpoint size: 2.604 GB;
- one real training pair: antibody length 111, antigen length 59;
- RTX 4090 extraction: 11.765 seconds;
- peak GPU memory: 2.8708 GB;
- embeddings: [111,1280] and [59,1280];
- four grouped cross-attention maps: [4,111,59];
- approximately 165-fold storage reduction versus retaining all raw maps.

Boundary: no FrameContact biological training improvement was demonstrated. The upstream paper/code licensing discrepancy (paper says MIT; current model package is noncommercial PolyForm) must be resolved before redistribution or commercial use.

Evidence: [`receipts/pplm-real-checkpoint-v1-1345959`](../receipts/pplm-real-checkpoint-v1-1345959).

### 6.4 Paired conditioner and EDM engineering smoke

RTX 4090 real-batch smoke:

- zero-gated conditioner reproduced baseline exactly (max difference 0);
- activating the path changed contact output by 0.610610, proving it is connected but not proving quality;
- projection error 0;
- FP32/BF16 SE(3) invariance error 0 in the corrected test;
- EDM feasibility loss 0.0063205;
- paired-gate gradient 0.000528922;
- distogram gradient L1 11.4257.

Two failed attempts were retained:

- job 1345856: BF16 invariance test context problem;
- job 1345874: BF16 `eigvalsh` unsupported;
- corrected implementation performs the eigenvalue calculation in FP32.

Evidence: [`FRAMECONTACT_FRONTIER_ENGINEERING_EVIDENCE_20260801.md`](FRAMECONTACT_FRONTIER_ENGINEERING_EVIDENCE_20260801.md).

### 6.5 Listwise partner-specificity scaling

The original K=1 pairwise training objective was inconsistent with evaluation against 31 candidate partners. Implemented listwise K-partner training and activation checkpointing.

- first K=4/K=8 attempts OOMed at roughly 32.4/29.0 GB host memory;
- checkpointing initially covered the wrong branches and was corrected;
- corrected K=4/K=8 jobs completed with roughly 1.95/2.10 GiB host memory;
- exact K=4, longest real training sample (Ab 231, Ag 2,363): 13.554 seconds, 10.459 GB peak CUDA, finite output;
- comparison smoke: K=1 16.996 seconds / 2.875 GB; K=4 28.847 seconds / 10.459 GB; K=8 29.918 seconds / 10.694 GB.

Boundary: single-example engineering evidence only; no optimizer-step or validation-quality claim.

Evidence: [`receipts/listwise-specificity-v1-1346200/ENGINEERING_RECEIPT.json`](../receipts/listwise-specificity-v1-1346200/ENGINEERING_RECEIPT.json).

### 6.6 A100 attention-kernel benchmark

Hardware: A100-PCIE-40GB, PyTorch 2.5.1, CUDA 12.4. This used PyTorch memory-efficient CUTLASS SDPA, not a custom fused kernel.

Correctness:

- FP32 output max error 7.45e-7; gradient max error up to 5.96e-7;
- BF16 output/gradient max error 0.0078125;
- all preset correctness gates passed.

Full-model synthetic benchmark:

| Ab x Ag length | Training speedup | Memory ratio | Memory reduction | Inference speedup |
|---|---:|---:|---:|---:|
| 220 x 128 | 1.0632x | 0.9390 | 6.1% | 1.0461x |
| 220 x 512 | 1.0679x | 0.9189 | 8.1% | 1.0191x |
| 231 x 2363 | 1.0072x | 0.9136 | 8.6% | 1.0203x |

Primitive attention speedups were approximately 1.28-1.94x depending on shape and dtype. These are synthetic performance measurements, not biological quality improvements.

Evidence: [`receipts/attention-kernel-a100-v3-ab03/BENCHMARK.json`](../receipts/attention-kernel-a100-v3-ab03/BENCHMARK.json).

### 6.7 Deployment and handoff

- MIPE and SPPIDER deployment bundles were transferred and smoke-tested on AB03 A100.
- Strict-loaded checkpoints: four SPPIDER and ten MIPE; finite full forwards passed.
- MIPE image ID: `sha256:5d64...`; SPPIDER image ID: `sha256:d90...`.
- The latest FrameContact live research handoff is v8 under the ParaCloud project root; 13 files, 2,073,744 bytes, all indexed hashes verified; handoff-index SHA-256 starts `ba14ad48...`.
- A100 smoke tests validate loading and inference mechanics; they do not recompute or independently establish reported model metrics.
- The main `geometry-self-attention` directory is not a Git repository, so the latest FrameContact work has **not** been verified as pushed to GitHub. Remote handoff copies are not equivalent to Git history.

## 7. Early SE(3)-RoPE prototype findings

The early prototype established two useful engineering facts:

- PCA-derived local frames are unstable in degenerate/symmetric neighborhoods: observed rotation drift 1.3002 and noise drift 1.5872 in the diagnostic example;
- a Triton pair-bias FP32 drop-in matched the reference with max difference 5.72e-6 under a 2e-5 threshold.

This motivated backbone N/C-alpha/C frames and explicit invariant relative pose. It did not establish a biological “SE(3)-RoPE” benchmark improvement.

Evidence: [`outputs/se3_geometry_results.json`](../../../outputs/se3_geometry_results.json).

## 8. What the project learned scientifically

1. **Independent residue heads are not partner-specific.** If the antigen encoder/head never receives antibody information, replacing the antibody cannot change epitope logits in eval mode. The same holds in reverse for paratopes.
2. **Paratope prediction is easier than epitope prediction.** Paratopes are concentrated around antibody CDRs; epitopes are more dispersed, low-prevalence, and partner-specific.
3. **Geometry is not monotonically beneficial.** Local direction helped a tiny SAbDab2 pilot, while AsEP development favored simple distance or dense attention over direction/full pose.
4. **Per-complex and pooled metrics can disagree.** Spatial topology helped some hard complexes and mean-complex AUPRC while harming pooled AUPRC, revealing cross-complex calibration and score-scale problems.
5. **Contact-first factorization is necessary but not sufficient.** An explicit contact map makes partner dependence supervisable, but current live distogram training still weakens the primary partner margin.
6. **Precision improvements can hide recall collapse.** The live distogram became sparser and slightly more precise while losing substantial epitope recall and paratope performance.
7. **Structural novelty must be audited at both endpoints.** Antigen-only fold isolation was insufficient; antibody-cluster crossings materially contaminated the earlier split.
8. **A model must beat wrong partners directionally.** Merely changing when a partner is shuffled is insufficient; the cognate partner must improve the correct epitope/contact region.

## 9. Failures and negative results retained

- MIPE attention did not predict contacts and showed negative partner specificity.
- SPPIDER-seq epitope partner-shuffle CI crossed zero.
- Full relative pose did not beat simpler geometry consistently.
- Dense attention was stronger than the four geometric AsEP conditions at the partial multi-seed cutoff.
- Topology routing improved contact/residue metrics but worsened partner margin in the seed-17 pilot.
- CHIMERA spatial topology reduced pooled epitope AUPRC and failed the mapping coverage gate.
- Current distogram training reduces recall, partner margin, and paratope AUPRC.
- First listwise K=4/K=8 attempts OOMed; activation checkpointing was initially placed incorrectly.
- BF16 eigendecomposition failed and was corrected to FP32.
- Direct current local test invocation is not self-contained without `PYTHONPATH=src` and optional dependencies.

These failures are part of the research result, not omitted runs.

## 10. Exact status matrix

| Work item | Status | Evidence boundary |
|---|---|---|
| MIPE reproduction/evaluation | Complete | internal official five-fold held-out; external split untouched |
| SPPIDER strict evaluation | Complete | frozen internal test with cluster bootstrap |
| MIPE/SPPIDER AB03 deployment | Complete | A100 load/forward smoke, not metric recomputation |
| real SAbDab2 N/CA/C adapter | Complete | code/tests/real batches |
| three requested geometric encodings | Complete | real development pilot plus broader AsEP development matrix |
| FrameContact contact-first architecture | Implemented | code/tests/development experiments |
| distogram/contact-marginal/hybrid matrix | Running | unary and distogram first waves running; later array tasks queued |
| triangle constraint | Implemented, queued | engineering tests; biological task not yet started |
| strict double-cold split | Complete and audited | test labels/predictions unopened |
| topology routing | Exploratory result | one seed only |
| Boltz Pairformer adapter | Engineering smoke complete | no biological benchmark |
| PPLM conditioner | Extraction/engineering complete | no trained quality result; license gate |
| listwise specificity | Engineering smoke complete | no validation-quality result |
| A100 SDPA performance | Complete synthetic benchmark | no custom fused kernel, no quality claim |
| CHIMERA external classification | Completed with failed coverage gate | post-hoc/censored evidence only |
| canonical CHIMERA co-design | Not done | no claim |
| SAbDab2 FrameContact frozen test | Not opened | awaiting frozen development selection |
| AsEP official test | Not opened | awaiting frozen development selection |
| native GVP/EGNN/IPA/Pairformer same-split benchmark | Not completed | adapters/options exist but no fair end-to-end result |
| apo/predicted-monomer control | Not completed | planned |
| latent pose ensemble/docking refinement | Not completed | design boundary only |
| GitHub publication of latest FrameContact repo | Not verified/not complete | local directory has no `.git` |

## 11. What can be claimed now

Defensible project claim:

> Built an evidence-first geometry-aware antibody-antigen modeling stack spanning strict SAbDab2 structural-novelty splits, real N/CA/C local-frame encodings, parameter-unshared monomer encoders, partner-conditioned contact/distogram prediction, listwise specificity controls, calibration and counterfactual evaluation; reproduced MIPE/SPPIDER and ran development/external/A100 audits while preserving negative results and unopened frozen-test boundaries.

Not yet defensible:

- “FrameContact beats SOTA”;
- “distogram improves partner specificity”;
- “SE(3)-RoPE improves the real benchmark”;
- “Pairformer/PPLM improves biological performance”;
- “fused kernel implemented”;
- “canonical CHIMERA co-design completed”;
- “latest code is on GitHub.”

## 12. Remaining promotion gates

Before a paper-level model claim:

1. Finish all live development tasks and freeze the primary configuration using validation only.
2. Require positive worst-direction partner margin across seeds, not merely better unary AUPRC.
3. Compare contact-only marginal, hybrid, and unary heads with fixed-threshold precision/recall and pooled calibration.
4. Run the triangle branch and retain it only if it improves contact consistency without reducing partner specificity.
5. Retrain on the strict double-cold split; report at least five seeds.
6. Add fair same-split baselines or explicitly document infeasibility.
7. Open SAbDab2 frozen test and AsEP official test exactly once after configuration freeze.
8. Repair CHIMERA mapping coverage before confirmatory external promotion.
9. Add apo/predicted-monomer controls to separate holo-conformation leakage from genuine generalization.
10. Only after contact-first classification is validated, proceed to pose ensembles and canonical co-design.

## 13. Artifact map

- Main code and protocols: repository root.
- Research update: [`reports/FRAMECONTACT_RESEARCH_UPDATE_20260731.md`](FRAMECONTACT_RESEARCH_UPDATE_20260731.md).
- Frontier engineering evidence: [`reports/FRAMECONTACT_FRONTIER_ENGINEERING_EVIDENCE_20260801.md`](FRAMECONTACT_FRONTIER_ENGINEERING_EVIDENCE_20260801.md).
- Current teacher-facing updates: [`docs/TEACHER_UPDATE_20260801_MATCHED_EPOCH18.md`](../docs/TEACHER_UPDATE_20260801_MATCHED_EPOCH18.md) and related `TEACHER_UPDATE_*` files.
- Strict split receipt: [`receipts/double-cold-split-v1-1346323`](../receipts/double-cold-split-v1-1346323).
- Sealed source receipt: [`receipts/double-cold-source-v17-final`](../receipts/double-cold-source-v17-final).
- A100 kernel benchmark: [`receipts/attention-kernel-a100-v3-ab03`](../receipts/attention-kernel-a100-v3-ab03).
- CHIMERA evaluation: [`evaluation/AB03_CHIMERA_EXTERNAL_RESULT_20260730.md`](../evaluation/AB03_CHIMERA_EXTERNAL_RESULT_20260730.md).
- MIPE/SPPIDER handoff: [`mipe-sppider-handoff-20260729-v2`](../../../mipe-sppider-handoff/mipe-sppider-handoff-20260729-v2).
- Raw early four-mode result: [`SABDAB2_SELF_ATTENTION_ABLATION.json`](../../../outputs/sabdab2_structure_self_attention_ablation/SABDAB2_SELF_ATTENTION_ABLATION.json).
- Live array audit receipts: [`receipts`](../receipts).

---

**Bottom line:** the project has produced a substantial research system, reproducible benchmark/audit infrastructure, several useful positive and negative findings, and a credible paper hypothesis. It has not yet crossed the decisive paper gate: reproducible partner-specific improvement on strict double-cold frozen tests.
