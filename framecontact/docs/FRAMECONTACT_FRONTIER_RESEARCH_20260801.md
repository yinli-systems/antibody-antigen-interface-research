# FrameContact frontier research and execution blueprint

**Frozen on:** 2026-08-01  
**Purpose:** determine what is scientifically defensible, what is directly
reusable, and what still needs controlled experiments before FrameContact can
support a paper, a benchmark claim, or a strong research-engineering portfolio
claim.

## Executive decision

The project should not be framed as “a better 3D positional encoding.” The
stronger and falsifiable paper question is:

> Can invariant intra-chain geometry, paired-sequence evidence, and an explicit
> cross-chain distance/contact representation recover partner-specific
> antibody--antigen interfaces under structural novelty?

The evidence must form a ladder:

1. **Single-chain geometry:** does local geometry improve interface propensity
   over a dense sequence-only encoder?
2. **Partner conditioning:** for a fixed antigen, does changing the antibody
   move the epitope map toward the new antibody's true epitope?
3. **Interaction geometry:** does a supervised cross-chain distogram improve
   calibrated residue-pair contact ranking?
4. **Structural generalization:** do the gains survive sequence-cluster,
   Foldseek/TM-score antigen-fold, temporal, apo/predicted-monomer, and external
   evaluation?
5. **Design utility:** does the contact representation improve the canonical
   CHIMERA co-design task, rather than only interface classification?

An overall epitope AUPRC increase without positive cognate-over-wrong-partner
margin is not success. A prediction changing after a partner swap is also not
success unless it changes in the correct direction.

## Current evidence boundary

The current FrameContact work contains:

- parameter-unshared antibody and antigen intra-chain encoders;
- distance, direction, full relative pose, spatial-neighbor, and
  sequence-only conditions;
- explicit cross-chain binary-contact or one-head distogram prediction;
- contact-derived and gated-hybrid residue heads;
- late and interleaved bidirectional invariant cross-attention;
- validation-only partner retrieval, wrong-partner controls, calibration,
  precision/recall, P@K/Recall@K, and topology diagnostics;
- a real-checkpoint Boltz-1 Pairformer adapter with explicit missing-frame and
  provenance receipts;
- a future-only, zero-gated paired-sequence adapter and a local Euclidean
  distance-matrix (EDM) feasibility regularizer.

The following claims are **not** currently supported:

- state of the art or “world first”;
- improvement on a sealed external test;
- apo/unbound structural generalization;
- improvement from PPLM features or the EDM loss after training;
- fair superiority to EpiFormer, ProTact, PPLM-Contact, Handshake, or a native
  Pairformer model;
- a fused CUDA/Triton performance result;
- a novel SE(3)-RoPE theorem.

Authenticated post-freeze engineering evidence now additionally shows that the
official PPLM 650M checkpoint can generate safe paired features for one real
SAbDab2 train-role antibody--antigen pair, and that the paired conditioner plus
local EDM path completes a CUDA/BF16 forward/backward smoke. Details and the
preserved failure trail are in
`reports/FRAMECONTACT_FRONTIER_ENGINEERING_EVIDENCE_20260801.md`. This does not
upgrade either component to a trained method or benchmark result.

## Closest 2024--2026 methods

| Method | Status | Directly relevant result/design | What FrameContact may reuse | What cannot be borrowed as evidence |
|---|---|---|---|---|
| [AsEP](https://proceedings.neurips.cc/paper_files/paper/2024/hash/15add6732964d5b1f0954058bf3ccc88-Abstract-Datasets_and_Benchmarks_Track.html) | NeurIPS 2024 Datasets & Benchmarks | antibody-specific epitope prediction as bipartite graph link prediction | contact-first task definition and official external benchmark | its official split is not automatically temporal, endpoint-disjoint, or a Foldseek structural-fold split |
| [EpiFormer](https://arxiv.org/abs/2606.04154) | 2026 preprint | interleaved bidirectional cross-attention; node, edge, distance and sparsity objectives | early-versus-late fusion comparison and explicit pair supervision | reported results do not establish performance under the present five-seed structural/temporal protocol |
| [ProTact](https://www.nature.com/articles/s42003-026-09809-2) | Communications Biology 2026 | independently invariant monomer encoders plus cross-protein tensor and trigonometry updates | pose-free pair-trunk principle driven by both monomers' intra-chain geometry | its 19-complex antibody transfer does not establish SAbDab2 structural novelty |
| [PPLM](https://www.nature.com/articles/s41467-026-70457-5) | Nature Communications 2026 | paired LM trained on >3.3M pairs; intra-chain RoPE and non-positional inter-chain attention; contact downstream model | paired residue embeddings and inter-protein attention as zero-gated optional evidence | checkpoint licensing, cutoff, overlap and current repository-license conflict require audit before comparison |
| [Handshake](https://www.biorxiv.org/content/10.64898/2026.06.04.730144v1) | 2026 preprint | ProstT5, LoRA, cross-chain attention and contact head; up to 32,503 pairs; redundancy-ablation | independently reimplement a partner-aware sequence-only control | general PPI benchmarks are not antibody-specific structural-fold evidence; the audited official revision has no license/checkpoint and its default split/contact-export paths are not suitable for a frozen comparison |
| [CALM](https://www.biorxiv.org/content/10.64898/2026.02.25.707916v1) | 2026 preprint | dual encoder plus cross-attentive decoder and bidirectional cognate retrieval | add Ab-to-Ag and Ag-to-Ab retrieval as a first-class task | 80%-identity clustering is much weaker than structural-fold novelty |
| [RaftPPI](https://openreview.net/pdf?id=Dp1RM3gPg8) | ICLR 2026 | factorized residue-kernel retrieval, ANN indexing and confidence-weighted hard negatives | mine hard wrong partners and report direction-specific Recall@K at screening scale | general PPI retrieval does not provide antibody-specific residue contacts or structural-fold evidence |
| [Germline-shortcut audit](https://www.biorxiv.org/content/10.64898/2026.07.21.739963v1) | 2026-07 preprint | random splits inflate immune-receptor retrieval by 15--28 percentage points; full-variable PLMs can exploit germline/framework information | add clone/CDR-cluster leakage audits and do not equate full-sequence retrieval with specificity | it is a receptor-similarity study, not an antibody--antigen contact benchmark |
| [Structural plausibility without binding specificity](https://www.biorxiv.org/content/10.64898/2026.03.02.709004v1) | 2026 preprint | 106 cognate complexes versus 11,342 shuffled pairs across AF3, Boltz-2 and Chai-1 | require explicit non-cognate ranking and calibrated partner scores | co-folding confidence and nanobody decoys are not a drop-in FrameContact baseline |
| [SAbDab2 ML Dataset](https://zenodo.org/records/20083995) | official dataset v0.1.0 | versioned sequence-connected antibody and antigen-aware splits with published thresholds | use official cluster authority plus Foldseek structural edges | official splits are sequence-based; `agclusters` is slash-separated per antigen chain and must be tokenized |
| [MSA Pairformer](https://www.sciencedirect.com/science/article/pii/S009286742600749X) | Cell 2026 | 111M MSA model with sequence--pair refinement, triangle operations and query-biased attention; trained on single chains but transfers to interface contacts | an MSA-enabled pair/contact upper baseline and evidence that pair states can be pretrained without complex coordinates | its interface-contact gain is not antibody-specific same-antigen retrieval evidence; MSA preprocessing and input information differ from the frozen-ESM budget |
| [LC-SEPLM](https://arxiv.org/abs/2607.22777) | 2026-07 preprint | ESM2 LoRA adapted with long-range contact supervision on 500,000 AlphaFold Swiss-Prot proteins | a future training-only contact-adapted sequence representation baseline | no direct partner-specific PPI evaluation; external structural pretraining requires cutoff, overlap and AlphaFold-label audits, and no auditable official code was found in this pass |
| [ABConformer](https://arxiv.org/abs/2509.23254) | 2025 preprint | iterative sliding attention between antigen and heavy/light chains; reports a precision--recall bandwidth trade-off | fixed-threshold trajectory and local-bandwidth sensitivity controls | its learned sequence-axis pseudo-position is not a physical 3D coordinate; phylogenetic fold construction is weaker than endpoint-disjoint structural-fold audit and sensitivity analysis is single-fold/single-seed |
| [EpiCLIP](https://openreview.net/forum?id=JImubudlk9) | ICML GenBio 2026 spotlight | approximate epitope/paratope candidates trained as dense retrieval | treat bidirectional specificity as listwise ranking and align training candidates with deployment candidates | patch-ranking evidence is not full residue-pair contact evidence; the public code path was not accessible in this audit |
| [DeepRank-Ab](https://www.nature.com/articles/s42003-026-10408-4) | Communications Biology 2026 | atom-level antibody-antigen decoy scoring on about 2.3M poses from 1,442 complexes | future pose-ensemble ranking baseline | assumes candidate complex poses and cannot be compared directly with unaligned-monomer contact prediction |
| [ConTact](https://arxiv.org/abs/2605.21600) | 2026 workshop/preprint | contact-then-act CDR design on CHIMERA | use contact-gated conditioning when connecting prediction to design | no public checkpoint/release/license was found in the audited repository; claims require retraining |
| [AgForce](https://arxiv.org/abs/2605.21610) | 2026 preprint | diagnoses antigen blindness and uses framework dropout, gated bottlenecks and antigen cycle consistency | treat partner usage as a causal failure mode; require wrong-partner and partner-evidence ablations in addition to cross-attention | it is a generative CDR-design method, so its reported gains do not establish residue-pair contact quality or structural-fold generalization |
| [CHIMERA](https://arxiv.org/abs/2603.13431) | 2026 workshop/preprint benchmark | canonical antibody--antigen co-design evaluation | final downstream design task and evaluator | its naming must not be converted into an unverified Foldseek structural-fold claim |
| [IgGM2](https://www.biorxiv.org/content/10.64898/2026.07.09.737510v1) | 2026-07 preprint | AF3-like Pairformer plus full-atom diffusion and immune-complex evaluation | longer-term pose/design comparison | no audited official checkpoint was available; full training used far more compute than the present project |
| [AAMFM](https://arxiv.org/abs/2607.20057) | 2026-07 preprint | ESM3 continual pretraining, antigen structure adapter and calibrated preference optimization | future antigen-conditioned representation/design study | current repository completeness and artifact custody are insufficient for a fair baseline |
| [Germinal](https://www.nature.com/articles/s41587-026-03187-0) | Nature Biotechnology 2026 | epitope-targeted design with low-n experimental validation | establishes that experimental binding and specificity are the ultimate utility standard | experimental success cannot be inferred from retrospective contact metrics |
| [SEPIA](https://www.biorxiv.org/content/10.64898/2026.04.17.719295v2) | 2026 preprint | >26M on/off-target affinity measurements and >75k variants | future hard-negative, mutation and calibration data | pseudo-structures and designed VHH targets require a separate domain-shift audit |

## 2026-08-01 literature delta and resulting decision

The newest papers strengthen the pair-representation direction, but do not
justify interrupting or changing the frozen development arrays.

- MSA Pairformer shows that sequence--pair refinement and triangle operations
  can learn interface-relevant structure even when pretraining examples are
  individual chains. This supports a pretrained pair-state baseline, not a
  claim that MSA coevolution solves antibody specificity.
- LC-SEPLM makes contact-supervised ESM adaptation a credible post-freeze
  experiment. It should be run only with a documented pretraining cutoff and
  explicit SAbDab2/AsEP decontamination. Its published downstream tasks do not
  establish residue-pair or partner-specific gains.
- ABConformer's precision gain comes with lower recall and uses learned
  one-dimensional pseudo-positions. The reusable lesson is to report full
  fixed-threshold trajectories and bandwidth sensitivity; it is not a solution
  to cross-chain coordinates for unaligned monomers.
- EpiCLIP and CALM make bidirectional retrieval a first-class task, but
  EpiFormer's own residue-level InfoNCE ablation decreased F1 by 0.029.  The
  post-freeze candidate therefore aligns cognate-versus-wrong-partner training
  with the 31-candidate retrieval panel while retaining contact/F1
  non-degradation gates; it does not assume contrastive learning is beneficial.

Accordingly, the highest-value next representation experiment after the current
selection freeze is a controlled two-axis comparison: frozen ESM versus a
training-only contact-adapted ESM, crossed with no pair trunk versus the same
small FrameContact pair trunk. This isolates whether any gain comes from the
single-chain representation or the partner-aware interaction module. An
MSA-Pairformer feature baseline is a separate, information-richer upper bound
and must be labeled as such.

The executable evidence requirements are recorded in
[`MSA_PAIRFORMER_BASELINE_AUDIT_20260801.md`](MSA_PAIRFORMER_BASELINE_AUDIT_20260801.md),
and the leakage-safe post-freeze factorial is specified in
[`CONTACT_ADAPTED_ESM_FACTORIAL_CANDIDATE.md`](CONTACT_ADAPTED_ESM_FACTORIAL_CANDIDATE.md).
The retrieval-objective diagnosis and implementation boundary are specified in
[`FRAMECONTACT_RETRIEVAL_ALIGNED_SPECIFICITY_20260801.md`](FRAMECONTACT_RETRIEVAL_ALIGNED_SPECIFICITY_20260801.md).

AgForce adds a complementary warning: architectural conditioning alone does
not prove that a prediction depends on the partner. FrameContact therefore
requires three distinct checks: a directed cognate-over-wrong-partner retrieval
margin, contact localization, and counterfactual removal of partner-interaction
evidence. The default-off contact bottleneck implements the third check using
positive counterfactual contact lift; it remains an untrained candidate and
does not inherit AgForce's generation results.

## 2026-08-01 specificity and split audit delta

The newest evidence changes the benchmark contract more than it changes the
network. The germline-shortcut study shows that a model can look strong in
retrieval while primarily using clonal or conserved framework information.
The shuffled-complex confidence study independently shows that structural
plausibility is not cognate specificity. RaftPPI shows that retrieval-scale
hard-negative mining and per-protein indexing are practical, but its factorized
score does not replace a residue contact map.

FrameContact therefore keeps two separate axes:

1. an interaction-localization axis, measured by contact AUPRC/P@K and
   epitope/paratope marginals;
2. a cognate-retrieval axis, measured in both partner directions with fixed
   wrong-partner panels and cluster-weighted margins.

The SAbDab2 v0.1.0 README was audited directly. In its antibody--antigen split,
CDRH3 is clustered at 95% identity, concatenated heavy/light CDRs at 100%,
protein antigens at 40%, and peptide/DNA/RNA antigens at 80%. Its separate
`ab_cluster` uses the antibody-only split authority (85% for each CDR grouping),
and `agclusters` contains one cluster per antigen chain separated by `/`.

The previous development antigen-fold split is exact-sequence and Foldseek-hard-
edge disjoint, but still has 8 antigen-cluster and 11 antibody-cluster values
crossing roles. A new label-free confirmatory split unions:

- reciprocal-min Foldseek TM-score hard edges at 0.50;
- exact PDB assembly, antigen sequence and antibody sequence equality;
- every token in the slash-separated SAbDab2 antigen cluster field; and
- the SAbDab2 antibody-only final cluster.

It has zero cross-role values for every enforced authority and zero Foldseek
hard edges. It is not called a germline or clonotype split because SAbDab2 does
not assign those semantics. The policy also produces one 2,584-instance
training component, so this is a strict confirmatory axis with a documented
component-imbalance caveat, not a silently improved version of the development
split. Exact artifacts and claim boundaries are in
[`FRAMECONTACT_DOUBLE_COLD_SPLIT_AUDIT_20260801.md`](FRAMECONTACT_DOUBLE_COLD_SPLIT_AUDIT_20260801.md).

## Representation decision

### Intra-chain geometry

For each residue, construct a right-handed local frame from real `N/CA/C` and
carry an explicit frame-valid mask. For a residue pair `(i,j)`, use only
independently rigid-transform invariant features:

- sequence separation and same-chain indicator;
- C-alpha distance expanded with RBFs;
- local direction `R_i^T(x_j-x_i)/||x_j-x_i||`;
- relative rotation `R_i^T R_j`;
- optional frame-point invariant distances and local surface/topology features.

These features can enter attention as head-specific logit bias, value gates, or
pair states. Raw global `xyz`, raw Euler angles, and a displacement between two
unaligned monomers are forbidden.

### Cross-chain representation

Before a relative complex pose exists, cross-chain information is restricted to
scalar/invariant evidence:

```text
h_Ab, h_Ag -> gated bidirectional cross-attention
             -> Z_ij = outer/bilinear fusion + pair prior
             -> distogram p(d_ij in bin k)
             -> contact probability sum_k<=cutoff p_k
             -> epitope/paratope marginals and partner score
```

Cross-chain sequential RoPE is not used. PPLM's separation of intra-chain RoPE
from non-positional inter-chain attention is a directly relevant precedent.

### Residue marginals to compare

The project must compare, without choosing on test data:

1. contact-only marginal from the partner dimension;
2. length-normalized contact marginal;
3. gated hybrid of unary self-attention residue logits and contact-derived
   logits;
4. unary-only propensity control.

True noisy-OR is sensitive to partner length. A length-normalized survival
ablation is allowed but must not be described as a probability of at least one
contact. Calibration must be assessed for each aggregation rule.

## New candidate: paired sequence plus invariant geometry

The future branch consumes frozen paired-protein features in addition to frozen
ESM residue embeddings:

- separate antibody and antigen projection parameters;
- optional PPLM inter-attention projected into a cross-chain pair prior;
- every new path initialized with an exact zero gate so the established model
  is reproduced before optimization;
- safe, hash-bound `npz` artifacts loaded with `allow_pickle=False`;
- checkpoint revision, weight hash, sequence identity, tensor shape, training
  cutoff and overlap receipt required for every cache.

This branch tests whether paired pretraining adds partner information beyond
local monomer geometry. It is not a PPLM baseline until the official artifacts,
license and decontamination questions are resolved.

## New candidate: pose-free local EDM consistency

Reverse triangle inequalities are necessary but can accept an unrealizable,
nearly constant cross-distance block. On a prediction-routed CDR-by-antigen
surface patch, combine known intra-chain distances and predicted cross-chain
distances into a complete squared-distance matrix `D^2`. With
`J = I - 11^T/n`, define

```text
B = -0.5 J D^2 J.
```

For a realizable three-dimensional Euclidean distance matrix, `B` must be
positive semidefinite and have rank at most three. The loss penalizes negative
eigenvalues and positive eigenvalues outside the largest three. It uses no
cross-monomer coordinate subtraction and is invariant to separate rigid
transforms of the two monomers.

This is a candidate inference from standard EDM theory and pair-trunk methods,
not an established novelty claim. Promotion requires five-seed development
evidence, overhead measurement, a broader prior-art search, and no external-test
feedback.

## Experimental ladder

### Stage 0: engineering invariants

Required tests before training:

- exact zero-gate reproduction of the previous model;
- binary-contact/distogram projection consistency;
- gradients reach paired conditioning, contact head and EDM objective;
- predictions unchanged after independent Ab and Ag rotations/translations;
- missing/degenerate frames are masked with finite outputs;
- BF16 forward/backward smoke and FP32 eigenvalue computation;
- source-tree manifest, environment, data-cache and output receipts.

### Stage 1: development-only model selection

Use at least five preregistered seeds. Compare on identical train/validation
roles:

1. dense sequence-only;
2. spatial topology;
3. distance RBF;
4. distance plus local direction;
5. full relative pose;
6. binary FrameContact;
7. distogram FrameContact;
8. interleaved distogram FrameContact;
9. paired-sequence zero/active gate;
10. local triangle and local EDM ablations.

Record every epoch:

- pair macro and pooled AUPRC/AUROC;
- P@Lmin, P@Lmin/2, P@Lmin/5 and matched Recall@K;
- epitope/paratope macro and pooled AUPRC, F1, MCC, precision, recall and
  predicted-positive ratio at a validation-fixed threshold;
- Brier score, ECE, calibration slope/intercept and global temperature;
- reciprocal partner margin and retrieval R@1/R@5/MRR;
- same-antigen/different-antibody directed epitope migration;
- attention entropy, spatial-neighbor enrichment, head reliance and component
  fragmentation;
- GPU peak memory, examples/s, compile/fused-kernel status and failures.

The primary checkpoint selector is the minimum reciprocal cognate-partner
margin, with contact macro AUPRC as a preregistered secondary objective. A model
that improves marginal propensity while losing partner specificity is rejected.

### Stage 2: intervention suite

All interventions use fixed trained checkpoints:

- wrong-partner and same-antigen partner swaps;
- geometry shuffle within length/chain-matched complexes;
- local-frame rotation corruption;
- spatial-neighbor graph rewiring;
- sequence-preserving coordinate perturbation;
- unary/contact branch knockout;
- paired-sequence embedding and inter-attention knockout;
- triangle/EDM branch knockout;
- holo-chain versus experimental apo or predicted-monomer structures.

Random pairs are diagnostic controls, not automatically labeled non-binders.
Unknown binding is a real confound; curated noncognate or competitive partners
are preferred.

### Stage 3: frozen evaluation

Freeze before opening any held-out label:

- source revision and manifest;
- data/split hashes and exclusions;
- five seeds, selected checkpoint and ensemble rule;
- threshold, temperature, marginalization and patch-routing rule;
- partner panel and scorer code;
- external-data decontamination report.

Then open once:

1. the newly frozen SAbDab2 sequence-cluster plus structural double-cold holdout;
2. SAbDab2 temporal holdout;
3. official AsEP test using its canonical evaluator;
4. external wrong-partner/specificity panel;
5. apo/predicted-monomer panel when data are available.

Report paired cluster bootstrap confidence intervals and seed dispersion. Never
select a model because of frozen-test results.

### Stage 4: canonical downstream task

Use the selected contact representation to gate antibody CDR co-design on
CHIMERA. Compare sequence recovery, structural quality, epitope awareness,
specificity/retrieval and diversity. This is the bridge from a classifier to an
antibody-engineering contribution.

## Strong baselines and comparability boundary

Priority order:

1. same-code dense ESM and partner-aware geometry-ablated controls;
2. a 2 x 2 frozen-representation factorial separating ESM/ProstT5 from
   geometry-ablated/geometry-aware interaction modeling, after ProstT5 custody;
3. authenticated real Boltz-1 Pairformer adapter on the identical split;
4. DeepSSInter external-checkpoint inference and then identical-task retraining
   only if its license, preprocessing and upstream data can be aligned;
5. ProTact retraining/evaluation when its data, 6 A label, crop policy and
   preprocessing can be aligned;
6. EpiFormer retraining with the same features, seeds and strict roles;
7. Handshake or PPLM sequence-only/paired controls after license and overlap
   audit;
8. published numbers only in a clearly marked non-comparable literature table.

Numbers from different contact cutoffs, residue scopes, cluster thresholds,
test sets, monomer sources or calibration procedures must never be placed in a
single “winner” column without those differences.

The source-level Handshake decision, including pinned revision, licensing,
split, deterministic-inference and clean-room baseline requirements, is in
[`HANDSHAKE_BASELINE_AUDIT_20260801.md`](HANDSHAKE_BASELINE_AUDIT_20260801.md).
The joint Handshake/ProstT5/DeepSSInter/ProTact authority and contact-definition
audit is in
[`FRONTIER_CONTACT_BASELINE_AUDIT_20260801.md`](FRONTIER_CONTACT_BASELINE_AUDIT_20260801.md).
The exact same-code coordinate ablation is specified in
[`FRAMECONTACT_GEOMETRY_ABLATED_CONTROL_20260801.md`](FRAMECONTACT_GEOMETRY_ABLATED_CONTROL_20260801.md).

## Innovation gates

### Candidate A: local-pose multi-basis attention bias

Mix global, sequential-local, spatial-local and frame-pose bases per attention
head with learnable nonnegative gates. Promote only if head/basis ablation shows
stable complementary behavior across seeds and structural holdouts.

### Candidate B: pair-conditioned local-pose rotary score

Use relative local-frame pose to rotate/query scalar subspaces or modulate the
query-key score. This requires a proof/test of global SE(3) invariance,
missing-frame fallback, and comparison against an equal-parameter MLP bias.
Do not market the name “SE(3)-RoPE” before these conditions and a prior-art
search are satisfied.

### Candidate C: local EDM feasibility

Promote only if it raises contact macro AUPRC and P@K, reduces isolated false
positive components, retains at least 95% of baseline recall, preserves partner
margin, remains numerically stable, and adds acceptable wall-time/memory.

### Candidate D: latent pose mixture

A single expected distogram can average incompatible binding modes. A later
model may predict `K` pose/contact modes and a calibrated mixture weight. This
is justified only if multimodal complexes or poor EDM feasibility remain a
measured failure mode.

### Candidate E: topology-routed sparse pair trunk

Restrict expensive updates to CDR residues and prediction-routed antigen surface
patches, with a global compatibility branch retained. Compare accuracy and
compute against dense pair updates at equal training budget. Labels may not be
used to route validation or test patches.

## Research-to-industry value

The strongest defensible portfolio positioning after the planned evidence is
**AI for Science / ML Research Engineer for protein interaction and antibody
design**. The work becomes Research Scientist-level evidence only after it has
an original, well-isolated contribution, a strong external comparison, a
first-author-quality manuscript, and preferably experimental or canonical
design validation.

High-value deliverables are:

- a public reproducibility package with immutable manifests and one-command
  evaluation;
- a leakage-audited structural/temporal benchmark and bootstrap reports;
- an original interaction-modeling method with causal controls;
- a real pretrained pair/foundation-model integration with provenance;
- a genuine Triton/CUDA fused pair/triangle kernel with Nsight and roofline
  evidence on A100 and RTX 4090, not merely `torch.compile`;
- multi-GPU scaling evidence and cost/throughput accounting;
- a canonical co-design result and, ideally, prospective binding validation.

This covers scientific modeling, data/benchmark rigor, distributed/GPU systems,
and reproducible engineering. It does not make every AI role automatic; claims
must remain tied to measured artifacts.

## Promotion and stop rules

Promote a component only if all of the following hold on development roles:

1. positive reciprocal partner margin in every seed or a preregistered robust
   aggregate with confidence interval above zero;
2. non-inferior contact macro AUPRC and calibration;
3. directed same-antigen epitope migration rather than arbitrary output change;
4. causal drop under the corresponding intervention;
5. acceptable compute/memory overhead;
6. frozen provenance and no test leakage.

Stop or simplify when:

- the method only improves unary epitope propensity;
- the branch learns a zero/ignored gate across seeds;
- gains disappear under structural novelty;
- calibration or partner ranking degrades;
- a simpler equal-parameter control matches it;
- artifact licensing or train/test overlap cannot be resolved;
- the compute cost prevents a complete five-seed comparison.

## Immediate execution order

1. Finish the running five-seed binary/distogram development arrays without
   changing their source or protocol.
2. Validate the paired-sequence and EDM paths with a hash-bound GPU smoke.
3. Freeze and run a small development pilot for paired features and EDM after
   the current model selection is locked.
4. After architecture selection, retrain the fixed winner on the tokenized
   SAbDab2 sequence-cluster plus Foldseek double-cold roles; leave its test role
   unopened until the new validation selection is sealed.
5. Complete the external decontamination ledger.
6. Run the real Pairformer, EpiFormer and ProTact comparisons only where the
   same evaluation contract can be enforced.
7. Freeze the winning architecture, calibration and partner panel.
8. Open SAbDab2 and AsEP frozen evaluations exactly once.
9. Use the selected contact representation in CHIMERA co-design.
10. Add fused-kernel and multi-GPU work only after the scientific computation is
   stable enough to benchmark honestly.

## Bottom line

The nearest high-value result is not a surprising overall AUPRC number. It is a
reproducible demonstration that a contact-first, independently invariant model
prefers the cognate antibody, moves the epitope to the correct region, and
retains that advantage on unseen antigen structures. Paired pretraining, local
EDM consistency and sparse topology routing are the most defensible next
candidates, but each remains an ablation until the registered evidence ladder is
complete.

## Systems research update: fused attention is useful but not the paper claim

PyTorch's official SDPA dispatcher can consume the materialized additive
per-head bias used by FrameContact.  On the authenticated RTX 4090 diagnostic,
it selected the memory-efficient CUTLASS backend and reduced full synthetic
training-step time by `4.2-7.2%` and peak allocated memory by `6.1-8.5%` across
the tested 220 x {128, 512, 1000} residue shapes.  An independent AB03 A100
replication (PyTorch `2.5.1`, CUDA `12.4`) passed FP32/BF16 output and all
query/key/value/bias-gradient checks.  Across 220 x 128, 220 x 512 and the
observed-size boundary 231 x 2363, it reduced allocated training peak memory by
`6.1-8.6%`; median synthetic train-step speedup ranged from `1.007x` to
`1.068x`.  The long boundary is therefore a memory result, not a meaningful
latency win.  This is now an optional, default-off execution path.  It does not
change the scientific architecture and it is not a custom fused-kernel result.

The newer official PyTorch FlexAttention work is relevant for the next systems
stage because it supports score modifications and block masks inside compiled
attention kernels:

- <https://pytorch.org/blog/flexattention/>
- <https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/>

However, FrameContact's current spatial kNN graph is irregular and the code
still materializes dense pair geometry before attention.  A block mask alone
may therefore leave the main memory/computation cost unchanged.  The defensible
kernel roadmap is:

1. profile dense frame construction, pair-feature projection, attention and
   distogram heads separately on authenticated long train complexes;
2. represent the spatial branch as explicit `O(Lk)` neighbor edges while
   retaining a dense/global sequence branch;
3. compare SDPA, FlexAttention block sparsity and a custom Triton segment-softmax
   kernel at identical values and gradients;
4. report Nsight kernel traces, achieved bandwidth/occupancy, end-to-end train
   throughput and A100/RTX 4090 scaling;
5. keep the reference implementation as the numerical oracle.

This systems work is high-value for an ML-systems portfolio, but the manuscript
must remain centered on partner-specific interaction under structural novelty.
Speeding up a model that has not passed the partner-specificity gate cannot
substitute for the biological result.
