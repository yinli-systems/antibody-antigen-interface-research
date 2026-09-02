# ABF-PairContact v1 research plan

Status: research plan only; this file is not execution authority.

## Decision

The next high-value intervention is interaction-aware pretraining that explicitly
targets partner-specific residue-pair signal. It should replace further readout,
pooling, static embedding-mixture, or marginal reweighting experiments.

The first scientific canary should use the completed 650M ABF30B checkpoint. A
3B transfer is justified only after the 650M mechanism gate passes and a final,
audited 3B Stage-A0 checkpoint exists.

## Evidence for the decision

The frozen development-only beyond-marginals audit reports:

- full contact mean-per-complex AP: `0.0276837`;
- unary-only contact mean-per-complex AP: `0.0247725`;
- gauge-fixed residual AP: `0.00584595`;
- raw-interaction AP: `0.00202999`;
- median partner-specific variance share by seed: `0.01403`, `0.02387`, and
  `0.00972`.

Thus most of the current contact score is explainable by antibody and antigen
residue propensities rather than cognate-pair compatibility. The full-development
energy-balanced embedding experiment independently failed promotion: its median
contact delta was `-0.0022060`, although epitope improved. Surface weighting,
hard-negative loss variants, confidence filtering, pooling variants, and CDR-only
adaptation have likewise not produced a robust three-task promotion.

The external literature points to the same missing ingredient:

- PPLM jointly encodes paired sequences with hybrid intra/inter-chain attention
  and trains on more than 3.3 million protein pairs.
- MINT adds cross-chain attention to ESM-2 and performs pair-aware MLM on a
  cluster-filtered 95.8 million-pair STRING corpus.
- PLM-interact jointly optimizes pair MLM and interaction classification rather
  than relying on frozen single-chain embeddings.
- SMP constructs pseudo-dimers from monomer structures and pretrains contact or
  docking objectives before real-dimer fine-tuning.

Primary sources:

- https://doi.org/10.1038/s41467-026-70457-5
- https://doi.org/10.1038/s41467-025-67971-3
- https://doi.org/10.1038/s41467-025-64512-w
- https://doi.org/10.1038/s41467-026-73885-5

## Existing implementation to preserve

Do not restart Stage-A1 from scratch. The repository already contains and has
audited the following pieces:

1. a joint pair-aware view for MLM/contact learning;
2. a chain-independent view that excludes all cross-chain keys for retrieval;
3. actual partner-swapped counterfactual forwards;
4. known-positive, source-cluster, and length-matched negative controls;
5. an eight-GPU, ten-update finite optimization and DCP save/restore systems
   canary.

The old cognate-joint-embedding retrieval objective is prohibited because it can
identify the partner from leaked context. The chain-independent retrieval view
is mandatory.

## Phase 0: freeze the canary authority

Before any GPU training, freeze hashes for:

- the completed ABF30B checkpoint (`c36e0b1d...ab3ea` is the current candidate,
  but the full hash must be copied from the immutable audit);
- source code and environment;
- train-only pair and structure-contact manifests;
- endpoint-cluster and known-positive exclusion maps;
- candidate-sampler seed;
- discovery split, confirmation split, and all metric definitions;
- the three arms and token/update budgets below.

No old SAbDab2 test, SABLE labels, fresh156, or temporal labels may be loaded.
The 549 SAbDab2 validation complexes are historically exposed development data,
not a blind test.

## Phase 1: 650M mechanism canary

Use identical initialization, batches, token budgets, optimizer settings, and
random seeds for all arms:

1. **Pair-MLM control**: joint pair-aware MLM only.
2. **Retrieval arm**: pair MLM plus bidirectional InfoNCE from the
   chain-independent view, using actual partner-swapped forwards.
3. **PairContact arm**: retrieval arm plus contact-aligned pretraining.

The PairContact arm should decompose each pair logit as

`z_ij = a_i + b_j + r_ij`,

where `a_i` and `b_j` model unary contact propensity and `r_ij` is row/column
centered. Train the full logit with the existing contact loss and train `r_ij`
with a within-complex ranking loss that contrasts a true contact against
row-matched and column-matched noncontacts. This directly penalizes solutions
that improve only the marginals.

Contact-aligned pretraining data should be introduced in two steps:

- pre-cutoff pseudo-dimers built from monomer structures, following the SMP
  split-and-merge proxy;
- pre-cutoff real heterodimers and SAbDab2 training complexes.

The first correctness canary uses one seed and a small frozen slice. Only after
it passes determinism, finite-loss, partner-swap, and checkpoint-restore checks
does the three-seed fixed screen run.

## Phase 2: frozen development transfer

Train the same FrameContact downstream architecture on representations from all
three arms. Use an antigen-cluster-disjoint discovery split constructed only
from the 2,565 training complexes. Freeze the selected arm and checkpoint before
one disclosed evaluation on the 549 historically exposed development validation
complexes.

The mechanism and performance gates to freeze before training are:

- median partner-specific variance share at least `0.05`;
- median gauge-fixed residual AP at least `0.010`;
- median contact mean-per-complex AP delta at least `+0.003` versus the
  same-budget Pair-MLM control;
- no seed contact delta below `-0.002`;
- median epitope and paratope deltas no worse than `-0.005`;
- no authority, provenance, finite-loss, or test-access failure.

Also report pooled AP, PDB-equal AP, antigen-cluster-equal AP, Hits@1, reciprocal
rank of the first positive, precision/recall at `L/5` and `L`, AUROC, Brier
score, and calibration. A gate passing only marginal AP but failing residual AP
does not authorize scaling.

## Phase 3: scale only after mechanism success

If the 650M PairContact arm passes, port the exact frozen objectives and data
authorities to the final 3B ABPairFM Stage-A0 checkpoint. Run a bounded scaling
ladder before production:

1. 10-update graph and DCP canary;
2. 1% token-budget loss/throughput canary;
3. 10% token-budget three-seed mechanism screen;
4. full continuation only if the mechanism and downstream gates remain true.

Use only `gpu_4090` under the current project authority. GPU utilization is a
systems gate, not scientific evidence.

## Evaluation boundary

All current public and historical cohorts are development or exposed
measurement. Final confirmation must wait for the preregistered post-cutoff
antibody-and-antigen double-cold cohort. The v2.1 screen uses exact RCSB initial
release dates and currently has zero eligible post-cutoff candidates. Model,
checkpoint, router, metrics, exclusion rules, and opening procedure must remain
frozen while waiting.

## Immediate implementation order

1. authenticate the final ABF30B checkpoint and copy the existing split-view
   Stage-A1 source into a new immutable candidate tree;
2. materialize a small pre-cutoff SMP pseudo-dimer/contact slice and audit
   decontamination;
3. implement the explicit `a_i + b_j + r_ij` head and row/column-matched ranking
   loss with synthetic unit tests;
4. run CPU and one-GPU correctness canaries;
5. freeze the three-arm protocol and submit the fixed three-seed screen;
6. scale to 3B only after the 650M gate passes.
