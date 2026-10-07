# Strict dual-axis antibody–antigen partner retrieval v1

**Status:** frozen algorithmic preregistration. This document freezes the scientific question, split semantics, models, metrics, budgets, and stopping rules. It does **not** claim that a physical split assignment has passed audit. Model execution is forbidden until `freeze_split.py` authenticates a create-only builder artifact and its independent audit and emits `FROZEN_SPLIT.json`.

**Frozen:** 2026-09-30  
**Parent research commit:** `a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc`

## 1. Scientific question

Under a prospectively frozen, foldwise direct-80/80, four-view, dual-axis homology-separated C3/S2-like boundary, does **explicit residue-contact/correspondence supervision** improve finite-library antibody–antigen partner retrieval over an otherwise tensor-identical pair-aware reranker?

The **primary causal comparison is R2 − R1**. R1 and R2 have the same retrieval architecture, parameter tensors, inputs, seeds, pair exposure, optimizer schedule, candidate rerank set, and inference function. The only intended difference is the auxiliary contact-loss weight during training:

- `R1_pair_reranker_contacthead_w0`: contact head instantiated, loss weight exactly `0.0`;
- `R2_pair_reranker_contacthead_w010`: identical model, contact loss weight exactly `0.10`.

The retrieval score never directly consumes contact labels or predicted contact probabilities.

## 2. Evidence boundary

This experiment is **authority-train development**. It must not read the historical authority-validation pool, frozen SAbDab2 test, post-cutoff OOD outcomes, affinity labels, or any sealed benchmark during split building, model fitting, model selection, calibration, or repair.

The existing 14/14 rejected post-cutoff OOD candidates remain rejected. Their gate is not relaxed.

Unobserved antibody–antigen cross-pairs are **unlabeled ranking candidates**, never “non-binders” or “negatives.” Observed validation pairs are positives for finite-library retrieval. Any loss that places unlabeled candidates in a denominator is a ranking/contrastive surrogate, not a biological non-binding assertion.

## 3. Source authority and prospective firewall

The pinned authority-train snapshot recorded by the existing split protocol contains:

- 3,229 source rows;
- 1,830 unique exact antibody–antigen pairs;
- 1,383 canonical antibody entities;
- 1,172 canonical antigen entities;
- authority-train membership SHA-256 `442095581ae9bf3c7440c1a818612298b5f81e5a62ef46f431e0867b9c8b466b`.

Physical input paths and physical SHA-256 values must be bound at execution. A count match without byte hashes is insufficient.

The model process for fold `h` may mount only entities and source rows in that fold's exact allowlists. Whole-pool embeddings, indexes, contact maps, statistics, or caches are forbidden even if code intends to filter them later.

## 4. Dual-axis split

The split implementation is the existing reviewed pair:

- `scripts/build_train_only_nested_neighborhood_folds.py`;
- `scripts/audit_train_only_nested_neighborhood_folds.py`.

A physical split is authorized only when both are pinned to exact source bytes and the independent audit passes.

### 4.1 Canonical entities

Antibody entity: ordered tuple `(VH, VL)`, including genuinely missing VL.  
Antigen entity: full antigen amino-acid sequence.  
Exact pair: canonical antibody ID × canonical antigen ID.

All source rows belonging to one exact pair are indivisible.

### 4.2 Four similarity views

Antibody:
1. VH;
2. non-empty VL;
3. heavy-then-light concatenation.

Antigen:
4. full antigen sequence.

A direct homology edge exists when exact global alignment satisfies both:

- residue identity ≥ 0.80;
- reciprocal coverage ≥ 0.80 on **both** full sequences.

Boundary equality counts as an edge.

The pinned implementation uses global Needleman–Wunsch-family alignment, Biopython 1.83, BLOSUM62, gap-open 10.0, gap-extension 1.0, with terminal gaps penalized identically.

### 4.3 Fold semantics

Three outer folds. Each outer training universe contains three inner folds.

For a fold, validation antibody and antigen endpoints are selected prospectively using sequence graph and observed positive-pair incidence only. Every direct homology neighbor of any validation endpoint is closed from that fold's training inventory. A pair is:

- **validation** iff both endpoints are assigned to the validation endpoint sets;
- **training** iff both endpoints are training-eligible after direct-neighborhood closure;
- **quarantined** otherwise.

Long transitive similarity chains do not force global component equality.

No assignment can be redrawn after any model score for that assignment is opened.

### 4.4 Mandatory retention/discard disclosure

Every fold receipt must expose, separately:

- universe/training/validation/quarantined exact-pair counts;
- universe/training/validation antibody counts;
- universe/training/validation antigen counts;
- source-row counts by role;
- pair retention fraction;
- antibody endpoint retention fraction;
- antigen endpoint retention fraction;
- quarantine/discard fraction;
- query counts in both directions;
- candidate-library sizes;
- positive counts per query;
- finite-library random R@1/R@5/R@10 chance.

This is mandatory because strict two-axis splitting can make interactions unassignable.

## 5. Frozen retrieval libraries

For every authorized fold, the evaluation library is created **only from validation entities**.

### Ab→Ag
For each validation antibody query:
- candidates = every validation antigen in that fold;
- positives = all observed validation exact pairs incident on that antibody;
- every other candidate = unlabeled ranking candidate.

### Ag→Ab
Symmetric rule:
- candidates = every validation antibody;
- positives = all observed validation exact pairs incident on that antigen.

Queries with zero positives are invalid and fail the split freeze.

Canonical entity digest is the tie-breaker for exactly equal scores.

The library, query lists, positive incidence matrix, and chance values are serialized and SHA-256 bound before any model evaluation.

## 6. External-method audit and admissible inputs

### CALM
CALM-1.0 reports both full-sequence retrieval and a masked paratope/epitope variant. Its methods state that the masks are generated from structure by marking residues within 5 Å of the interface, and those masks are applied to encoder inputs for the masked retrieval experiment. Therefore:

- CALM **full-sequence** retrieval is a relevant external protocol reference;
- CALM masked Pa/Ep retrieval is **not an admissible test-time baseline** for this experiment because it consumes true structural interface labels;
- no validation/test epitope, paratope, or contact mask may enter H0/D0/L1/R1/R2 inference.

### Ab-CASLR
Ab-CASLR is used as design motivation for L1: CDR-aware antibody local representation and late interaction under strict OOD retrieval. Its published library and split differ from ours; its R@K values are not numerically compared against ours.

### DataSAIL
DataSAIL motivates controlling both entity axes and explicitly disclosing interactions that become unassignable when endpoints are assigned incompatibly. This experiment uses its own direct-neighborhood fold protocol and does not claim to be a global DataSAIL S2 split.

## 7. Model ladder

All trainable models use the same fold-local frozen residue embeddings unless a model explicitly declares otherwise:

- antibody: pinned AntiBERTy residue embeddings, VH and VL encoded separately;
- antigen: pinned ESM-2 `esm2_t33_650M_UR50D` residue embeddings;
- no validation/test contact masks;
- no predicted bound complex;
- no affinity labels.

Seeds for every trainable model: **17, 42, 73**. No best-seed selection.

### H0a — k-mer pair-homology transfer
No training.

For query/candidate pair ((a,g)), score is the maximum over observed training pairs ((a',g')) of

[
\sqrt{s_A(a,a')\,s_G(g,g')}.
]

`s_A` is the maximum of 3-mer Jaccard over VH, non-empty VL, and concatenated antibody views, with missing-view comparisons omitted. `s_G` is antigen 3-mer Jaccard.

### H0b — exact-NW pair-homology transfer
Same pair-transfer formula, replacing k-mer similarity with the pinned exact global-alignment similarity:

[
s(x,y)=\mathrm{identity}(x,y)\times\min(\mathrm{coverage}_x,\mathrm{coverage}_y).
]

For antibody similarity, take the maximum across legal antibody views. This baseline is allowed to be slow; it is a no-training upper-strength homology-transfer control.

### D0 — global contrastive dual encoder
- project antibody and antigen residues separately to width 128;
- masked length mean;
- L2 normalize;
- score = dot product / temperature;
- symmetric multi-positive contrastive loss;
- 58 epochs, fixed;
- no early stopping or checkpoint selection;
- same pair exposures for all three seeds.

### L1 — sequence-only CDR-aware local late interaction
Purpose: strong local-retrieval baseline without test interface labels.

- same frozen AntiBERTy/ESM-2 inputs as D0;
- CDR residues are identified **from antibody sequence/numbering only**, never from bound structure/contact labels;
- six possible antibody CDR summaries: H1/H2/H3/L1/L2/L3; absent chains/regions contribute no slot;
- project antibody CDR residues and antigen residues to width 128 and L2 normalize;
- each CDR slot is the mean of projected residues in that CDR;
- per-slot antigen compatibility is normalized log-mean-exp over **all antigen residues**, not a true-epitope subset;
- pair score = mean over present CDR-slot compatibilities;
- symmetric multi-positive contrastive loss;
- 58 epochs, same seeds and same training-pair exposure as D0;
- no epitope mask or contact label in training or inference.

A failure to number an antibody deterministically is fail-closed for L1 and is disclosed; it does not modify the fold.

### R1/R2 — tensor-identical pair-aware reranker
Both use the exact same modules:

- frozen fold-local D0 coarse checkpoint;
- frozen residue embeddings projected to width 256;
- two pre-LN bidirectional cross-attention blocks;
- 8 heads;
- FFN width 512, GELU;
- residual and attention dropout 0.10;
- pair representation `[m_A, m_G, m_A*m_G, |m_A-m_G|]`;
- pair head `Linear(1024,256) → GELU → Dropout(0.10) → Linear(256,1)`;
- final retrieval score `s_coarse + 0.5*tanh(r)`;
- contact head present in **both** R1 and R2:
  - separate 64-d projections;
  - pair contact logit = scaled dot product + bias.

Evaluation rerank set = union of coarse top-64 antigens per antibody and coarse top-64 antibodies per antigen. Positives are **not** force-added at evaluation.

Training may add observed positives to the training candidate set solely so the multi-positive loss is defined.

R1 and R2:
- exactly 40 epochs;
- same seed;
- same initialization seed and complete module parameterization;
- same batch size 32 exact pairs;
- AdamW lr 1e-4, wd 0.01, betas (0.9,0.98), eps 1e-8;
- gradient clip 1.0;
- 5% linear warmup then cosine decay;
- no early stopping;
- same number/order of optimizer steps;
- same coarse D0 bytes.

Only difference:
- R1 contact-loss weight = **0.00**;
- R2 contact-loss weight = **0.10**.

The contact head is instantiated in R1 as well, ensuring identical parameter count. Contact logits never enter retrieval score.

### Contact auxiliary labels
Available only for source rows belonging to the fold-local training pair inventory.

A residue pair is a contact iff any resolved non-hydrogen atom pair is within **4.5 Å**. Unresolved/ambiguous residue pairs are masked. Contact labels are forbidden in split construction, H0/D0/L1, validation input, test input, candidate selection, threshold selection, or retrieval scoring.

## 8. Primary endpoints

Reported separately for both directions and equally weighted bidirectional mean:

- query-level MRR;
- R@1;
- R@5;
- R@10.

No pooled binary AUROC/AUPRC is a partner-retrieval primary endpoint.

For each query with (m) positives among (N) candidates, exact random hit probability at K is

[
1-\binom{N-m}{K}/\binom{N}{K},
]

with the usual truncation when (K>N). Fold-level chance enrichment is observed R@K divided by the mean exact random hit probability over its queries.

## 9. Inference and uncertainty

Every model emits the complete validation score matrix before metrics are opened.

Intervals:
1. antibody-query bootstrap for Ab→Ag;
2. antigen-query bootstrap for Ag→Ab;
3. evolutionary-component bootstrap;
4. crossed antibody×antigen bootstrap for bidirectional effects;
5. paired seed × outer-fold/component bootstrap for model differences.

Primary registered effect:
**R2 − R1** for bidirectional MRR, R@1, R@5, R@10.

Secondary effects:
- L1 − D0;
- R1 − L1;
- D0/L1/R1/R2 − H0a/H0b.

All intervals are conditional on this finite benchmark and split construction.

## 10. Contact/interface diagnostics

Contact/interface metrics are mechanistic secondary endpoints and are reported separately from retrieval.

For R1/R2, report:
- macro per-complex contact AP;
- epitope AP;
- paratope AP.

Also report correlation/dissociation between:
- retrieval score deltas;
- contact AP deltas;
- candidate-library size;
- positive count;
- score mean/variance and calibration;
- query/candidate lengths.

Do not infer binding affinity from these scores.

## 11. Compute budget and stopping rules

Before submission a machine-readable budget receipt must bind actual requested GPUs and estimated GPU-hours.

Default upper bounds, subject to cluster policy:
- split/H0: CPU only;
- D0: ≤1 GPU per seed/fold training;
- L1: ≤1 GPU per seed/fold training;
- R1/R2: ≤1 GPU per seed/fold training unless memory preflight proves impossible; any change requires an amendment **before scores are opened**.

No model family may receive more epochs or training pair exposures because another model underperformed.

Stop immediately on:
- split or independent-audit mismatch;
- any validation/test/OOD label read during split/model fitting;
- missing physical input hash;
- fold-local cache inventory mismatch;
- NaN/nonfinite training;
- score-matrix/query/library mismatch;
- R1/R2 parameter-name/shape/count mismatch;
- R1/R2 optimizer-step or candidate-set mismatch;
- test access;
- duplicate experiment hash with an existing complete receipt.

OOM/numerical failures are preserved and may be repaired only without changing the scientific contract.

## 12. One-time structural representation decision

The ongoing 36-task SaProt/ProSST sweep remains independent. No structural-PLM arm enters retrieval until all 36 tasks, matched random controls, and registered aggregation complete.

Exactly one post-sweep decision is allowed:
- if a real structural layer shows a registered positive effect over its exact-width/coverage matched random control, freeze one secondary representation arm before retrieval execution;
- otherwise no structural-PLM retrieval arm is added.

No additional PLM search follows a negative decision.

## 13. Future evidence hierarchy

After this authority-train nested experiment:
1. blind admission of a genuinely untouched OOD cohort; existing 14/14 rejects stay rejected;
2. experimentally measured affinity/specificity ranking on a separately defined dataset;
3. co-folding with equal sampling budgets: unconstrained vs predicted-epitope vs oracle-epitope, where oracle is explicitly an upper-bound control.

These are separate endpoints and must not be mixed into the retrieval claim.
