# EPIC camera-ready reviewer-response plan

## Scope and evidence policy

EPIC was accepted, with the decision noting that stronger ablations and robustness analysis were needed. This package preserves the accepted scientific core and addresses every reviewer point through one of three explicit mechanisms:

1. **camera-ready correction** when the submission text was factually inaccurate or insufficiently scoped;
2. **post-submission evidence** when a frozen, auditable analysis already exists on GPU/repository authorities;
3. **claim moderation / unresolved boundary** when the exact submitted checkpoint or input authority is unavailable and a stronger statement would be unsupported.

Post-submission experiments are never silently rewritten as if they had been part of the submission. Historical/exposed test sets are labelled as such, and later FrameContact evidence is separated from exact-EPIC evidence when the architecture or structure predictor differs.

## Decision-level response

We agree with the decision. The revised presentation no longer frames EPIC as winning every task. It foregrounds threshold-free average precision under extreme imbalance, identifies the epitope recall/ranking trade-off, separates imported literature numbers from same-protocol measurements, and states that individual components are established techniques whose contribution is the integrated multi-resolution formulation and its controlled evidence—not a claim that each block is independently novel.

## Reviewer VUfK

### V1. Extreme residue-pair imbalance

**Concern.** The contact task has 0.0705% positives in training, making AUROC and raw recall difficult to interpret.

**Response.** We make contact AUPRC / mean-per-complex AP the primary contact-ranking quantity, place prevalence adjacent to every contact result, and add practical per-complex ranking readouts: P@10/20/50/100, P@Lmin, precision at the true-contact count, recall at those budgets, top-1%-candidate enrichment, precision at fixed recall, recall at fixed precision, Brier score, ECE, and complex-bootstrap intervals. We report both pooled and equal-complex estimands and do not compare them as interchangeable quantities.

The accepted Table 1 remains historically preserved; the new practical analysis is labelled a post-submission audited retrain on the official 455 split, because its frozen geometry authority uses AF3 rather than the submitted OpenDDE authority.

### V2. Dependence on the monomer-structure predictor

**Concern.** The submission uses OpenDDE but does not compare an alternative structure predictor.

**Response.** The revised manuscript changes the input description from “sequence-only” to **sequence input with predicted monomer geometry** and specifies the structure predictor/version/authority. We add a separate AF3-predicted-monomer analysis and a predicted-vs-bound geometry sensitivity analysis. These show what remains stable under a different predicted-geometry authority, but they are not presented as a bit-for-bit OpenDDE→AF3 substitution because the corresponding models were retrained and the submitted OpenDDE artifacts are not available in the current frozen GPU authority. The claim is therefore robustness/sensitivity, not predictor equivalence.

### V3. Missing component ablations

**Concern.** Joint supervision, the geometry branch, contact-gated cross-attention, and the low-rank decoder are not isolated.

**Response.** The supplement now contains a component-evidence table with strict evidence classes:

- joint task supervision: frozen 2×2 task-objective ablation;
- geometry: equal-parameter no-geometry / topology / distance / direction / full-pose controls plus predicted-vs-bound sensitivity;
- pair decoder: three-seed full pair score versus unary-only and marginal outer-product diagnostics, plus the pre-existing bilinear-vs-Hadamard single-seed ablation;
- contact-gated cross-attention: reported only where the exact source/configuration supports the intervention; later content-only attention evidence is not mislabeled as an ablation of the submitted learned contact gate;
- auxiliary epitope losses: existing count/proportion and pooling screens are reported as development evidence; no claim of independent benefit is made without the exact matched multi-seed intervention.

Where evidence comes from a successor architecture or a post-submission retrain, that is stated directly.

## Reviewer 531W

### R1. AsEP is claimed but absent

**Response.** The camera-ready abstract and contribution list remove the sentence claiming that the submitted EPIC model was evaluated on AsEP. Existing AsEP results from the later FrameContact line are retained in the repository as separate external evidence and are not back-projected onto EPIC.

### R2. “Sequence-based” wording is misleading

**Response.** We use **sequence-input, predicted-monomer-geometry framework** throughout. A modality column is added to every comparison table. Methods using native complex structures, predicted monomer structures, sequence only, and mixed sequence/structure inputs are not grouped under a single “sequence” label.

### R3. Missing ablations

**Response.** Addressed by the evidence table described under V3, with exact distinction between submitted-model, post-submission same-protocol, and successor-architecture evidence.

### R4. One benchmark, no external/OOD result, no uncertainty

**Response.** We add complex-bootstrap intervals and per-complex distributions for the official 455 analysis. We also report the later antigen-cluster-disjoint/fresh-holdout and AsEP results in a clearly separated “subsequent robustness evidence” subsection, not as evidence available to the original submission. We do not claim prospective biological OOD generalization; the temporal admission gate that rejected all 14 candidate cases remains a negative result rather than being relaxed.

## Reviewer 5FFF

### F1. AsEP overclaim

**Response.** Removed from the EPIC abstract and conclusions unless an exact EPIC AsEP run is supplied. Later AsEP results remain separately labelled.

### F2. Limited novelty

**Response.** The novelty claim is narrowed. The manuscript no longer claims that PLM features, invariant geometry attention, low-rank pair scoring, or cross-attention are individually novel. The contribution is the auditable integration of a shared residue-pair representation with coupled contact/paratope/epitope readouts under a sequence-input/predicted-geometry regime, plus controlled evidence about when that coupling helps or fails.

### F3. Favorable metric packaging for contact prediction

**Response.** AUPRC/AP and prevalence are moved before AUROC and recall; the text explicitly states that EPIC's reported 0.0761 contact AUPRC is below ImaPEp's 0.0957 in the literature-context table. Practical top-k and precision/recall operating-point analyses are added. “Best contact result” is removed.

### F4. Epitope recall trade-off and conditional-proportion loss

**Response.** The text states that high recall coexists with weaker epitope ranking and likely reflects the training/readout operating point. We add precision, F1, calibration, and precision-at-matched-recall reporting. Existing conditional-proportion/count-weight screens are reported as development evidence only. We do not attribute the recall gain causally to that term without a matched multi-seed ablation.

### F5. Baseline comparability

**Response.** The current Table 1 is split into:

1. a same-protocol internal table, containing only measurements rerun on the bound split/evaluation authority;
2. a literature-context table, explicitly marked “not directly comparable,” with modality, dataset/split, aggregation, threshold policy, and source for each number.

All “best” wording across imported and rerun results is removed.

### F6. Undefined TOPK2 expert

**Response.** The orphan sentence is deleted from the EPIC camera-ready. A separately documented post-submission top-k readout study exists, but it was not part of the submitted model and is not inserted retroactively as an unexplained expert.

### F7. Weak translation/OOD/calibration engagement

**Response.** We add reliability diagrams, ECE/Brier, practical ranking budgets, and explicit OOD boundaries. The camera-ready states that no prospective clinical/experimental validation is claimed. Later strict dual-axis retrieval and temporal-admission work is referenced only as future/subsequent work.

### F8. Wrong venue footnote

**Response.** Replaced with the AI4DD workshop camera-ready footnote. “Do not distribute” and main-track wording are removed.

### F9. Global recall vs per-complex AUROC/AUPRC

**Response.** Both pooled and equal-complex recall are now reported, with equations and names that identify the estimand. Global recall is retained only as an operating-point statistic; it is no longer presented beside macro AP/AUROC without qualification.

### F10. Training cost

**Response.** We report trainable parameter count, epochs selected, aggregate epoch wall-time for each seed, accelerator type, and inference inputs. These quantities are taken from frozen run receipts; they are not estimated from memory.

### F11. Typographical and formatting issues

**Response.** Fixed capitalization in “Interleaved Contact Learning,” corrected residue-count spacing/typography, removed the stale conference footnote, and proofread table labels and modality terminology.

## Remaining scientific boundary

This response can address every reviewer concern, but it cannot turn post-submission or successor-architecture evidence into proof about the exact submitted OpenDDE checkpoint. The camera-ready therefore distinguishes:

- **exact textual correction**;
- **exact same-authority post-submission analysis**;
- **same-protocol retrain with a changed predictor**;
- **later successor evidence**;
- **unresolved exact-submission intervention**.

That distinction is part of the response, not a weakness to hide.
