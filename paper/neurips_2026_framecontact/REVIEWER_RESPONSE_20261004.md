# EPIC (Submission 182) — Author Responses

---

## 1. General Response to All Reviewers

We thank all three reviewers for careful, constructive reviews. The reviewers raised the same five concerns: (i) missing component ablations, (ii) how to read contact metrics under extreme class imbalance, (iii) the unsupported AsEP claim in the abstract, (iv) dependence on the backbone predictor, and (v) uncertainty and generalization. Since the reviews we ran preregistered ablation campaigns (75 models), a cross-predictor study with 3,015 newly predicted structures, a matched retraining of EpiFormer on the same split, and an improvement study (a further 75 models) — 150 trained models in total. Below we summarize what we did, what we found, and what we will change in the manuscript. We have aimed to report results that do not favor our method as clearly as results that do.

### 1.1 What we ran (and what it is not)

- **Two matched campaigns, 3 seeds each (17, 29, 43), on the official SAbDab2 split: 455 official test complexes, plus 2,887 training and 321 validation complexes carved from the official training partition.**
  - **Campaign A (36 models):** full model plus 11 ablations, with the distance auxiliary disabled in all arms.
  - **Campaign B (39 models):** an independent reconstruction that follows the paper's equations: a sigmoid learned-λ contact gate plus a CA-distance auxiliary (weight 0.1). It has the same 11 ablations plus a matched `no_distance_aux` arm.
- **Test hygiene.** All checkpoints were selected on validation only. Calibration maps and operating points for all 75 models were fitted on validation and frozen before any test record was loaded. Test results were never used for selection. A terminal byte-level audit of all 75 runs (455/455 test complexes and 321/321 validation complexes each) passed.
- **Uncertainty.** Each estimate comes from a paired bootstrap with 10,000 replicates, resampled over **PDB-entry blocks** (257 blocks, so that complexes sharing a PDB entry move together). Per-complex scores are averaged over the three seeds before resampling. Every component contrast uses a **simultaneous** 95% interval across the whole family (33 contrasts in A, 36 in B). We call a component effect supported only if all three seeds agree in sign *and* the simultaneous interval excludes zero. Otherwise we report it as inconclusive.
- **Important disclosure.** The submitted checkpoint and its OpenDDE geometry cache could not be restored bit-for-bit. Both campaigns therefore use a **retrained EPIC that follows the paper's equations, with geometry from AlphaFold3-predicted monomers** (each chain predicted from sequence alone; no complex structure is used). This makes the campaigns a fair test of *which components matter*, but their absolute numbers are not the Table 1 numbers. The retrained model is somewhat weaker than the submitted one (for example, contact per-complex AUROC is 0.90 vs. 0.953, and epitope AUROC is 0.61 vs. 0.657). We will report both models in the revision and label each one clearly. The retrained models use the identical 455-complex test set (19,786 positive pairs, as in Table 2). Their 2,887/321 train/validation division of the 3,208 development complexes, however, differs from the one summarized in Table 2: the totals are identical (123,970 positive pairs), but membership differs.

### 1.2 Component ablations (Δ = full − ablation, per-complex average precision)

`*` = supported (all seeds agree in sign and the simultaneous 95% CI excludes 0). Brackets show the simultaneous 95% CI.

| Component removed | Task | Campaign A | Campaign B |
|---|---|---|---|
| **Predicted-geometry branch** | epitope | **+0.031** [+0.011, +0.051]* | **+0.025** [+0.008, +0.043]* |
| | paratope | +0.006 [−0.006, +0.018] | **+0.018** [+0.005, +0.032]* |
| | contact | **+0.0045** [+0.0006, +0.0085]* | +0.0031 [−0.0002, +0.0064] |
| **Residue-level supervision** (contact-only training) | paratope | **+0.032** [+0.014, +0.049]* | **+0.040** [+0.023, +0.058]* |
| | epitope | −0.000 [−0.019, +0.018] | +0.001 [−0.020, +0.023] |
| **Contact supervision** (residue-only training) | contact | **+0.018** [+0.013, +0.023]* | **+0.017** [+0.013, +0.022]* |
| | epitope | +0.010 [−0.007, +0.026] | +0.011 [−0.008, +0.029] |
| **Contact gate** (→ content-only cross-attention) | paratope | **+0.021** [+0.008, +0.034]* | +0.007 [−0.005, +0.018] |
| | contact / epitope | inconclusive | inconclusive |
| **Cross-attention** (removed entirely) | paratope | **+0.026** [+0.011, +0.042]* | +0.016 [−0.001, +0.032] |
| **Low-rank pair decoder** (→ unary + bias) | contact | +0.001 [−0.003, +0.004] | −0.001 [−0.004, +0.003] |
| | paratope | −0.002 [−0.020, +0.015] | **−0.017** [−0.030, −0.004] (unary better) |
| **Rank 128 → rank 1** | all tasks | inconclusive | inconclusive |
| **Contact-marginal residue readout** (→ unary readout) | all tasks | inconclusive | inconclusive |
| **Epitope Dice + proportion losses** | epitope | +0.006 [−0.007, +0.019] | −0.000 [−0.019, +0.018] |
| **Distance auxiliary** | epitope | — | +0.011 [−0.006, +0.028] |

Note: the "contact-only" and "residue-only" arms also change which validation metric drives checkpoint selection. They are therefore comparisons between training regimes, not surgical removals of a single component.

**What we conclude, and what we no longer claim:**
1. **The predicted-geometry prior is the one architectural component whose benefit replicates in both campaigns.** It improves epitope AP by about +0.03 (a ~14% relative gain), and the improvement is supported in both campaigns.
2. **Joint supervision is necessary but not synergistic for epitopes.** Removing residue-level losses reliably hurts paratope prediction. Removing the contact loss destroys contact prediction (as expected). However, we find no evidence that contact supervision improves epitope prediction.
3. **We find no evidence of a benefit from the low-rank pair decoder over a unary decoder, from the contact-marginal residue readout, or from the epitope auxiliary losses.** Results for the contact gate and cross-attention are mixed: both help paratope prediction in Campaign A, and neither does in Campaign B.
4. **We will moderate the manuscript accordingly.** Claims of the form "contact gating / the low-rank decoder / cross-task consistency improves accuracy" will be removed or rewritten as design choices. The text will present predicted geometry and joint training as the components with empirical support. EPIC's remaining practical value is a single model that produces internally consistent contact, paratope, and epitope outputs from sequence. We will no longer claim that this consistency improves accuracy.

### 1.3 Manuscript changes (summary)

1. Remove "and AsEP" from the abstract. No AsEP evaluation is reported.
2. Add the ablation table above, with seeds, simultaneous CIs, parameter counts, and wall-clock time.
3. Reframe the contact results around per-complex AP, top-k precision, and enrichment over prevalence (see the reply to Reviewer 5FFF). Report AUROC as secondary.
3b. Add the improved model (EPIC v2, §1.6) as a separate subsection, with the seed-ensembling vs new-component decomposition and the selection-transfer failure stated.
3c. Add the matched EpiFormer comparison and state that the epitope column of Table 1 was misleadingly favourable because it mixed pooled and per-complex aggregation.
4. Report epitope precision at a matched recall and clarify the operating point behind the 0.6097 recall.
5. Add an "input modality" column to Table 1 and state where each baseline number comes from.
6. Add calibration results (raw vs. validation-fitted ECE/Brier).
7. Add bootstrap CIs for all headline numbers.
8. Correct the contact-label definition to heavy-atom ≤ 4.5 Å (lines 135 and 226; see §1.4).
8b. Remove the partner-conditioning motivation for contact-gated cross-attention (lines 11, 50–51, 104) and state the partner-specificity result (§1.5) as a limitation.
8c. Restate the consistency contribution as a property of joint supervision, measured (§1.4), not of the gating architecture.
9. Rewrite contribution 4 (lines 54–55). "Matches or outperforms state-of-the-art structure-informed methods across epitope, paratope, and contact prediction" is not supported for epitopes. It will say instead that EPIC is best on contact AUROC/recall, comparable on paratope, and clearly below structure-informed methods on epitope AUROC/AUPRC.
10. Replace "sequence-based" (line 47, Conclusion) with "sequence-input with predicted monomer geometry".
11. Remove the dangling reference "in the following subsection, we further modify the aggregation of contact evidence on the epitope side" (lines 116–118). Define or remove the "TOPK2 expert" (line 273).
12. Correct Appendix A.1 (lines 215–216): the 321 validation complexes are carved out of the official SAbDab2 training partition. They are not an official SAbDab2 validation split.
13. Fix the workshop footnote and the capitalization typo "contact Learning" (line 44).

### 1.4 We now measure cross-task consistency instead of asserting it

The paper's central architectural claim is that contacts, paratopes and epitopes are coupled views of one interface. It was never measured. We now measure it on the 455 test complexes. For each complex the contact map implies its own residue-level marginals through the paper's own aggregation (log-mean-exp over the partner axis, Eq. 5); we compare those marginals with what the residue heads predict.

| Condition (3 seeds) | Epitope: head–marginal ρ | AP, epitope head | AP, contact marginal alone | Paratope: head–marginal ρ | AP, paratope head | AP, contact marginal alone |
|---|---|---|---|---|---|---|
| Full model | **+0.97** | 0.214 | **0.217** | **+0.99** | 0.671 | 0.644 |
| Contact evidence removed from readouts | +0.96 | 0.215 | 0.221 | +0.97 | 0.677 | 0.641 |
| No contact supervision (residue-only) | +0.75 | 0.204 | 0.188 | **+0.17** | 0.670 | **0.206** |
| No residue supervision (contact-only) | +0.98 | 0.213 | 0.217 | +0.98 | 0.631 | 0.642 |

Three results, two of which correct the paper:

1. **The coupling is real and quantifiable.** Rank agreement between each residue head and the contact map's own marginal is ρ ≈ 0.97–0.99, and **the contact map alone predicts epitopes as well as the dedicated epitope head** (AP 0.217 vs 0.214). The pairwise representation genuinely carries the residue-level signal rather than the heads solving three loosely related problems.
2. **The contact-gated readout is not what produces the coupling.** Removing contact evidence from the readouts leaves agreement essentially unchanged (ρ 0.96 / 0.97). This matches §1.2, where that component showed no accuracy benefit either.
3. **Joint supervision is what produces it.** Without contact supervision the paratope coupling collapses (ρ +0.17) and the contact marginal becomes useless as a paratope predictor (AP 0.206 vs 0.670).

We will therefore restate the contribution precisely: *joint supervision over a shared pairwise representation* yields a model whose contact map and residue predictions are measurably the same object (ρ ≈ 0.97–0.99, with the contact map matching the epitope head), and we will stop attributing that property to the gating architecture. To our knowledge no competing method reports such a measurement, and it is what EPIC provides that a collection of separately trained predictors cannot.

### 1.5 A direct test of partner specificity, which the architecture does not pass

The paper motivates contact-gated cross-attention as conditioning predictions "on the specific antibody–antigen pair". That is testable, so we tested it. Each test antibody was paired with its true antigen and with 9 decoy antigens drawn from other test complexes, restricted to antigens sharing **no** antigen cluster with the true one, and we asked whether the true partner scores highest. Scores use the contact map only; labels are never read.

Because a decoy antigen differs in length from the true one, a scoring rule can rank by antigen size rather than by compatibility. We therefore report six rules together with each rule's rank correlation between score and antigen length:

| Scoring rule from the contact map | True partner ranked first | ρ(score, antigen length) |
|---|---|---|
| Mean of top-50 logits | 0.123 | +0.62 (size-driven) |
| Mean of top-10 logits | 0.130 | +0.59 (size-driven) |
| Maximum logit | 0.110 | +0.56 (size-driven) |
| **Log-mean-exp over all pairs** | **0.110** | −0.12 |
| **Mean of top 1% of pairs** | **0.103** | +0.10 |
| **Mean over all pairs** | **0.103** | −0.04 |
| *Chance* | *0.100* | — |

Under the three rules that are not confounded by antigen size, the true partner is ranked first **0.103–0.110 of the time against a chance rate of 0.100**. The apparent 0.12–0.13 of the size-correlated rules is substantially an artefact of antigen length. Removing cross-attention entirely does not reduce this (0.130 top-50-mean, 0.299 MRR vs 0.311), so the module is not providing the partner conditioning it was introduced for.

**What this means.** EPIC's contact map is a strong *interface-propensity* predictor (it enriches true contacts 20–50× over prevalence, §1.6) but it carries essentially no information about *which* antigen a given antibody binds. We will therefore remove the partner-conditioning motivation from the method section and state this limitation explicitly. It is consistent with the rest of our evidence: contact gating and cross-attention show no reliable accuracy benefit (§1.2), and the cross-task coupling comes from joint supervision rather than from the gate (§1.4). Partner-specific prediction remains an open problem that EPIC, as published, does not solve.

### 1.6 An improved model, and an honest account of how we selected it

The ablations above say which components carry signal. We then asked whether that knowledge yields a better model, and found that it does. Three changes were tested, each motivated by a diagnosis rather than a guess:

1. **Our validation split was leaking.** The 321 validation complexes are carved from the official *training* partition, so they share sequence clusters with training data (Section 3.2, lines 133–134). On a **cluster-disjoint** validation split we built for this purpose (324 complexes, 82 whole clusters, zero cluster overlap with training), epitope AP falls from 0.79 to 0.26 — and the best epoch moves from late to **epoch 1–2**. The model overfits to training antigens far sooner than the original validation set could reveal. Every selection below uses the cluster-disjoint split.
2. **Regularization and weight averaging**, which target that overfitting: dropout 0.3, weight decay 0.05, learning rate 1e-4, and an exponential moving average of the weights evaluated in place of the raw weights.
3. **Stronger inputs**: SaProt-650M residue embeddings computed from the same predicted monomer backbones (no experimental structure, no new information about the complex), plus six solvent-exposure features derived from those backbones, plus a hard-negative contact ranking loss.

We trained 75 further models (two rounds, 6 and 12 arms, 3 seeds each), compared every single checkpoint and every seed/arm ensemble on the cluster-disjoint validation split alone, then opened the test set once. Against the campaign's single-model reference:

| Test, 455 complexes | Original single model | Improved ensemble | Paired difference (95% CI) |
|---|---|---|---|
| Contact per-complex AP | 0.0214 | **0.0346** | +0.0132 [+0.0086, +0.0181] (**+62%**) |
| Epitope per-complex AP | 0.2148 | **0.2502** | +0.0354 [+0.0167, +0.0555] (**+16%**) |
| Paratope per-complex AP | 0.6841 | **0.7215** | +0.0374 [+0.0257, +0.0496] (**+5.5%**) |

The contact gain is the one that matters practically (Reviewer 5FFF, Weakness 3):

| Contact, per complex | Original | Improved |
|---|---|---|
| Precision in top 10 pairs | 0.027 (23× prevalence) | **0.054 (53×)** |
| True contacts found in top 10 | 0.27 | **0.54** |
| Precision in top 50 | 0.027 | **0.041** |
| Precision in top L_min | 0.023 | **0.033** |

Epitope precision also improves at every matched recall (0.174 → 0.237 at recall 0.1; 0.122 → 0.134 at the paper's 0.6097).

**Two things we must report against ourselves.**

- **Roughly half of the contact and epitope gain is plain seed ensembling, not the new components.** Ensembling three seeds of the *unchanged* model already gives contact +27% and epitope +8% over a single model (paratope unchanged). The new components add a further +27% / +8% / +6%. We will report the decomposition, not the total alone.
- **Our per-task validation selection partly failed to transfer.** We selected a separate ensemble per task on validation. The paratope choice transferred; the contact choice did not — the ensemble that won contact on validation (AP 0.0446) scored 0.0240 on test, *below* the ensemble baseline (0.0273). Its validation margin over the runner-up was 0.0001, i.e. noise: with 324 validation complexes and 72 candidates, selection overfits. The ensemble selected for paratope turned out to be best on all three tasks on test, and the table above reports it — but we state plainly that its contact and epitope superiority was **observed on test, not selected there**, and we will present it that way rather than as a successful selection.

In the revision this becomes a separate subsection (EPIC v2) with the original model retained for comparability, and the limitation that our selection protocol needs a larger cluster-disjoint validation set stated explicitly.

### 1.7 Two matched baselines we had not run, and what they show

**Paraplume, the sequence-only paratope specialist.** Paraplume's released model is an MLP over ESM-2 650M residue embeddings — the same input representation EPIC uses — so we ran it unchanged on our test antibodies. Its weights were trained on SAbDab, so we listed its training PDBs: 69 of our 455 test complexes appear there, leaving 386 clean ones as the primary comparison.

| Paratope, per-complex AP | All 455 | Clean 386 (no Paraplume training overlap) |
|---|---|---|
| EPIC as published | 0.671 | 0.658 |
| Paraplume (released) | 0.710 | 0.691 |
| EPIC improved (§1.6) | **0.722** | **0.709** |
| EPIC published − Paraplume | −0.039 [−0.063, −0.014] | **−0.034 [−0.062, −0.004]** |
| EPIC improved − Paraplume | +0.012 [−0.010, +0.034] | +0.018 [−0.006, +0.044] |

**As published, EPIC is significantly worse than a dedicated sequence-only paratope predictor** (the interval excludes zero). Our improved model reaches parity and is nominally ahead, but that difference is not significant. Table 1 already showed Paraplume with the higher paratope AUPRC; this confirms it under one protocol and removes any ambiguity. We will therefore not claim paratope leadership. What survives is that EPIC attains specialist-level paratope accuracy *while also* producing contacts and epitopes from the same representation, and that it is far ahead of the structure-informed EpiFormer on this task (+0.284 AP, reply to 5FFF Q4).

**AsEP as an external cohort, and why the abstract claim could not have been a clean external test.** We scored the frozen SAbDab2 checkpoints on all 1,553 AsEP complexes. First, the overlap: AsEP and SAbDab2 both derive from SAbDab, and the **median sequence identity of an AsEP complex to our training set is 0.97**; only 412 (27%) fall below 0.5.

| AsEP stratum by identity to SAbDab2 training | n | Contact AP (prev.) | Epitope AP (prev.) | Paratope AP (prev.) |
|---|---|---|---|---|
| < 0.3 | 34 | 0.009 (0.004) | 0.164 (0.096) | 0.412 (0.294) |
| **< 0.5 (genuinely external)** | **412** | **0.012 (0.004)** | **0.186 (0.099)** | **0.450 (0.322)** |
| 0.5 – 0.9 | 187 | 0.011 (0.003) | 0.187 (0.082) | 0.414 (0.301) |
| ≥ 0.9 (near-duplicates of training) | 954 | 0.021 (0.004) | **0.316 (0.108)** | 0.443 (0.320) |

Two conclusions:

1. **Apparent AsEP performance is driven substantially by training overlap.** Epitope AP is 0.316 on near-duplicates versus 0.186 on the genuinely external subset — a 70% relative difference under an identical evaluation. Any AsEP number reported without this filter mostly measures memorisation. This is a stronger reason to drop the abstract's AsEP claim than our original one ("it was not run"), and we will give this reason instead.
2. **On genuinely external data EPIC still works, but much less well.** Epitope AP is 1.9× prevalence externally versus 2.6× on SAbDab2 test; contact AP is 2.9× versus 20×. We will report this as the paper's out-of-distribution result rather than claiming generalisation.

A caveat that must travel with every number in this table: AsEP's antibody axis is CDR residues and its antigen axis is surface residues, whereas EPIC was trained on full VH+VL and full antigen axes (note the paratope prevalence of 0.32 versus 0.096 on SAbDab2). Absolute values across the two benchmarks are therefore not comparable; the identity stratification *within* AsEP uses one convention throughout and is the reliable comparison.

### 1.8 A correction we found ourselves

While auditing the training and validation labels for this response, we found that the label definition in the manuscript (Section 3.2, line 135; Appendix A.1, line 226: "Cα–Cα distance … at most 4.5 Å") **does not describe the labels we actually used**. On all 3,208 training and validation complexes, we recomputed contacts from the experimental structures:
- A **Cα–Cα ≤ 4.5 Å** rule recovers only **1.8%** of the labeled positive pairs. It never labels a pair the labels leave negative.
- A **minimum heavy-atom (non-hydrogen) distance ≤ 4.5 Å** rule reproduces the labels **exactly in 3,047 of 3,208 complexes (95.0%)**.
- With a 0.002 Å rounding tolerance (≤ 4.502 Å), the match rises to **3,169 complexes (98.8%)**.

The labels are therefore standard heavy-atom ≤ 4.5 Å contacts, and the "Cα" in the text is a writing error. We will correct both sentences. This changes no reported number (the counts in Table 2 and the 0.07–0.11% prevalence already refer to these labels), but the manuscript must describe the labels correctly.

---

## 2. Response to Reviewer VUfK

Thank you for the positive assessment and for three precise requests. We address each below and refer to the General Response for the full ablation table.

**(1) Contact imbalance (0.07% positives) makes contact performance hard to interpret.**
We agree. The revision makes **per-complex average precision, top-k precision, and enrichment over prevalence** the primary contact metrics, and AUROC secondary. For the retrained model in the ablation campaigns (455 test complexes, 19,786 positive pairs among 18.47M candidate pairs, pooled prevalence 0.107%):

| Ranking budget per complex | Precision | Recall | Enrichment over prevalence | True contacts recovered per complex |
|---|---|---|---|---|
| top 10 | 0.030 | 0.007 | ~22× | 0.30 |
| top 50 | 0.026 | 0.031 | ~20× | 1.3 |
| top L_min | 0.023 | 0.068 | ~17× | 2.9 |
| top 1% of pairs | 0.019 | 0.138 | ~14× | 5.9 |

*(Average of the two campaigns' full models over 3 seeds. Top-k precision counts complexes with no positive pairs as zeros. L_min is the length of the shorter chain.)*

We will state plainly what these numbers mean. The contact map **concentrates interface signal 14–22-fold above chance**, but most of its top-ranked pairs are still false positives. A practitioner should treat it as a soft prior for docking or restraint weighting, or as a way to narrow down candidate interface regions, and not as a list of reliable residue-residue restraints. The high AUROC (0.90 in the retrained model) mostly reflects correctly ranking the very large number of easy negative pairs, and we will say so explicitly.

**(2) No comparison across backbone predictors.**
We ran two complementary analyses.

**(a) Retraining with a different predictor.** We replaced OpenDDE with **AlphaFold3** (monomer predictions from sequence only) and retrained from scratch. The benefit of the geometry branch reproduces: it gives a supported epitope gain of +0.025 to +0.031 AP in both ablation campaigns. The value of a predicted-geometry prior is therefore not specific to OpenDDE.

**(b) Swapping the predictor at inference.** We predicted all 231 unique partner entities of the 321 validation complexes with OpenDDE v1.1.1 (no MSA, no templates, 200 diffusion steps, 10 recycles; all 231 predictions completed with exact sequence identity). We then scored the *same* frozen AF3-trained checkpoints (3 seeds) with OpenDDE geometry in place of AF3 geometry. Effects are paired differences in per-complex AP (OpenDDE − AF3), with PDB-block bootstrap 95% CIs:

| Subset of validation complexes | n | Epitope ΔAP | Contact ΔAP | Paratope ΔAP |
|---|---|---|---|---|
| All | 321 | −0.039 [−0.056, −0.023] | −0.012 [−0.016, −0.007] | −0.003 [−0.005, −0.001] |
| Predictors agree (max chain Cα RMSD < 2 Å) | 62 | +0.001 [−0.002, +0.006] | +0.001 [−0.001, +0.002] | −0.000 |
| Moderate disagreement (2–5 Å) | 138 | +0.001 [−0.002, +0.004] | −0.003 [−0.005, −0.002] | −0.003 |
| Large disagreement (> 5 Å) | 121 | **−0.104** [−0.139, −0.067] | −0.027 [−0.037, −0.018] | −0.004 |
| OpenDDE pLDDT, lowest tertile | 107 | **−0.120** [−0.158, −0.080] | −0.032 [−0.043, −0.022] | −0.006 |
| OpenDDE pLDDT, middle / top tertile | 107 / 107 | +0.003 / +0.001 (CIs include 0) | −0.001 / −0.001 | −0.003 / −0.000 |

Predictions are **stable when the two predictors produce consistent geometry, or when OpenDDE is confident**. The loss is concentrated in complexes where the predicted structures disagree strongly or OpenDDE reports low confidence. The drop grows with disagreement: Spearman ρ between epitope ΔAP and RMSD is −0.38, p < 10⁻¹¹. Paratope prediction is essentially unaffected. Two practical consequences follow. First, EPIC's epitope and contact outputs inherit the quality of the predicted backbone, which is consistent with the ablations showing that geometry carries real signal. Second, the predictor's own confidence (pLDDT) flags in advance which predictions to trust; we will recommend using it as a filter.

**(c) Matched retraining, which separates predictor quality from distribution shift.** Analysis (b) alone cannot tell whether OpenDDE geometry is less accurate or merely different from what the model trained on. We therefore predicted **all 3,015 partner entities** of the full cohort with OpenDDE (2,784 new predictions; every one audited for exact sequence identity) and retrained EPIC from scratch on OpenDDE geometry, 3 seeds, identical split, objective and hyperparameters. Each model is then evaluated on the 455 test complexes **with its own predictor's geometry**, so the comparison is symmetric and the test set never took part in selection:

| Test, 3-seed ensemble each | Contact AP | Epitope AP | Paratope AP |
|---|---|---|---|
| Trained and evaluated on AF3 | 0.0273 | 0.2321 | 0.6833 |
| Trained and evaluated on OpenDDE | 0.0252 | 0.2157 | 0.6821 |
| OpenDDE − AF3 (paired 95% CI) | −0.0020 [−0.0041, −0.0000] | **−0.0164** [−0.0294, −0.0030] | −0.0012 [−0.0079, +0.0056] |

Under matched conditions AF3 geometry is modestly better for epitope prediction (−7.1% for OpenDDE), marginally better for contacts, and equivalent for paratopes. Comparing (b) and (c): the inference-time swap costs more than matched retraining does, so part of that penalty was distribution shift and part is genuine predictor quality. The ranking of the two predictors is therefore stable, but the effect is small relative to the component effects in §1.2.

One entity (a 2,363-residue antigen) exceeded the GPUs we could schedule even with context-parallel inference; the single training complex that uses it keeps AF3 geometry, and the cache records that fallback explicitly.

Limits: (b) uses validation complexes that took part in checkpoint selection, so only its paired differences are informative; (c) is on test and unbiased. Neither isolates a single predictor property such as backbone accuracy. In the revision we will describe EPIC as using "a predicted monomer backbone (OpenDDE in Table 1; AF3 in the ablation study)", add both tables, and recommend confidence-aware use.

**(3) Isolate joint supervision, the geometry branch, contact-gated cross-attention, and the low-rank pair decoder.**
We have done this (General Response §1.2). In short: the **geometry branch** and **joint residue-plus-contact supervision** have replicated, statistically supported benefits. The **contact gate** and **cross-attention** help paratope prediction in one campaign but not the other. The **low-rank pair decoder** shows no benefit over a unary decoder, and the contact-marginal residue readout shows no benefit either. Following your suggestion, we will **both** add the ablations **and** moderate the corresponding claims.

---

## 3. Response to Reviewer 531W

Thank you for the detailed and expert review. We agree with all four weaknesses and address them in turn.

**(1) AsEP is claimed in the abstract but never evaluated.**
You are correct, and this is an error in the abstract. `[AUTHOR TO CONFIRM — one sentence on why AsEP was named in the abstract]`. We have now run it (General Response §1.7), and the result justifies removing the claim rather than satisfying it: **the median sequence identity of an AsEP complex to our SAbDab2 training set is 0.97**, and EPIC's epitope AP is 0.316 on the near-duplicate stratum versus 0.186 on the 412 complexes below 0.5 identity. An unfiltered AsEP number would therefore have measured memorisation, not transfer. We will remove AsEP from the abstract and report the identity-stratified result in the body as our out-of-distribution analysis.

**(2) "Sequence-based" is inaccurate when predicted backbones are used.**
We agree. The revision will describe EPIC as a **"sequence-input"** method that uses **"sequence-derived (predicted-monomer) geometry"**. It will add an **input-modality column** to Table 1 with these categories: sequence-only; sequence + predicted monomer structure (EPIC); experimental antigen structure; experimental antibody structure; experimental complex.

On a matched sequence-plus-predicted-structure baseline: the closest controlled comparison we can offer is **EPIC with the geometry branch removed**, retrained under the same protocol. This isolates exactly what the predicted structure adds. The gain is +0.025 to +0.031 epitope AP, supported in both campaigns, with a smaller and less consistent effect on contacts and paratopes. We have not re-run an external baseline (e.g., MIPE or EpiFormer) with predicted structures in place of experimental ones, and we will list this as a limitation.

**(3) No ablations.**
See General Response §1.2: 12 conditions (Campaign A) and 13 conditions (Campaign B), each with 3 seeds. Every condition starts from the same initialization protocol, is trained from scratch, and is selected on validation only. Contrasts use paired, PDB-block bootstrap CIs that are simultaneous across the whole family. The resulting conclusions are less flattering to the architecture than the original text. Only predicted geometry and joint supervision have robust support, and we will rewrite the claims to match.

One limitation of this design: our `no_geometry_branch` arm removes the predicted-geometry input *and* the geometry-aware attention together, because the attention exists only to consume those features. The ablations therefore measure their joint contribution and do not separate "predicted geometry" from "geometry attention". A variant that injects the same geometric features through a simpler mechanism (e.g., an additive pair bias) would be needed to separate them; we will state this explicitly.

**(4) One benchmark, no CIs, no out-of-distribution analysis.**
- **Confidence intervals.** We now report 95% bootstrap CIs (PDB-block resampling, 10,000 replicates) and the spread across seeds for every headline metric. For the retrained full model (Campaign B), per-complex AP is: contact 0.0219 [0.0191, 0.0248], epitope 0.214 [0.197, 0.233], paratope 0.671 [0.649, 0.693]. The seed SD is 0.001, 0.007, and 0.011 respectively. Seed SD of per-complex AUROC is ≤ 0.008 for all tasks. We will add matching intervals for the Table 1 model.
- **The test set is already an antigen-held-out split, and we now quantify how strictly.** The official SAbDab2 ab-ag split (Appendix A.1) puts antigen clusters (40% identity) and antibody clusters (CDRH3, CDRH123, CDRL123) into disjoint partitions. We verified this directly:
  - **0 of 455** test complexes share an antigen cluster, an antibody cluster, or a CDRH3 cluster with the training set.
  - Every test antigen has **< 50%** local-alignment identity to every training antigen (identity normalized by the test antigen's length): **median 20%**, 90th percentile 26%.

  So the headline numbers already measure performance on unseen antigen families. Within the test set, per-complex epitope and contact AP look lower for the least similar antigens. However, that apparent trend is entirely explained by antigen size: less similar antigens are longer (ρ = −0.28), and longer antigens have lower epitope prevalence. After normalizing AP by prevalence, performance is **independent** of residual similarity:
  - epitope: Spearman ρ = 0.04, p = 0.38;
  - contact: ρ = 0.005, p = 0.91;
  - a regression controlling for prevalence and antigen length gives a coefficient whose PDB-block bootstrap CI includes 0.

  What we cannot assess is performance on antigens closely related to the training set: the official split contains none. We have no external test set.
- **Thresholds need recalibration.** Our validation complexes are a subset of the official SAbDab2 *training* partition (Section 3.2, lines 133–134). They are therefore not cluster-disjoint from the training data, unlike the test set. As a result, validation scores are optimistic: per-complex epitope AP is 0.79 on validation vs. 0.21 on test for the retrained model. Operating points chosen on validation also transfer poorly; for example, a contact threshold chosen for 50% recall on validation gives ~7–9% recall on test. We will state this explicitly and correct Appendix A.1, which currently implies that SAbDab2 itself supplies the 321-complex validation set. For deployment on new antigens, thresholds should be recalibrated on cluster-disjoint data.
- **Unseen antibody families and flexible interfaces.** These also remain untested. Because EPIC uses rigid, independently predicted monomer backbones, interfaces that involve substantial conformational change on binding are a structural blind spot of the method, not only an evaluation gap. The revision will state both limitations explicitly.

---

## 4. Response to Reviewer 5FFF

Thank you for a thorough and fair review. We answer Q3 and Q4 here, with the contact-metric and matched-baseline evidence they call for; ablation details are in the General Response.

**Q3 — Precision at practical operating points; how to interpret AUPRC 0.0761 at ~0.1% prevalence.**
This is the right question. The table below gives practitioner-facing numbers for the retrained model from the ablation campaigns (3-seed average of both full models; pooled prevalence 0.107%):

| Ranking budget per complex | Precision | True contacts per complex | Enrichment |
|---|---|---|---|
| top 10 | 0.030 | 0.30 | ~22× |
| top 50 | 0.026 | 1.3 | ~20× |
| top L_min | 0.023 | 2.9 | ~17× |
| top 1% | 0.019 | 5.9 | ~14× |

At a fixed recall, pooled precision is 0.016 at 10% recall and 0.008 at 50% recall. For this model, no threshold achieves 5% pooled precision at non-trivial recall. Put plainly: even an AUPRC several times the prevalence means that **most top-ranked pairs are not contacts**. The contact map is useful as a ranking prior that enriches for interface pairs 14–22-fold, not as a source of high-confidence restraints. We will (i) make per-complex AP, top-k precision, and enrichment the primary contact metrics, (ii) present AUROC as secondary and explain why it is uninformative here, and (iii) restate the contact claim in the abstract as "best AUROC and recall, but not best AUPRC". The retrained model's per-complex AP (0.022) is lower than the submitted model's (0.0761). `[AUTHOR TO CONFIRM: whether Table 1 AUPRC used the same per-complex average-precision estimator; if not, state the difference.]`

**Q4 — Does the conditional-proportion loss explain the high epitope recall? What is precision at matched recall?**
In our controlled ablations, **no**. Removing the Dice and conditional-proportion losses, or keeping only one of them, does not raise recall at threshold 0.5. In Campaign B, recall@0.5 is 0.19 for the full model, 0.25 without either auxiliary loss, 0.26 with Dice only, and 0.30 with the proportion loss only. Precision at a matched recall is also essentially unchanged. At the paper's recall level (0.61), pooled epitope precision is 0.11–0.12 in every variant, against a prevalence of 0.082. No auxiliary-loss contrast is statistically supported. Your underlying point is correct: the 0.6097 recall in Table 1 reflects where the operating point was set, not better discrimination. At matched recall our epitope precision is only ~1.4× chance, consistent with EPIC's weaker epitope ranking.

**Matched-protocol comparison with EpiFormer.** To answer the second half of your question we retrained EpiFormer ourselves, using its published geometric-only configuration, on the official SAbDab2 training split (3 seeds, selected on our validation split), and scored it on the same 455 test complexes with the same inherited labels and EPIC's per-complex protocol:

| Metric (455 test complexes) | EPIC | EpiFormer | EPIC − EpiFormer (paired 95% CI) |
|---|---|---|---|
| Epitope per-complex AP | 0.214 | **0.372** | **−0.158** [−0.178, −0.139] |
| Epitope per-complex AUROC | 0.614 | **0.776** | −0.162 [−0.181, −0.143] |
| Epitope precision at recall 0.1 | 0.183 | **0.497** | — |
| Epitope precision at recall 0.6097 | 0.117 | **0.282** | — |
| Paratope per-complex AP | **0.671** | 0.387 | **+0.284** [+0.263, +0.304] |

So at the operating point the paper highlights, EpiFormer's epitope precision is **2.4× ours**. Restricting both models to the residues EpiFormer scores (antigen surface, antibody CDR), which removes any coverage advantage, leaves the gap essentially unchanged (−0.153 AP). Our improved model (General Response §1.6) narrows but does not close it: epitope AP 0.250 vs 0.372.

One asymmetry must be stated: this EpiFormer uses **experimental backbones taken from the bound complex structures**, while EPIC uses independently predicted monomers. The comparison therefore measures a structure-informed model against a sequence-input one, and it reinforces the paper's own observation that epitope prediction depends heavily on structural input. It does not show that EPIC is a poor sequence-input model; it shows that the epitope column of Table 1 gave a misleadingly favourable impression, which came from mixing pooled and per-complex aggregation.

We did not re-run MIPE. The revision will (a) remove "highest epitope recall" as a headline result, (b) report precision at matched recall and per-complex AP, (c) state plainly that structure-informed methods are clearly stronger at epitope discrimination, and (d) present paratope prediction and the sequence-only inference regime as EPIC's actual strengths.
