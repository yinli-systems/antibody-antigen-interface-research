# Reviewer 5FFF — sections held back from the posted reply

> Removed from `EPIC_rebuttal.md` §4 on 2026-10-04 at the author's request. Q3 and Q4 remain posted. Paste any of these back verbatim when they are ready.

---

**Q1 — Where are the AsEP results?**
Including AsEP in the abstract was an error, and no AsEP result was in the submission. `[AUTHOR TO CONFIRM: one sentence on why it was named]` We have since run it properly (General Response §1.7). The reason it cannot simply be added is quantitative: AsEP's median sequence identity to our SAbDab2 training set is 0.97, and epitope AP falls from 0.316 on near-duplicates to 0.186 on the 412 genuinely external complexes. We will remove the abstract claim and report the identity-stratified analysis in the body instead.

**Q2 — Provenance of the baseline numbers in Table 1.**
Thank you for pressing on this. Re-checking our evaluation records shows that **Table 1 does not use a single protocol**, and we will correct it.
- **EpiFormer and SPPIDER-seq:** we ran both on the official 455-complex test set. However, the values in Table 1 are **pooled (micro) metrics over all residues**, whereas EPIC's are per-complex averages (Appendix A.3).
  - EpiFormer, same run: pooled AUROC/AUPRC is 0.8008/0.2775 (as in Table 1); per-complex it is **0.768/0.348**.
  - SPPIDER-seq: pooled epitope 0.6718/0.1454 and pooled paratope 0.9486/0.6511 (as in Table 1).
  - The EpiFormer run used a released-style checkpoint applied to experimental structures, not a model retrained on the SAbDab2 training split.
- **MIPE, ImaPEp, DiscoTope-3.0, ParaDeep, Paraplume:** `[AUTHOR TO CONFIRM: source of each row and its aggregation/threshold]`.

The revision will add a provenance and aggregation column to Table 1, report every baseline we ran in both per-complex and pooled form, and avoid ranking numbers across protocols. To give a protocol-matched comparison, we additionally retrained EpiFormer on the official SAbDab2 training split, selected it on our validation set, and evaluated it with EPIC's per-complex protocol and at matched recall (see Q4).
---

**Q5 — What is the "TOPK2 expert" in Appendix A.2?**
`[AUTHOR TO CONFIRM — accurate explanation, e.g. "a leftover name from an internal ensemble variant that was not used in any reported result; the reported model is a single network selected on validation."]` We will remove the term or define it.

**Q6 — Sensitivity to predicted-geometry quality, and the missing ablations.**
For ablations, see General Response §1.2. Removing the geometry branch (with retraining) lowers epitope AP by 0.025–0.031, a supported effect in both campaigns. The effect on contacts is +0.003 to +0.005 (supported in one campaign) and on paratopes +0.006 to +0.018 (supported in one campaign). Contact gating and cross-attention have mixed support. The low-rank decoder, the contact-marginal readout, and the epitope auxiliary losses have no measurable benefit, and we will moderate those claims. On sensitivity to geometry quality, we re-scored the frozen checkpoints with OpenDDE instead of AF3 geometry for all 321 validation complexes (full table in the reply to Reviewer VUfK, point 2):
- Where the two predictors agree, or where OpenDDE pLDDT is in the middle or top tertile, epitope AP is unchanged (ΔAP between −0.001 and +0.003; all CIs include 0).
- Where they disagree by more than 5 Å Cα RMSD (38% of complexes), or OpenDDE confidence is in the lowest tertile, epitope AP drops by 0.10–0.12.
- Paratope AP changes by less than 0.01 in every stratum.

So EPIC is sensitive to poor predicted geometry, and the predictor's own confidence identifies those cases.

**Weakness 2 — Novelty.**
We accept this assessment, and the ablations sharpen it. The individual components are established, and several of them (gate, low-rank decoder, contact-marginal readout) do not measurably drive accuracy in our controlled experiments. You asked what EPIC enables that was previously impossible; our best answer is now measured rather than asserted (General Response §1.4): under joint supervision the contact map and the residue-level predictions become the same object, with rank agreement ρ ≈ 0.97–0.99 and the contact map predicting epitopes as well as the dedicated epitope head. A pipeline of separately trained predictors has no such guarantee, and removing contact supervision destroys the property (paratope ρ falls to +0.17). We will present that as the contribution, together with the sequence-only inference regime. We will present the contribution as an empirical result in a practically relevant input regime. A single sequence-input model, using only predicted monomer geometry, produces mutually consistent contact, paratope, and epitope outputs. It reaches a top-tier paratope AP (0.67) without experimental structures, and the predicted-geometry prior gives a replicated epitope gain. We will drop language that suggests a new capability.

**Weakness 7 — Calibration and translation.**
We now report calibration. Raw outputs are poorly calibrated, with pooled 20-bin ECE of about 0.10–0.13 for contacts, 0.12–0.14 for epitopes, and 0.12–0.13 for paratopes. This is largely a side effect of the class-balanced and focal losses. A Platt map fitted on validation only reduces ECE to 0.001 (contact), 0.06 (epitope), and 0.01 (paratope) on test. Operating thresholds frozen on validation transfer poorly to test (see the reply to Reviewer 531W, point 4), so we will advise users to recalibrate on in-domain data. We agree the paper is a benchmark study and will describe it as such. We have no prospective or case-study validation.

**Minor comments.**
- *Global recall vs. per-complex AUROC/AUPRC:* the revision reports per-complex (macro) and pooled versions of every metric, including recall and precision at fixed thresholds, so the two aggregations can be compared directly.
- *Training cost:* EPIC has 2.40M trainable parameters (ESM-2 650M is frozen and its embeddings are precomputed). Training takes ~4.5–5.2 GPU-hours on a single RTX 4090 at batch size 1, for at most 20 epochs with early stopping. Inference on all 455 test complexes takes ~50 s, given cached embeddings and predicted backbones. The upstream costs of ESM-2 embedding and monomer structure prediction will be reported separately. We did not time the baselines under the same hardware, so we will not claim a speed advantage over them. `[AUTHOR TO CONFIRM: if you have timing for the Table 1 OpenDDE model, add it.]`
- *Footnote and typo:* the NeurIPS main-conference footnote will be replaced with the workshop footnote, and "Learning" will be lower-cased.

**Weakness 3/4 — Favorable-metric packaging.** We agree. The revised abstract and results section will report contacts (AUROC is best, AUPRC is not) and epitopes (recall is high, but AUROC/AUPRC are clearly below the structure-informed methods) without selecting the favorable metrics.
