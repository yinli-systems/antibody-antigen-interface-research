# Cohorts, estimators and comparisons

[Atlas](README.md) · [Audit findings](REVIEW_FINDINGS.md)

# # A sample count is not a dataset identity

| Authority | Role and valid interpretation | Invalid inference |
|---|---|---|
| Development 217 /68 components | Exposed SAbDab2 development; matched grid/arm comparisons | Fresh test; repeated corruption rows as independent samples |
| Registered holdout 217 | Separate registered read, disclosed 32-case overlap with another exposed baseline test | Same cohort as development 217; replacing the answer by 185-case sensitivity |
| Aligned historical test 549 | Checkpoint-specific exposed historical test | Fresh blind or proved sequence-double-cold generalization |
| Old FrameContact cache 455 |272 train /73 validation /110 held out for that checkpoint; provenance correction | A valid headline test score, or proof all 455-case studies are invalid |
| Official 455 in EPIC reconstruction | Distinct authority, historically exposed test | Original submitted OpenDDE checkpoint or pristine benchmark |
| EPIC 321 validation | Carved from official training | An official validation release or demonstrated cluster-disjoint split |
| Newer EPIC 324 validation | Reported cluster-disjoint 324 /82 clusters; evidence pending | Automatic test transfer of task-specific selection |
| AsEP group/ratio | Distinct task axes and test histories; same-split per-complex MCC | Pooled scores compared to macro published tables |
| Newer AsEP 1553 | Reported overlap-stratified frozen-model diagnostic | Strict OOD from one incompletely specified identity threshold |
| Strict dual-axis retrieval | Prospective authority-train protocol; builder AND independent audit required | Results from a canary, or unknown pairings treated as verified negatives |
| AbCite PMC/ViDoRe/VQA/audio-video | Distinct query construction, galleries and answerers | A unified model leaderboard or antibody accuracy |

# # Keep each estimator attached

Macro AP averages per-complex AP. Pooled AP concatenates candidate pairs. Equal-component AP averages within components and then across components. They weight cases differently and can order models differently.

A mean or median of per-seed AP is not AP of averaged ensemble predictions. Compare a new ensemble with an unchanged-model seed ensemble before assigning gains to new components.

Ratio of means and mean of per-seed ratios differ; keep absolute paired differences primary. Resampling seeds, complexes, PDB blocks and components addresses different uncertainties. A crossed seed×component bootstrap is not a PDB-block bootstrap after averaging seeds. Pointwise intervals are not simultaneous family intervals.

AP is not accuracy or P@k: AP 0.024 does not imply 97.6% false predictions in any specified top-k set. Show measured precision/recall at candidate budgets. AP/prevalence ratios are not a matched random-baseline experiment.

A confidence interval crossing zero does not establish equivalence, clean-safety or uselessness. Noninferiority/equivalence requires a justified prespecified margin and corresponding analysis.

A comparison key should include cohort hash, split role, label rule, candidate axes, input access, checkpoint selection, seed rule, aggregation and uncertainty unit. Missing keys allow contextual discussion, not an automatically fair ranked table. Exact sources are bound in the result cards.
