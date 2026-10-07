# Concise author response for the camera-ready record

We thank the reviewers and agree that the original evaluation over-emphasized favorable metrics and did not sufficiently isolate components. EPIC is accepted; the camera-ready preserves the accepted core while making the following corrections.

1. **Sparse contacts.** Contact positives are 0.0705% of train candidates. We now foreground mean per-complex AP/AUPRC and prevalence, and add fixed-budget precision/recall, enrichment, calibration, and complex-bootstrap intervals. AUROC and pooled recall are secondary.
2. **Input modality.** EPIC is now described as sequence-input with predicted monomer geometry, not sequence-only. Baselines are labelled by input modality.
3. **AsEP.** The unsupported exact-EPIC AsEP claim is removed from the abstract and conclusion. Later AsEP experiments are separated as successor evidence.
4. **Baseline comparability.** Same-protocol measurements and imported literature context are split into separate tables; imported values are not treated as a controlled leaderboard.
5. **Ablations.** We add frozen evidence for joint objectives, equal-parameter geometry channels, full pair scores versus marginal controls, and the epitope count/proportion trade-off. Evidence class, seed count, and limitations are stated. A later single-seed successor ICG campaign is reported with full provenance, but its intended no-gate M2 control timed out without a result and the completed variants change other objectives. It therefore cannot identify a causal submitted-EPIC gate effect; the claim is moderated accordingly.
6. **Epitope recall.** We now describe high recall as a precision/ranking trade-off. A development screen confirms that the count/proportion term can increase recall while decreasing precision, but its effect interacts with regularization and is not presented as a universal causal gain.
7. **Uncertainty/OOD/calibration.** Complex-bootstrap intervals and reliability/Brier diagnostics are added. The official 455 cohort is labelled historical/exposed. No prospective or clinical generalization claim is made.
8. **TOPK2, metric weighting, cost, and formatting.** The undefined TOPK2 sentence is removed; pooled and macro recall are named separately; parameter count, GPU and wall-time are reported; the workshop footer and typographical issues are fixed.

Post-submission AF3 retraining and later FrameContact analyses are explicitly separated from the exact submitted OpenDDE model. We do not use successor evidence to manufacture a stronger historical claim.
