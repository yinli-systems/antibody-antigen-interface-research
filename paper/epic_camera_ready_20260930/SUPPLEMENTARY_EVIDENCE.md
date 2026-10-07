# EPIC post-submission supplementary evidence

## Evidence classes

This supplement deliberately separates five evidence classes:

1. **submitted EPIC text/results** — the accepted anonymous manuscript;
2. **camera-ready corrections** — factual and interpretive changes that require no new experiment;
3. **official-455 post-submission retrain** — same official 2,887/321/455 split, but AF3 monomer geometry rather than the submitted OpenDDE authority;
4. **development-only component studies** — frozen experiments on development data;
5. **successor FrameContact evidence** — later models that test related hypotheses but are not silently relabelled as the exact submitted model.

The accepted paper reports contact/epitope/paratope AUROC–AUPRC–recall of `0.9531/0.0761/0.3832`, `0.6572/0.2327/0.6097`, and `0.9540/0.6962/0.7464`, respectively. Those historical values remain unchanged. New analyses below are not substituted into that table when the checkpoint or geometry authority differs.

## S1. Sparse contact ranking and practical operating points

### Why AP is primary

The official SAbDab2 contact split has only 0.0705% positives in train, 0.0871% in validation, and 0.1071% in test. AUROC can therefore look strong while precision remains low. The camera-ready foregrounds mean per-complex AP/AUPRC, prevalence, and fixed-budget ranking metrics.

### Post-submission official-455 audit

We reanalysed three completed raw prediction sets from an audited from-scratch retrain on the official 455 historical test split. This retrain uses AF3-predicted monomer geometry and is **not** the exact OpenDDE submission model. No new inference was performed for this analysis, and test labels did not participate in training or checkpoint selection.

Three-seed probability ensemble:

| Contact quantity | Value |
|---|---:|
| Test candidate pairs | 18,473,412 |
| Positive contacts | 19,786 |
| Pooled prevalence | 0.0010711 |
| Macro per-complex AP | 0.02479 |
| Pooled AP | 0.01084 |
| Macro AUROC | 0.90970 |
| Pooled AUROC | 0.91195 |
| P@10 / R@10 | 0.03934 / 0.00923 |
| P@20 / R@20 | 0.03538 / 0.01661 |
| P@50 / R@50 | 0.02809 / 0.03299 |
| P@100 / R@100 | 0.02415 / 0.05723 |
| P@Lmin / R@Lmin | 0.02379 / 0.06972 |
| P@true-contact-count | 0.02941 |
| Top-1%-candidate precision / recall | 0.02041 / 0.14797 |
| Brier / ECE-15 | 0.05506 / 0.12170 |

The average P@Lmin enrichment over each complex's prevalence is 17.2×; P at the true-contact count is 22.5× prevalence. Absolute precision remains low, which is the practical interpretation the original table lacked.

Pooled precision at fixed recall is:

| Recall target | Achieved recall | Precision |
|---:|---:|---:|
| 0.01 | 0.0109 | 0.0302 |
| 0.05 | 0.0500 | 0.0223 |
| 0.10 | 0.1002 | 0.0191 |
| 0.20 | 0.2000 | 0.0147 |
| 0.30 | 0.3001 | 0.0125 |
| 0.50 | 0.5002 | 0.00947 |

At 5% precision, recall is approximately 0.0051; no threshold achieves 10% pooled precision. These results prevent favorable interpretation of AUROC/recall in isolation.

### Uncertainty and aggregation

Averaging per-complex AP across the three fixed seeds and bootstrapping the 455 complexes gives:

| Task | Macro AP | 95% complex-bootstrap CI |
|---|---:|---:|
| Contact | 0.01943 | [0.01763, 0.02129] |
| Epitope | 0.20639 | [0.19452, 0.21873] |
| Paratope | 0.65464 | [0.64049, 0.66826] |

The same artifact reports both pooled and equal-complex recall. They are different estimands and are never presented interchangeably.

Canonical artifact: `evidence/rank_calibration_v2/ANALYSIS.json` (SHA-256 `85fbce6e1c245091cafc0c02833637f1e3904551a0a2d6d4fc3a04e7fcae43eb`).

## S2. Geometry and predictor sensitivity

### Equal-parameter geometry controls

A locked development study held parameter count and graph topology fixed while changing only the invariant edge-feature channels: no geometry/topology, distance, distance+direction, and full pose. On 36 validation complexes grouped into 16 endpoint-connected components, distance+direction minus topology-only yielded:

- joint macro AUPRC: `+0.01422`, bootstrap CI `[+0.00194,+0.02295]`;
- paratope AUPRC: `+0.02484`, CI `[+0.01832,+0.03104]`;
- epitope AUPRC: `+0.00359`, CI `[-0.01597,+0.01719]`.

The multiplicity-adjusted confirmatory gate did not pass and the study is single-seed; the camera-ready therefore describes this as development evidence, not proof of universal geometry benefit.

Source: `results/sabdab2_structure_self_attention/paired_statistics_summary.json`.

### Alternative predicted geometry / bound-geometry sensitivity

A later three-seed development analysis using independently predicted monomer geometry reports median contact/epitope/paratope macro AP of `0.02673/0.24378/0.68847`. Relative to the bound-geometry routed reference, median deltas are `−0.00155/+0.00666/−0.02327`.

This supports sensitivity to geometry authority, particularly for paratope prediction, but it is not a checkpoint-matched OpenDDE-versus-AF3 comparison. The camera-ready states that limitation explicitly rather than claiming predictor invariance.

Source: `results/sabdab2_predicted_geometry_v1/PREDICTED_GEOMETRY_DEVELOPMENT_ANALYSIS_PUBLIC_V1.json`.

## S3. Joint supervision and objective interference

The frozen objective-component study shows:

- adding contact supervision to residue-only training increases pair AUPRC by approximately `+0.0833`;
- removing the complex-ranking objective increases pair AUPRC by `+0.0214` and node MCC by `+0.0369` across ten paired seeds;
- ranking-only supervision reduces node MCC by `−0.0748` versus node-only training.

Thus joint supervision is not uniformly beneficial: explicit contact labels are useful, while the global ranking objective can interfere with local interface tasks. The novelty claim is therefore an integrated formulation with measured objective trade-offs, not “all auxiliary tasks always help.”

Source: `results/objective_component_localisation_v1/PART_TWO.md`.

## S4. Pair decoder / beyond-marginal evidence

A new aggregate of existing three-seed, 549-complex diagnostics compares the full contact score with two restricted scores without retraining or checkpoint selection.

| Contrast | Equal-antigen-cluster/equal-seed Δ AP | 95% crossed seed×cluster CI |
|---|---:|---:|
| Full pair score − unary-only | +0.00129 | [−0.00070,+0.00311] |
| Full pair score − residue-head outer product | **+0.00426** | **[+0.00229,+0.00629]** |

The full pair score reliably exceeds the separable marginal outer product, supporting information beyond independent epitope/paratope propensity. Its advantage over the unary terms already embedded in the pair head is smaller and unresolved. This is successor FrameContact evidence, not an exact retraining ablation of the anonymous EPIC checkpoint.

A separate single-seed decoder screen found bilinear AP `0.02397`, Hadamard `0.02268`, and an outer-product control `0.01806`; it is retained only as underpowered development evidence.

Canonical artifact: `evidence/pair_decoder_v1/ANALYSIS.json` (SHA-256 `171985f3a433c31dc31d76dd920426c7e93f1c79d0c2dc040ad0815e3a5baae9`).

## S5. Contact-gated cross-attention

The connected authorities do not contain the exact submitted OpenDDE checkpoint/source needed for a tensor-identical learned-gate versus no-gate EPIC factorial. A later frozen FrameContact ICG campaign provides bounded successor evidence at seed 17:

| Successor variant | Contact macro AP | Contact pooled AP | Epitope macro AP | Paratope macro AP |
|---|---:|---:|---:|---:|
| M3 learned full gate | 0.025956 | 0.012446 | 0.226989 | 0.726048 |
| M4 learned interaction gate | 0.026632 | 0.012076 | 0.208928 | 0.737060 |
| M6 gate + consistency/distogram objectives | 0.020295 | 0.009174 | 0.195934 | 0.709847 |

The intended ordinary/no-gate M2 control (`1351883_5`) timed out after 3 days and produced no final result or receipt. The three completed variants also change more than the gate alone and use one training seed. Consequently, they demonstrate that contact-conditioned communication is trainable and endpoint-dependent in a successor architecture, but **do not estimate the causal effect of the submitted EPIC gate**. The camera-ready therefore retains the mechanism description and removes any claim that the gate independently drives the gain.

Canonical artifacts: `evidence/icg_successor_v1/ICG_SUCCESSOR_AUDIT.json`, `ICG_SUCCESSOR_READOUT.md`, and the preserved M2 timeout digest/tail.

## S6. Auxiliary epitope losses and the recall trade-off

A frozen single-seed development factorial varies dropout and the epitope conditional-count/proportion weight. At dropout 0.2, enabling count weight 0.2 versus 0.0 changes:

| Metric | Count 0.0 | Count 0.2 | Difference |
|---|---:|---:|---:|
| Epitope macro AP | 0.26308 | 0.26596 | +0.00288 |
| Epitope precision | 0.17518 | 0.16018 | −0.01500 |
| Epitope recall | 0.37066 | 0.40885 | +0.03819 |
| Paratope macro AP | 0.56643 | 0.59733 | +0.03090 |
| Contact macro AP | 0.02940 | 0.02972 | +0.00032 |

The term shifts the operating point toward higher epitope recall and lower precision in this screen, matching the reviewer’s hypothesis. However, at dropout 0.1, removing the term improves epitope macro AP by `+0.00551`. The direction therefore depends on regularization, and the evidence is single-seed development-only. The camera-ready may report the observed sensitivity–precision trade-off, but it must not claim a stable independent benefit or use this screen as a multi-seed EPIC ablation.

Canonical artifact: `evidence/auxiliary_loss_screen/JOINT_REGULARIZATION_OBJECTIVE_SCREEN_RESULT_V3.json`.

## S7. External/OOD robustness

The exact submitted EPIC model does not have an AsEP result in the manuscript evidence. The abstract claim is removed. Later FrameContact evidence on AsEP and antigen-cluster-disjoint/historical holdouts is presented only as subsequent work.

No prospective temporal OOD claim is made. A later admission protocol rejected all 14 candidate temporal cases before labels were opened; the gate is not relaxed for a more favorable table.

## S8. Calibration and training cost

For the official-455 post-submission retrain:

- trainable parameters: `2,395,374`;
- accelerator: one NVIDIA RTX 4090 per seed;
- selected epochs: 18, 14, and 20 for seeds 17, 29, and 43;
- executed epochs: 20, 19, and 20;
- aggregate epoch wall-time: 15,140 s, 15,016 s, and 15,253 s (approximately 4.17–4.24 h per seed);
- test labels were not opened by training or checkpoint selection.

The ensemble calibration diagnostics are poor enough to warrant caution: contact ECE-15 is `0.1217`, epitope `0.1167`, and paratope `0.1438`. The revised paper therefore does not advertise calibrated deployment probabilities.

## S9. Baseline comparability and modality

The original mixed table is split. Imported literature values remain context only, while same-protocol reruns are reported separately. EPIC is described as sequence-input + predicted monomer geometry, not sequence-only. Native-complex, predicted-monomer, and antigen-agnostic methods are labelled explicitly.

See `BASELINE_PROTOCOL_TABLE.md`.

## S10. Exact unresolved items

The response does not hide the following unavailable evidence:

- a checkpoint-matched OpenDDE-versus-AF3 substitution on the exact anonymous submission model;
- a frozen exact-EPIC learned-contact-gate versus content-only three-seed retrain;
- a multi-seed exact-EPIC Dice/count factorial;
- a fresh prospective biological test.

The accepted core is retained by narrowing claims, not by filling these gaps with successor results.
