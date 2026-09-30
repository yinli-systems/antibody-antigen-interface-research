# Canonical results index

This page is the human-readable index for the strongest completed evidence in the repository. It is deliberately more detailed than the root README and deliberately shorter than the full branch history.

## 1. Frozen representation transfer and correspondence

| Evidence | Cohort / protocol | Result | Status | Canonical artifact |
|---|---|---|---|---|
| ESM-IF1 aligned vs ESM-2 | SAbDab2 validation, 10 paired seeds | Δ macro pair AUPRC **+0.005111**; +22.8%; 95% bootstrap CI [0.003780, 0.006439] | Development attribution | [`representation_attribution_v1/report.json`](representation_attribution_v1/report.json) |
| ESM-IF1 shuffled vs ESM-2 | same | Δ **−0.003093**; −13.8%; CI [−0.004391, −0.001807] | Development attribution | same |
| Aligned vs shuffled | same | Δ **+0.008204**; +42.4%; 10/10 favourable seeds | Development attribution | same |
| Width-matched random vs ESM-2 | same | Δ −0.000595; CI crosses zero | **Undecided**, not equivalence | same |
| k=8 row displacement | corruption ladder | −0.0175 AP vs aligned; ≈94% of global permutation damage | Development perturbation | [`correspondence_corruption_curve_v1`](correspondence_corruption_curve_v1/) |
| Same-AA shuffle, ESM-IF1 | 217 validation complexes, 68 components, 3 fixed seeds | identity 0.027995 → 0.011100; equal-component/equal-seed Δ **−0.017151**, CI [−0.022688, −0.012434] | Post-freeze exploratory development | [`../extensions/correspondence_v2/READOUT.md`](../extensions/correspondence_v2/READOUT.md) |
| Holdout ProteinMPNN | one registered 217-complex read, 10 paired seeds | aligned−shuffled **+0.003594**, 9/10, CI [0.001559, 0.005881] | Holdout confirmation | [`holdout_confirmation_v1/confirmation.json`](holdout_confirmation_v1/confirmation.json) |
| Holdout ESM-IF1 | same | **+0.002260**, 6/10, CI [−0.000007, 0.004901] | **Undecided** | same |

The representation paper's complete number-to-artifact mapping is [`CLAIMS_LEDGER.md`](CLAIMS_LEDGER.md). The ledger explicitly withdraws an earlier unsupported “+30.5%, 20 seeds” headline and records the correct arm-specific values instead.

### Three-seed learned correspondence rescue (development only)

A learned residue-correspondence matcher was evaluated on the same 217 development complexes across three paired rescue/downstream seeds and 68 frozen split-authority components. For same-amino-acid shuffling, the crossed equal-component/equal-seed rescued−corrupted effect is **+0.015721 pair AP** with 95% CI **[+0.011256,+0.021153]**; rescued−untrained is **+0.016605 [ +0.011902,+0.022490 ]**. The same matcher also reproduces a clean-input cost: clean-rescued−aligned is **−0.001430 [−0.002675,−0.000107]**. This is strong development evidence for recoverable synthetic misregistration, but it is **not OOD, not a production improvement, and not clean-input noninferiority**; no noninferiority margin was added after seeing the result.

Canonical research snapshot: [`kevin/rescue-qualification-20260930@fdb51b2`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/fdb51b29c6fb22ccc55e89f05e38a779d64389e2/extensions/rescue_qualification_v2).

#### Train-calibrated label-free gate

The follow-up gate freezes a 99.5th-percentile threshold from 256 clean **training** sequence keys per matcher seed and never uses contact AP to choose the threshold. On the 217 development complexes, all three registered synthetic corruptions trigger at 100%, so gated recovery equals unconditional recovery: same-AA equal-component/equal-seed Δ **+0.015721**, crossed seed×component 95% CI **[+0.011160,+0.021192]**. Clean damage is reduced but not eliminated: gated-clean−aligned is **−0.000460 [−0.001143,+0.000020]**, versus unconditional **−0.001430 [−0.002716,−0.000085]**.

The important negative result is calibration shift. Clean trigger rates are 0.39% on the train calibration keys for every seed, but **15.67% / 17.97% / 0.46%** on validation for seeds 101/202/303. The 101/202 trigger sets overlap strongly (31 shared cases; Jaccard ≈0.738) and concentrate in 15–16 components. Therefore the train quantile is **not** presented as a validation or deployment guarantee. The synthetic corruptions are also not treated as real biological OOD; a frozen label-free heuristic diagnostic now tests whether 100% detection is largely a property of the explicit permutation construction.

Raw 651-row seed receipts, crossed analysis, hashes and reproduction code are pinned at [`kevin/rescue-qualification-20260930@cc10181`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/cc101811c7452a593d24040c7b4159f7bff641d0/extensions/rescue_qualification_v2/evidence/20260930/gated_rescue_v1). The human-readable gate readout is at [`20d55fe`](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/20d55fe5f8bd5606b569ab1a7767e9baf9fa344a/extensions/rescue_qualification_v2/GATED_RESCUE_READOUT.md).

## 2. Objective interference and optimisation dynamics

| Evidence | Result | Boundary | Artifact |
|---|---|---|---|
| Contact supervision | pair AUPRC **+0.0833** (`0.0905` vs `0.0072`) | Development factorial ablation | [`objective_component_localisation_v1/PART_TWO.md`](objective_component_localisation_v1/PART_TWO.md) |
| Remove complex-ranking term | pair AUPRC **+0.0214** and node MCC **+0.0369** across 10 paired seeds | Intervention on training procedure | same |
| Ranking alone | node MCC **−0.0748** vs node-only | Development negative result | same |
| Gradient sign change | crossings 77–110 steps on AsEP; 38–50 on SAbDab2 | Diagnostic; not a causal explanation | [`gradient_alignment_dynamics_v1/step_trace_summary.json`](gradient_alignment_dynamics_v1/step_trace_summary.json) |

## 3. FrameContact and predicted geometry

### No-privileged-distance development

Three fixed seeds using sequence + independently predicted intra-chain geometry:

| Task | Median mean-per-complex AP |
|---|---:|
| Contact | **≈0.0261** |
| Epitope | **≈0.2327** |
| Paratope | **≈0.6962** |

Source: [`../paper/neurips_2026_framecontact/TEACHER_HANDOFF_20260903.md`](../paper/neurips_2026_framecontact/TEACHER_HANDOFF_20260903.md).

### Correctly aligned historical test549

The original 455 cache was found to overlap the selected checkpoint's training/validation authority (272 train / 73 validation / only 110 held-out). The corrected checkpoint-aligned test contains 549 historical instances. Fixed routed three-seed medians:

| Task | AUROC | Pooled AP | MPC-AP | MCC |
|---|---:|---:|---:|---:|
| Contact | 0.92580 | 0.01001 | **0.02667** | 0.03461 |
| Epitope | 0.67184 | 0.12599 | **0.22031** | 0.12475 |
| Paratope | 0.95399 | 0.63403 | **0.69859** | 0.61144 |

This is an **exposed historical antigen-fold test** with AF3-predicted geometry; it is not fresh blind evidence and does not prove sequence-double-cold generalisation. See [`aligned_test549_replay_v1/RESULT.json`](aligned_test549_replay_v1/RESULT.json) and the correction note in the paper directory.

### Top-10 case-study signal

A fixed 20-instance historical case study contains 9 true contacts among 200 Top-10 predictions: precision 0.0450, recall 0.01048. Mean contact prevalence is 0.000979, so Top-10 precision is ~46× the equal-complex random expectation, while absolute recall remains low. See [`sabdab2_top10_case_study_v1`](sabdab2_top10_case_study_v1/).

## 4. Official RePaRank on this project's exact pair grid

Official RePaRank **sequence** checkpoint, no added head/fine-tuning, 217 validation complexes:

| Metric | Value |
|---|---:|
| Macro mean-per-complex pair AP | **0.0378078** |
| Equal-component AP estimate | **0.0432216** |
| Component bootstrap 95% interval | **[0.0263310, 0.0621867]** |
| Pooled pair AP | **0.0087386** |
| Pooled pair AUROC | **0.786064** |
| Epitope macro AP | **0.202039** |
| Paratope macro AP | **0.557707** |
| Candidate pairs | **9,638,880** |
| Positive contacts | **9,036** |

Source: [`../extensions/frontier_multimodel_v1/evidence/reparank_same_protocol_v1/sequence_full/evaluation.json`](../extensions/frontier_multimodel_v1/evidence/reparank_same_protocol_v1/sequence_full/evaluation.json).

Boundary: the public checkpoint's training accessions are not bound by the official repository, so overlap with this cohort is unknown. Treat this as an **exposed pretrained checkpoint diagnostic**, not a blind architecture comparison.

### Sequence+structure checkpoint: official-style centering correction

The strict predicted-chain authority now covers all 217 cases (3,385 chain units; 40 newly strict-audited AF3 chains) without using bound antibody–antigen poses. The first complete structure run omitted RePaRank's official per-chain `CenterMesh` preprocessing and is retained as an uncentered negative control. The corrected hash-bound centered resume (`1642417`) completed 217/217 cases using **12 validated centered artifacts + 205 fresh forwards**, with the exact same 9,638,880-pair / 9,036-positive grid as sequence-only.

| Arm | Macro pair AP | Pooled pair AP | AUROC | Epitope AP | Paratope AP |
|---|---:|---:|---:|---:|---:|
| sequence | **0.037808** | **0.008739** | **0.786064** | **0.202039** | **0.557707** |
| structure, uncentered protocol mismatch | 0.010743 | 0.002896 | 0.660171 | 0.182829 | 0.409777 |
| structure, official-style centered | **0.016014** | 0.002366 | 0.672153 | **0.193265** | **0.479202** |

Across 68 frozen components, centered−uncentered pair AP is **+0.004142**, 95% component-bootstrap CI **[−0.003506,+0.011360]** (directional but undecided); centered−sequence is **−0.026838 [−0.043918,−0.011823]**. Centering clearly recovers paratope performance relative to uncentered (**+0.065083 [ +0.022944,+0.107171 ]**) but does not close the sequence gap. Raw pooled AP moves in the opposite direction from macro AP; post-hoc within-complex percentile/z calibration reverses the centered-vs-uncentered pooled ordering, indicating cross-complex score calibration contributes to that discrepancy. Those normalized pooled values are diagnostic only and do not replace the registered raw endpoint.

This does **not** show that structure is useless: the checkpoint receives independently predicted chain-local N/CA/C/O geometry, not a correctly docked complex pose, and checkpoint training overlap is unknown. Full audit, failure provenance for `1642386/1642395`, exact commands, grid hashes and heterogeneity are pinned at [`kevin/rescue-qualification-20260930@fdb51b2`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/fdb51b29c6fb22ccc55e89f05e38a779d64389e2/extensions/frontier_multimodel_v1).

## 5. Partner-score dissociation

A full deduplicated 122×119 Cartesian grid (14,518 pairs; nine checkpoint/condition GPU tasks) resolves an earlier small-panel anomaly:

- the MRR increase after correspondence corruption does **not** robustly replicate; pointwise CIs span zero;
- the current complex score has mean Spearman ≈ **0.9743** with double-centred interaction-matrix spread;
- substantial within-anchor associations exist with candidate length, surface proxy and node priors;
- score changes are weakly coupled to matched-complex contact-AP changes.

Canonical readout: [`../extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_READOUT.md`](../extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_READOUT.md).

## 6. External benchmark check

AsEP is retained to prevent internal improvements from being mistaken for benchmark leadership:

- `epitope_group` test, 10-seed single-model macro MCC: **0.1572**;
- `epitope_ratio` test macro MCC: **0.258**;
- published configurations on those protocols can be higher.

See [`asep_benchmark_v1/BENCHMARK.md`](asep_benchmark_v1/BENCHMARK.md).

## 7. Active structural-PLM programme (historical snapshot below)

For the latest timestamped observation, use [operations/README.md](operations/README.md). The table below preserves the first consolidation snapshot.

The frozen SaProt/ProSST formal sweep contains 36 tasks. The current committed snapshot contains six completed SaProt tasks:

| Layer | Seed APs | 3-seed mean AP |
|---|---|---:|
| SaProt 0 | 0.017988 / 0.021676 / 0.022082 | **0.020582** |
| SaProt 8 | 0.020780 / 0.020481 / 0.021858 | **0.021040** |

These are **intermediate development results**, not a layer selection. Exact-width/coverage random controls and the remaining SaProt/ProSST layers must finish before promotion. See [`CURRENT_STATUS_20260930.md`](CURRENT_STATUS_20260930.md).

## 8. Negative and failed results remain first-class evidence

The consolidated main intentionally preserves, among others:

- Hadamard pair decoder did not win;
- focal pair-loss intervention failed against its control;
- geometry five-seed superiority gate failed;
- the first 455-case FrameContact test alignment was invalid for the selected checkpoint and was corrected;
- temporal double-cold admission failed closed rather than opening labels;
- compiled antibody pretraining improved throughput but failed correctness/tail-latency gates;
- RePaRank structure full evaluation failed closed when predicted-chain authority was incomplete;
- AF3 completion task-0 currently has a pre-inference orchestration failure and is not counted as a structure result.

For exhaustive discovery, use the [clickable branch evidence directory](ALL_BRANCHES.md), [`BRANCH_EVIDENCE_INDEX.json`](BRANCH_EVIDENCE_INDEX.json), and the [complete tracked-tree manifest](../archive/FULL_SNAPSHOT_MANIFEST_20260930.json).

All paper-level interpretation must also follow [CLAIM_BOUNDARIES.md](../docs/CLAIM_BOUNDARIES.md). The first consolidation README used overly general “necessary” wording; it is superseded by scoped empirical statements. Historical raw reports remain unchanged.

## 9. Execution repair and subsequent operational snapshots

The [engineered recovery](../extensions/execution_recovery_v1/README.md) repairs AF3 staging/runtime and the frozen analyzer receipt schema without modifying existing experiments. See [current timestamped operations](operations/README.md). These are engineering and scheduling outcomes, not additional model-performance claims.
