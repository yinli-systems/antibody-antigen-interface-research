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
