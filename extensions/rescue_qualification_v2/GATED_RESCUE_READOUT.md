# Label-free gated rescue readout — 2026-09-30

## Status

Job array `1642673_[0-2]` completed `0:0`. Each seed receipt contains 651 rows = 217 development complexes × three registered synthetic corruptions. The gate did not use contact labels, did not run the downstream contact model, and did not read test/OOD labels. It reuses the already-completed unconditional-rescue AP receipts.

This remains **development-only synthetic-corruption evidence**. It is not a biological misregistration benchmark, OOD result, deployment guarantee, or SOTA claim.

## Frozen gate

For each biological chain the trained matcher computes

`(maximum one-to-one assignment logit sum - identity-diagonal logit sum) / residue_count`.

The complex score is the maximum across present antibody/antigen sides. Each rescue seed freezes its threshold from at most 256 **training** sequence keys only: the 99.5th percentile of clean train scores, NumPy `method="higher"`. No validation contact AP, test/OOD label, or hidden true corruption permutation is used for threshold selection.

## Trigger coverage

| Rescue seed | train-clean trigger | validation-clean trigger | same-AA | shift-8 | global shuffle |
|---|---:|---:|---:|---:|---:|
| 101 | 0.39% | **15.67%** | 100% | 100% | 100% |
| 202 | 0.39% | **17.97%** | 100% | 100% | 100% |
| 303 | 0.39% | **0.46%** | 100% | 100% | 100% |

The 0.39% calibration rate is **not** a validation guarantee. Seeds 101/202 show a large calibration→validation shift.

The clean-trigger sets are not independent random accidents: 31 of the 34 seed-101 triggers overlap the 39 seed-202 triggers (Jaccard ≈0.738). They are concentrated in only 15–16 of the 68 components. Seed 303 has a much higher frozen threshold (2.997 vs 0.988/0.872) and triggers only one validation complex. This points to a mixture of systematic cohort/component shift and seed-dependent matcher score calibration.

## 68-component × three paired-seed crossed analysis

The analysis first averages within each of the 68 frozen split-authority components and then resamples both the three paired seeds and 68 components. Three corruption conditions are never counted as independent biological replicates.

### Same-AA corruption

| Contrast | Equal-component/equal-seed Δ pair AP | 95% crossed seed×component CI |
|---|---:|---:|
| gated rescue − corrupted | **+0.015721** | **[+0.011160,+0.021192]** |
| unconditional rescue − corrupted | **+0.015721** | **[+0.011210,+0.021393]** |
| gated − unconditional | 0 | [0,0] |

All synthetic corruptions trigger, so gated and unconditional recovery are identical. This is useful as a mechanism check but also means the synthetic detection task may be easy by construction.

### Clean input

| Contrast | Equal-component/equal-seed Δ pair AP | 95% crossed seed×component CI |
|---|---:|---:|
| gated clean − aligned | **−0.000460** | **[−0.001143,+0.000020]** |
| unconditional clean-rescued − aligned | **−0.001430** | **[−0.002716,−0.000085]** |
| gated − unconditional clean | +0.000971 | [−0.000306,+0.002206] |

The gate reduces the point-estimate clean damage by roughly two thirds, but the crossed interval still reaches slightly above zero and the seed effects are heterogeneous:

- seed101: −0.000788 equal-component clean effect;
- seed202: −0.000611;
- seed303: +0.000020.

Therefore the gate is **not clean-safe or production-qualified**. No post-hoc noninferiority margin is introduced.

## Why 100% synthetic detection needs a stronger control

The registered perturbations explicitly permute the extra-feature rows relative to the base-feature rows. The current gate directly compares identity assignment against the matcher's optimal assignment, so strong synthetic detection can partly reflect this construction. The Hungarian solver itself is a complete one-to-one assignment optimizer; detecting a deliberately permuted presentation is not evidence that the same score detects naturally occurring biological/representation misregistration.

A new label-free diagnostic is frozen in `gate_heuristic_diagnostic.py`. For the same train-only 99.5% threshold rule it compares:

1. current optimal-assignment gain;
2. fraction of residues moved by the optimal assignment;
3. normalized mean assignment displacement;
4. row-wise top-1 non-identity fraction;
5. mean row-max minus diagonal score.

It also records the known synthetic permutation moved fraction **only as an oracle perturbation-severity diagnostic**, never as a deployable gate. Job array `1642830_[0-2]` was submitted after source hash freeze; it performs no contact inference and was still pending at this snapshot because the existing structural-PLM sweep occupied GPUs. No result is claimed before those receipts complete.

## Selective-prediction boundary

Selective classification formalizes abstention as a risk–coverage trade-off, but guarantees require the assumptions of the particular method and calibration setting. Gangrade et al. (AISTATS 2021) provide guarantees for their one-sided-prediction formulation; those guarantees do not automatically apply to this molecular gate. Cattelan & Silva's distribution-shift study shows that selective behavior/confidence can change under shift. The observed 0.39%→15–18% clean-trigger shift is therefore reported empirically rather than treated as a contradiction of, or guarantee from, those papers.

References:
- https://proceedings.mlr.press/v130/gangrade21a.html
- https://openreview.net/pdf?id=FiqXqKR26c

## Canonical artifacts

- `evidence/20260930/gated_rescue_v1/runs/seed_0.json`
- `evidence/20260930/gated_rescue_v1/runs/seed_1.json`
- `evidence/20260930/gated_rescue_v1/runs/seed_2.json`
- `evidence/20260930/gated_rescue_v1/GATED_RESCUE_CROSSED_ANALYSIS.json`
- `aggregate_gated_rescue.py`
- `GATED_RESCUE_PROTOCOL.md`
- `gate_heuristic_diagnostic.py`

The next scientific step is not to tune this threshold on the same 217 contact outcomes. It is to finish the label-free heuristic diagnostic, preserve the negative calibration-shift result, and then evaluate any frozen selective rule only on an actually untouched cohort when one passes the admission gate.

## Cheap-gate diagnostic: synthetic detection is largely construction-driven

Job array `1642830_[0-2]` completed `0:0` without reading contact metrics or test/OOD data. The same train-only 99.5th-percentile threshold rule was applied to five label-free statistics. On validation clean inputs, seed101/202/303 false-trigger rates were:

| statistic | seed101 | seed202 | seed303 |
|---|---:|---:|---:|
| optimal assignment gain | 15.67% | 17.97% | 0.46% |
| moved fraction | 11.06% | 13.36% | 0.46% |
| mean absolute displacement | 10.14% | 16.13% | 0.46% |
| row top-1 nonidentity | 8.29% | 16.13% | 0.46% |
| row-max minus diagonal | 15.21% | 25.35% | 0.46% |

For same-AA corruption, validation detection is 100% for optimal-gain, moved-fraction and mean-displacement in all three seeds; row-top1 is 100% / 100% / 98.16%. Global shuffle is 100% for every listed detector and seed. Shift-8 is also 100% for most detectors, although seed303 mean-displacement detects only 10.60%, demonstrating that the detectors are not mathematically identical.

The known synthetic construction moves a very large fraction of residues: mean validation moved fraction is **1.00** for shift-8, **0.9969** for global shuffle and **0.9115** for same-AA shuffle. Those values use the hidden construction only as a severity audit and are forbidden as a deployable gate.

Therefore the headline “100% synthetic corruption detection” is not a unique property of the learned optimal-gain selector and should not be treated as evidence for natural biological misregistration detection. The synthetic rescue line is considered mechanistically informative but saturated for development tuning. No threshold will be retuned on the 217 contact outcomes. The next high-value experiment is strict partner-specific retrieval under prospectively frozen dual-axis homology separation.
