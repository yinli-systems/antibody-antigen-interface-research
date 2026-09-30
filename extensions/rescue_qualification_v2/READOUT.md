# Completed rescue qualification work and fixed-rescoring readout

Snapshot: **2026-09-30T09:57:46.518065+00:00**. Branch `kevin/rescue-qualification-20260930`.
This extension did not change protected jobs 1640503, 1640514 or 1641627, did not rerun AF3 inference, and did not retrain the already-queued rescue models.

## 1. A proposed score correction failed on the complete fixed evaluation

Before transformed metrics were opened, exactly five label-independent score rules were frozen. They reuse official RePaRank sequence checkpoint outputs on the same 217 development-validation complexes, 68 split-authority components, 9,638,880 candidate pairs and 9,036 positives. This is not new blind evidence.

| Fixed score rule | Macro mean-per-complex AP | Mean Top-10 precision |
|---|---:|---:|
| identity | 0.037807791 | 0.054377880 |
| row_center | 0.015820726 | 0.023041475 |
| column_center | 0.010487973 | 0.019815668 |
| double_center | 0.003394227 | 0.004147465 |
| local_scale_10 | 0.032856294 | 0.047004608 |

All four proposed corrections reduce overall observed AP relative to identity. None is promoted. The identity AP replays the original receipt exactly (absolute difference 0). No model training, new feature extraction or GPU inference was required for this 1,085-row result set.

The primary local_scale_10 contrast is -0.0049515 equal-complex AP. Its equal-component effect is -0.0026992, with pointwise component-bootstrap 95% interval [-0.0071240, +0.0018503]. The interval spans zero: this is not a statistically established harm or equivalence claim. Double centering gives an equal-component effect of -0.0396716, interval [-0.0579679, -0.0231286]. These are two different weighting estimands; do not subtract the table means to recreate an equal-component interval.

The result rejects the simple assumption that all row/column interface propensity is nuisance. Useful interface signal is removed too. It does not establish causal binding specificity or prove that every possible learned correction fails. The appropriate next algorithmic hypothesis is to preserve informative marginal propensities while testing a separately learned residual; no such new model has yet been trained by this extension.

CSLS-inspired local scaling uses negative Euclidean scores here, not the exact MUSE cosine setup. k=10 was fixed, not selected after inspection. Full contrasts, signs by component and every per-complex result remain in `evidence/20260930/evidence_attempt2/RESCORE_RESULT.json`.

## 2. Pure permutation families are not independent rescue learning problems

For the current rowwise two-tower matcher, permuting extra features by p only permutes logit columns: Lp=L[:,p]. Row cross-entropy with argsort(p) and column cross-entropy with p are exact reindexings of the original losses. The parameter gradients agree in exact arithmetic. The feature reconstruction also agrees for every pure permutation when the optimal one-to-one assignment is unique.

An actual CPU float64 probe with an untrained copy of the original model and four real feature pairs confirmed a maximum loss difference of 8.88e-16 and gradient difference of 6.66e-16. Shift-8, same-AA, global shuffle and identity yielded identical reconstructed vectors in the measured cases. This is an architectural/mathematical check, not trained-model quality, a throughput benchmark, or a novel matching algorithm.

The downstream contact model still sees different corrupted inputs without rescue. Only the matcher problem is redundant under these pure permutations. Do not count those conditions as independent biological replications. Additive row/column potentials also do not change a complete square assignment optimum, so simply adding CSLS to that assignment is not an automatic rescue improvement.

A secondary content-canonical solver makes candidate-column tie ordering independent of presentation order. It uses only observed feature bytes, not oracle residue IDs or the true permutation. Its exact/duplicate/tied-input tests passed. It is not applied as a one-to-one biological antibody-antigen contact constraint.

## 3. Actual missing-feature failure reproduced and a mask-preserving path added

Original rescue-v1 admits 2,552 training and 240 validation full keys, but the validation cohort has 241 unique full keys. A missing 121-residue antibody ESM-IF1 key affects `pdb_00002p45_B_+` and `pdb_00002p47_B_+`. Calling the original evaluator loader on this real key raises ValueError. Its unconditional one-valued presence construction must not be used to fabricate availability.

The supplemental path retains all 217 complexes and delegates loading to the original CompositeEmbeddingStore. The missing side stays at extra_features=0, presence=0 in aligned, corrupted, rescued and clean-rescued conditions; no matcher is called on it. This was exercised on the real missing key. Partial-residue missingness, cross-chain movement or invented feature vectors fails closed.

Train/validation exact full-key overlap is 0, and exact biological-chain-part overlap is 0 (3,435 train parts / 286 validation parts). This does not establish homology-level separation or pretrained-model decontamination.

Our first new metadata probe wrongly assumed physical per-chain ESM-IF1 files rather than the legacy combined VH+VL full-key layout. That failed probe log is retained unchanged at `evidence/20260930/evidence/REAL_PROBE.log`. The lookup assumption was corrected in a new source snapshot; no training data, model or frozen authority was changed.

## 4. Existing trained rescue models and the new pending qualification

The original three training jobs `1641809_0–2`, already submitted before this review, all finished with exit 0:0. Their frozen synthetic-matching validation selection, not contact AP, produced:

| Rescue seed | Selected epoch | Synthetic matching validation accuracy |
|---:|---:|---:|
| 101 | 5 | 0.833653700 |
| 202 | 5 | 0.835892519 |
| 303 | 3 | 0.836141478 |

These accuracies are the original validation matching criterion, over admitted sequence keys. They are NOT contact AP, blind accuracy, held-out biological generalization or three independent corruption experiments. All 12-epoch terminal histories and selected checkpoint/seed/plan identities were verified before preparing the supplemental plan.

`qualified_evaluate.py` now reports aligned/corrupted/rescued/clean-rescued AP, an untrained-matcher baseline, and the canonical-solver alternative. It preserves historical masks and requires historical identity AP parity within 1e-6. All three seed-pair receipts and every 217-case/condition record are required for final component/paired-seed aggregation.

One new full-cohort evaluation was submitted as **1641906**, rescue seed101 paired with downstream seed1, maximum one GPU for 45 minutes. It was PENDING at the original snapshot and has since completed `0:0`; its terminal result is recorded in Section 7. Seeds202/303 are not part of that completed contact evaluation. No completed multi-seed qualification is claimed.

## 5. Current external work — observed, not modified

- AF3 `1641712_1–39`: all COMPLETED 0:0; with recovered task0, 40/40 prediction directories exist.
- The previously existing strict finalization job `1641841` now reports CANCELLED by the owning account. This review only read that state; it did not cancel it, retry blocked actions, or verify a finished assembled authority.
- SaProt/ProSST `1640503`: 12/36 completed; 12–17 running; 18–35 pending.
- `1640514` and `1641627`: pending dependencies, untouched.
- No layer winner was selected, and no partial sweep metric was used for tuning this extension.

## 6. Reproducible evidence and limits

48 tests passed from exact source commit `6f0962403b7d94bdbf37c38c155da8d97293f941`. The source archive SHA-256 is `4288e4fd44b6f18bd4c6e9afa8e0f65d09439b936ea51c270845fa84d1b2b35d` on both Mac and cluster. Earlier 32/44-test logs and the failed initial integration probe remain preserved.

- Fixed rescoring plan SHA: `34e638f4659c1bca7530f96cc850ca0cf5d039dbfac0992c0ae479ffcf5a178d`.
- Original rescue training plan SHA: `dbab0019832814ed89ab9c63c299e969509604e9d96606e9f0dfe6d4a58a7919`.
- Supplemental evaluation plan SHA: `9b84a97c3f68dbc0f6ba167b2ee573ee8a6e6f5c7663794b74b9af1a42ba907a`.
- The 20 exported evidence files are SHA/size verified against `evidence/20260930/IMPORT_MANIFEST.json`; no full embeddings or model weights are redistributed.

Completed: fixed full-cohort rescoring, original-loss algebra check on real features, real missingness reproduction and correction, strict plan preparation, unit tests, source implementation, and the first full trained-rescue contact evaluation (seed101/downstream1). Not completed: full three-seed rescue qualification, structural layer-sweep conclusions, untouched OOD, new affinity or co-folder endpoints. No SOTA or first-place claim is supported by this work.

## 7. First full trained-rescue qualification result (seed 101 / downstream seed 1)

`1641906` completed `0:0` on all 217 validation complexes. The imported raw receipt is
`evidence/20260930/evidence_attempt3/evaluation_runs/seed_101/COMPLETE.json` (SHA-256 `4e98f0798e3e7df0e1ce601069547188d56c01d8982db9dfe05f2c773e72f864`).
Identity parity error is exactly `0.0`; no test or OOD labels were read and contact AP was not used for rescue checkpoint selection.

| Condition | Aligned AP | Corrupted AP | Rescued AP | Rescue gain | Gap recovered |
|---|---:|---:|---:|---:|---:|
| `same_aa_shuffle:0` | 0.029983990 | 0.011026202 | **0.027099407** | **+0.016073205** | **84.78%** |
| `chain_shift:8` | 0.029983990 | 0.009560514 | **0.027099407** | **+0.017538893** | **85.88%** |
| `global_shuffle:0` | 0.029983990 | 0.009154905 | **0.027099407** | **+0.017944502** | **86.15%** |

The same trained matcher applied to already-aligned inputs gives AP `0.027099407`, a change of `-0.002884583` from aligned. This clean-input cost is material and prevents presenting the matcher as an unconditional production improvement.

This is a strong **single seed-pair mechanistic result**, not a finished promotion result. Seeds 202/303 and complete grouped/paired-seed inference are still required before claiming a replicated rescue effect. The result supports the narrower statement that, for seed101/downstream1, a label-free correspondence matcher recovers about 85–86% of the AP lost under three registered synthetic correspondence corruptions.

## 8. Seed-101 grouped inference after terminal completion

The terminal `1641906` receipt contains 651 rows because three corruption conditions are recorded for each of the same 217 complexes. They are **not 651 independent biological cases**. The registered grouping has 68 independent split-authority components. Recomputing paired effects by first averaging within each component and bootstrapping the 68 component effects gives:

| Condition | Contrast | Equal-component effect | Component bootstrap 95% CI | Positive components |
|---|---|---:|---|---:|
| same-AA | rescued − corrupted | +0.0157334 | [+0.0105792, +0.0221971] | 89.7% |
| shift-8 | rescued − corrupted | +0.0180131 | [+0.0129516, +0.0243604] | 95.6% |
| global shuffle | rescued − corrupted | +0.0183485 | [+0.0133422, +0.0247360] | 94.1% |
| all three pure permutations | rescued − untrained rescue | +0.0173965 | approximately [+0.0125, +0.0239] | 94.1% |
| clean input | clean-rescued − aligned | **−0.0020299** | **[−0.00324, −0.00089]** | 35.3% positive / 60.3% negative |

The equal-complex clean-input effect is `−0.002884583`, roughly a 9.6% reduction relative to aligned macro AP. No preregistered clean-input noninferiority margin exists in the frozen v1 training protocol, so this extension does **not** retrofit a margin after seeing the result and does not declare noninferiority. The interval being below zero establishes a development-set clean-input cost under this grouped estimand, not a blind generalisation claim.

The published gap-recovery ratio is exactly `(mean(rescued AP) − mean(corrupted AP)) / (mean(aligned AP) − mean(corrupted AP))` and reproduces `0.84784 / 0.85876 / 0.86151` for same-AA / shift-8 / global shuffle.

`canonical_rescued_ap`, `rescued_ap`, and `clean_rescued_ap` are numerically identical for every one of the 651 records in this seed-pair. This is expected for this matcher under pure column permutations when the one-to-one optimum is stable: changing presentation order permutes score-matrix columns but does not create a new correspondence problem, and applying the trained assignment to an already aligned input maps to the same learned canonical arrangement. This equality is therefore an architectural property/diagnostic, **not three independent successful rescue replications**.

The exact recomputation is preserved in `evidence/20260930/evidence_attempt3/SEED101_GROUPED_ANALYSIS.json`; the raw 651-row terminal receipt is preserved under `evaluation_runs/seed_101/COMPLETE.json`.

Remaining qualification requires the already trained rescue seeds 202 and 303 paired with downstream seeds 2 and 3 under the same frozen evaluation plan, followed by the preregistered crossed paired-seed/component aggregation. Until then the result remains single-seed-pair development evidence.

## 9. Final three-seed paired/component qualification

The remaining two evaluations completed successfully after preserving a zero-second launcher-path failure from job `1642017`. That failed launcher pointed to the earlier source directory and performed no model evaluation. Recovery job `1642022` changed only the launcher source path to the frozen `source_attempt3`; it did not retrain rescue models or rerun seed101.

The final frozen aggregate contains **1,953 records = 3 paired rescue/downstream seeds × 217 development complexes × 3 registered synthetic corruptions**. The three corruption families are not treated as independent biological replications. The authoritative aggregate SHA-256 is `b5dc2f31682a17502262c60c72cee9d926557f520ffe532da2b070ee421adcdc`.

| Condition | Aligned AP | Corrupted AP | Rescued AP | Gap recovered | Equal-component/equal-paired-seed rescue effect | 95% crossed bootstrap CI |
|---|---:|---:|---:|---:|---:|---|
| same-AA | 0.027994971 | 0.011100353 | **0.025606964** | **85.87%** | **+0.0157209** | **[+0.0112560, +0.0211527]** |
| shift-8 | 0.027994971 | 0.009683473 | **0.025606964** | **86.96%** | **+0.0173471** | **[+0.0128400, +0.0227690]** |
| global shuffle | 0.027994971 | 0.009398676 | **0.025606964** | **87.16%** | **+0.0177846** | **[+0.0133127, +0.0232857]** |

Against the untrained-matcher control, the equal-component/equal-paired-seed effect is **+0.0166045**, with pointwise crossed-bootstrap 95% CI **[+0.0119025, +0.0224902]**. `canonical_rescued_ap - rescued_ap` is exactly zero for all three conditions, consistent with the pure-permutation equivariance analysis; this is not an additional independent replication.

The clean-input cost also replicates across all three seed pairs. Equal-complex clean-rescued minus aligned AP is **−0.0023880**. The registered equal-component/equal-paired-seed effect is **−0.00143046**, with 95% crossed-bootstrap CI **[−0.00267527, −0.00010714]**. The frozen protocol did not preregister a clean-input noninferiority margin, so no post-hoc margin is introduced and no noninferiority claim is made.

Per-complex heterogeneity remains material. Averaged over the three seed pairs, same-AA rescue improves 87.1% of complexes but harms 12.9%; shift-8 and global rescue improve 95.9% and harm 4.1%. Clean rescue improves only 33.2% of complexes and harms 66.4%; its median effect is close to zero (`−0.0001001`) but the lower tail is substantial. Full quantiles and component summaries are in `THREE_SEED_HETEROGENEITY.json`.

The frozen split authority reports 68 validation components and zero split-component violations under its MMseqs gates: antibody full-sequence identity 0.80, antigen full-sequence identity 0.70, bidirectional coverage 0.80, plus the registered CDR-H3 rule. This remains development-only evidence; the split itself states that a prospective claim requires a post-cutoff cohort. No test or OOD labels were opened.
