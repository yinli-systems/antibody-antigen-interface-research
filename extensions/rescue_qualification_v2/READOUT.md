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

One new full-cohort evaluation was submitted: **1641906**, rescue seed101 paired with downstream seed1, maximum one GPU for 45 minutes. At the snapshot it is PENDING (Priority). Seeds202/303 have not been submitted by this review. No trained rescue contact-AP improvement or completed multi-seed qualification is claimed.

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

Completed: fixed full-cohort rescoring, original-loss algebra check on real features, real missingness reproduction and correction, strict plan preparation, unit tests, source implementation and one new evaluation submission. Not completed: trained rescue contact evaluation, full three-seed qualification, structural layer-sweep conclusions, untouched OOD, new affinity or co-folder endpoints. No SOTA or first-place claim is supported by this work.
