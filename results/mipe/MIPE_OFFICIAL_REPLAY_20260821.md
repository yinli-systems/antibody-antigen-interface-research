# MIPE five-fold replay and pair-level audit

This release records the completed five-fold evaluation of the released-code-
compatible `lr=1e-3` track on the 64-complex provided test split, OOF-only
threshold calibration, and a bound-coordinate oracle diagnostic. The supplied
test split had pre-scoring schema access, so this is **not** a pristine blind
test or evidence of external generalization.

## Main five-fold ensemble

| Selection | Side | AUROC | AUPRC | MCC at 0.5 | OOF-calibrated MCC |
|---|---|---:|---:|---:|---:|
| Epoch 800 | Epitope | 0.875361 | 0.479469 | 0.219392 | 0.454020 |
| Epoch 800 | Paratope | 0.844130 | 0.616560 | 0.396438 | 0.458100 |
| Validation best | Epitope | 0.892775 | 0.513218 | 0.317949 | 0.484753 |
| Validation best | Paratope | 0.875432 | 0.669642 | 0.492002 | 0.509856 |

The published joint-MIPE reference reports epitope AUROC/AUPRC/MCC
`0.815/0.422/0.235` and paratope `0.894/0.738/0.531`. The validation-best replay
exceeds the published epitope values but remains below the published paratope
values. This is not a clean reproduction claim because the released code uses
`lr=1e-3`, while Appendix section 2.3 states `lr=1e-4`.

## Pair-contact ranking

The provided test set contains 1,070,174 candidate residue pairs and 2,056
contacts (prevalence `0.001921`).

| Selection and score | Macro AUPRC | Pooled AUPRC | Recall@1 | MRR |
|---|---:|---:|---:|---:|
| Epoch-800 native mean-bidirectional attention | 0.005400 | 0.002040 | 0.000000 | 0.007801 |
| Validation-best native mean-bidirectional attention | 0.005535 | 0.002080 | 0.000000 | 0.008100 |
| Epoch-800 residue outer product | 0.066065 | 0.035808 | 0.078125 | 0.206008 |
| Validation-best residue outer product | 0.073666 | 0.041906 | 0.078125 | 0.207587 |

The native interaction matrices do not support a useful contact-ranking claim:
their Recall@1 is zero and pooled AUPRC is only slightly above class prevalence.
The residue-score outer product is materially stronger, but it is a marginal-
propensity baseline rather than direct evidence of partner-specific coupling.

## Bound-coordinate oracle diagnostic

**Bound-complex coordinate oracle diagnostic only.** It directly uses the common
antibody-antigen coordinate frame and is not a deployable unbound or sequence-
only predictor.

| Side/task | AUROC | AUPRC | CV-threshold MCC | Recall@1 | MRR |
|---|---:|---:|---:|---:|---:|
| Epitope | 0.988908 | 0.892403 | 0.782226 | - | - |
| Paratope | 0.946485 | 0.862829 | 0.680113 | - | - |
| Pair contact | - | 0.485569 macro / 0.445096 pooled | - | 0.609375 | 0.785156 |

This result verifies that the labels are geometrically recoverable from the
bound structures. It must not be compared to sequence-only or unbound models as
if it were a deployable predictor.

## Appendix-correct replay status

A fresh five-fold, 800-epoch replay with Appendix section 2.3 settings
(`lr=1e-4`, Adam, batch size 16, seed 42) is still running. At the release
snapshot, the five folds were at epochs `15/13/13/12/11`; no terminal metric is
reported. The immutable settings are in
[`mipe_paper_appendix_lr1e4_seed42.json`](../../protocols/mipe_paper_appendix_lr1e4_seed42.json).

The machine-readable result and SHA-256 receipts are in
[`MIPE_OFFICIAL_REPLAY_20260821.json`](MIPE_OFFICIAL_REPLAY_20260821.json). Full
predictions, checkpoints, raw data, cluster paths, and job identifiers are
intentionally excluded from Git.

## Sources

- [MIPE official repository](https://github.com/WangZhiwei9/MIPE)
- [IJCAI 2024 paper and appendix](https://www.ijcai.org/proceedings/2024/0669.pdf)
