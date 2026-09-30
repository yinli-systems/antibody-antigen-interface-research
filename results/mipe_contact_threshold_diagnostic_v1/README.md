# Retained MIPE attention threshold diagnostic

For the checkpoint track behind the manuscript's historical MIPE residue rows,
use [the paper-track recovery](../mipe_paper_track_fixed05_recovery_v1/README.md).
This directory retains the earlier export, whose selected epochs differ.

This directory reconstructs fixed-threshold pair statistics from the 560
retained five-fold held-out MIPE interaction-map artifacts. The source maps are
head-averaged, row-normalized bidirectional cross-attention weights. They are
not independent sigmoid contact probabilities. The retained export summary and
`scripts/mipe_train.py::joint_loss` show that both directional maps were trained
with BCE against the pair-contact labels. This is a supervised attention
readout; its performance should not be dismissed merely because it is attention.

At threshold 0.5, the mean-bidirectional map gives:

- TP / FP / TN / FN: `1 / 161 / 11,412,941 / 18,315`
- precision: `0.00617284`
- recall: `0.00005460`
- MCC: `0.00043020`

The micro metrics sum TP/FP/TN/FN over all 11,431,418 candidate pairs before
division; there are 18,316 positive pairs. Prediction uses `score >= 0.5`.
The threshold can be reported as requested, with the normalized-attention score
definition stated. The low Recall does not authorize choosing a new threshold
on these held-out labels.

This retained `formal_seed42` export uses selected fold epochs
`7 / 80 / 71 / 78 / 97`. It covers 560 five-fold held-out complexes, not the
455-sample SAbDab2 test set. It may be reported as **MIPE attention-readout
contact adaptation (fixed 0.5)** in a separately scoped row. A main table on the
455 cohort should leave this entry unreported until a matching evaluation is
available. Do not label the recovered value as an official published contact
head result or combine it with another MIPE checkpoint track.

The historical residue Recall values `0.5523` (epitope) and `0.7954` (paratope)
in `results/baseline_performance_20260806.csv` use **fold-specific
validation-fitted thresholds**. They are not fixed-0.5 results. The preceding
chat handoff omitted this qualification; it is corrected here. That CSV also
records EpiFormer threshold 0.3 and SPPIDER validation-selected thresholds, so
the statement that all historical baselines used 0.5 is not supported.

Artifacts:

- `MIPE_CONTACT_THRESHOLD_RESULT.json`: aggregate statistics and claim scope;
- `MIPE_CONTACT_THRESHOLD_PER_COMPLEX.csv`: per-complex confusion counts;
- `MIPE_CONTACT_THRESHOLD_RECEIPT.json`: source-tree and output hashes.
- `MIPE_CONTACT_SOURCE_AUDIT.json`: independent raw-array/CSV confusion check,
  directional normalization, original export checkpoint epochs, and thresholds.

The source audit retains the original pooled ranking AUROC/AP
`0.5071899891 / 0.0017185450` for the mean-bidirectional map. They are historical
ranking metrics, not newly selected scores. The original summary binds
checkpoint paths and epochs but does not record checkpoint byte hashes.
