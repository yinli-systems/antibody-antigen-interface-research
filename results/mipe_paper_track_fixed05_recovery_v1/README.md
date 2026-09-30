# MIPE paper-row checkpoint recovery at fixed threshold 0.5

Status: `PASS_MIPE_PAPER_TRACK_FIXED05_RECOVERY`.

This recovery uses the exact original evaluation behind the historical MIPE
residue rows AUROC/AP `0.8074187010 / 0.2804073689` (epitope) and
`0.8810671199 / 0.6892213182` (paratope). It verifies 1,120 original NPZ hashes,
ten fold manifests, checkpoint hash bindings, and original evaluation SHA-256:
`6f2eee95167e75ae0fb88ceb3dec047c01b8d5812c063df0a45f6e0375896a5a`.

The cohort is **560 five-fold held-out complexes per checkpoint-selection
role**, not the SAbDab2 455 test cohort. No inference, training, or threshold
selection was rerun. Ranking metrics below come from the original evaluation
for those same verified predictions; threshold counts were recovered from the
raw arrays using `score >= 0.5` and pooled before calculating ratios.

## Validation-best checkpoint selection

Selected epochs by fold 0–4: `572 / 605 / 704 / 522 / 771`.

| Task | Pooled AUROC | Pooled AP | Micro MCC @0.5 | Micro Recall @0.5 | Micro Precision @0.5 |
|---|---:|---:|---:|---:|---:|
| Contact, mean-bidirectional attention | 0.498119 | 0.001596 | -0.000172 | 0.000000 | 0.000000 |
| Epitope | 0.807419 | 0.280407 | 0.232395 | 0.144572 | 0.464835 |
| Paratope | 0.881067 | 0.689221 | 0.508601 | 0.565147 | 0.675023 |

Contact TP / FP / TN / FN: `0 / 210 / 11,412,892 / 18,316`.

## Epoch-800 endpoint

| Task | Pooled AUROC | Pooled AP | Micro MCC @0.5 | Micro Recall @0.5 | Micro Precision @0.5 |
|---|---:|---:|---:|---:|---:|
| Contact, mean-bidirectional attention | 0.499233 | 0.001593 | 0.000116 | 0.000055 | 0.002364 |
| Epitope | 0.791853 | 0.251433 | 0.188911 | 0.106420 | 0.430453 |
| Paratope | 0.843347 | 0.621497 | 0.402637 | 0.402944 | 0.655593 |

Contact TP / FP / TN / FN: `1 / 422 / 11,412,680 / 18,315`.

## Corrections to the prior chat handoff

- The previously recovered `1 / 161 / 11,412,941 / 18,315` contact confusion
  came from the earlier `formal_seed42/test_interactions` export, selected at
  epochs `7 / 80 / 71 / 78 / 97`. It is retained separately under
  `results/mipe_contact_threshold_diagnostic_v1`; it does **not** correspond to
  the paper-row validation-best checkpoints above.
- Historical validation-best residue Recall `0.552345 / 0.795442` uses
  fold-specific validation-fitted thresholds. Under fixed 0.5 the corresponding
  values are `0.144572 / 0.565147`. Neither convention should be relabeled as
  the other.
- The contact matrices are row-softmax cross-attention scores with pair-label
  supervision. Fixed-0.5 results can be reported as a **MIPE attention-readout
  contact adaptation**, explicitly identifying the readout and cohort. They
  must not be described as an official published independent contact head, and
  poor performance is not a reason to hide the result or refit the threshold on
  held-out labels.
- A table claiming all methods were evaluated on the same 455 SAbDab2 samples
  cannot insert this 560-cohort row without correcting that scope statement.

## Artifacts and remote path

`MIPE_PAPER_TRACK_FIXED05_METRICS.csv` provides six compact table rows.
`MIPE_PAPER_TRACK_FIXED05_RESULT.json` binds source evaluation, checkpoint and
prediction manifests, metrics, and historical fitted-threshold counts.
`MIPE_PAPER_TRACK_FIXED05_PER_COMPLEX.csv` preserves per-complex confusion counts;
the receipt hashes all three outputs. No raw predictions or model weights are
published in this directory.

```text
/ssd/scxi253/model_deployment/results/mipe/paper_track_fixed05_recovery_v1/
```

The recovery is implemented by
`remote_tools/mipe_replay_v1/recover_mipe_paper_track_fixed05_v1.py`.
