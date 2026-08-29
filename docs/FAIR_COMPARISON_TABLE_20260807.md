# Fair baseline tables (receipt-generated)

> Generated from the frozen receipts by `scripts/render_fair_comparison_table.py`.
> Do not remove the scope labels: these rows are not one common leaderboard.

## Scope ledger

| Model | Target | Scope | Test opened for selection? | Native pair output? |
|---|---|---|---:|---:|
| EpiFormer | epitope | frozen SAbDab2 test (455 complexes; epitope projection only) | no | no |
| MIPE (validation-best) | epitope/paratope | official five-fold held-out SAbDab2 complexes | no | no |
| SPPIDER-seq | epitope/paratope | frozen SAbDab2 test (455 complexes) | no | no |
| FrameContact | epitope/paratope/contact | antigen-fold development validation | no | contact only in the pair head |

### Epitope (receipt-backed baseline rows)

| Model | AUROC | AUPRC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| EpiFormer | 0.8008 | 0.2775 | 0.2577 | 0.4785 | 0.3350 | 0.2715 |
| MIPE (validation-best) | 0.8074 | 0.2804 | 0.2246 | 0.5523 | 0.3194 | 0.2808 |
| SPPIDER-seq | 0.6718 | 0.1454 | 0.1309 | 0.5359 | 0.2104 | 0.1269 |
| FrameContact (frozen test not opened) | N/A | N/A | N/A | N/A | N/A | N/A |

## FrameContact contact development snapshot

> Native residue-pair contact output on the SAbDab2 antigen-fold validation split; 5 seeds at matched synchronized epochs. These values are development evidence only, not frozen-test results.

| Condition | Epoch | Pooled contact AUPRC | Mean per-complex AUPRC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| FrameContact unary-only | 21 | 0.0107 ± 0.0008 | 0.0227 ± 0.0019 | 0.0120 ± 0.0011 | 0.3200 ± 0.0712 | 0.0231 ± 0.0020 | 0.0567 ± 0.0045 |
| FrameContact distogram | 18 | 0.0109 ± 0.0012 | 0.0214 ± 0.0030 | 0.0147 ± 0.0012 | 0.2291 ± 0.0437 | 0.0275 ± 0.0022 | 0.0539 ± 0.0063 |

The fixed threshold is the raw-logit development diagnostic; no test checkpoint or threshold was selected from these rows.

### Paratope (receipt-backed baseline rows)

| Model | AUROC | AUPRC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| EpiFormer (no paratope head) | N/A | N/A | N/A | N/A | N/A | N/A |
| MIPE (validation-best) | 0.8811 | 0.6892 | 0.5586 | 0.7954 | 0.6563 | 0.5346 |
| SPPIDER-seq | 0.9486 | 0.6511 | 0.5693 | 0.7313 | 0.6402 | 0.6023 |
| FrameContact (frozen test not opened) | N/A | N/A | N/A | N/A | N/A | N/A |

### Contact (native pair score only)

| Model | AUROC | AUPRC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| EpiFormer (residue projection only) | N/A | N/A | N/A | N/A | N/A | N/A |
| MIPE (residue marginal only) | N/A | N/A | N/A | N/A | N/A | N/A |
| SPPIDER-seq (residue scores only) | N/A | N/A | N/A | N/A | N/A | N/A |
| FrameContact (development-only; not ranked here) | N/A | N/A | N/A | N/A | N/A | N/A |

## Required interpretation

- Use the epitope/paratope rows only as receipt-backed context; MIPE is five-fold held-out, while EpiFormer and SPPIDER are frozen SAbDab2 test.
- Keep FrameContact out of this frozen-test table until the validation-only winner, threshold and test-opening receipt exist.
- Never convert a residue score into a contact score. Contact AUPRC/precision/recall/F1/MCC require a native residue-pair output and the same candidate mask.
- The final fair comparison still requires one common AsEP split, five seeds, validation-only selection, calibration, partner shuffle/retrieval and the pre-registered external opening.
