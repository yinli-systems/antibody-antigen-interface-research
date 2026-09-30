# Frozen SAbDab2 Top-10 contact case study

Slurm job `1552028` completed `0:0` on one RTX 4090. The evaluator returned
`PASS_SABDAB2_TOP10_CONTACT_CASE_STUDY`; stderr is empty.

## Outcome

The fixed random sample contains 20 of the 549 historical SAbDab2 test
instances. Ten residue pairs were selected by model score in every instance,
giving 200 selected pairs in total.

| Quantity | Value |
|---|---:|
| True negatives | 1,060,299 |
| False positives | 191 |
| False negatives | 850 |
| True positives | 9 |
| Top-10 precision | 0.0450 |
| Top-10 recall | 0.01048 |
| Instances with at least one TP@10 | 4 / 20 |
| Unique PDB entries with at least one TP@10 | 2 |

The mean per-complex contact prevalence is `0.000979`. Thus the observed
Top-10 precision is approximately 46-fold above the equal-complex random
expectation, although absolute recall remains low. This is descriptive evidence
from a small, previously exposed historical test sample, not a fresh blind
performance estimate.

## Case-study candidates

- `pdb_00008ol9_B_A`: 3 true contacts among the Top-10, first true contact at
  rank 1, full-grid AP `0.14151`. This is the strongest single candidate.
- PDB `6vmk`: three sampled antibody/antigen instances each contain two true
  contacts among the Top-10 and place the first true contact at rank 1. The
  `H:G` instance has full-grid AP `0.08438`; `V:U` and `i:h` are near-identical
  at `0.08317` and `0.08418`. Treat these as one structural case family rather
  than three independent examples.

The remaining 16 instances contain no true contact among their Top-10 pairs.
They remain useful as failure cases but should not be presented as successful
qualitative examples.

## Files

- `TOP10_CASE_STUDY_RESULT.json`: complete machine-readable result, including
  all 20 confusion matrices and all 200 ranked residue pairs.
- `top10_case_confusions.csv`: one row per sampled instance.
- `top10_ranked_pairs.csv`: one row per selected residue pair with 1-based
  antibody and antigen positions, score, and label.
- `top10_case_study_summary.png` and `.pdf`: visually checked summary figure.
- `TOP10_CASE_STUDY_RECEIPT.json`: result, protocol, checkpoint, and sample
  bindings.

The confusion-matrix convention is `[[TN, FP], [FN, TP]]`, with rows denoting
the experimental non-contact/contact label and columns denoting exclusion from
or inclusion in the per-complex Top-10.

## Evidence boundary

- Checkpoint selection used development-validation contact MPC-AP only.
- The 20 instance identifiers were frozen before residue-pair labels were read.
- No model parameters, calibration values, thresholds, or sample identifiers
  were changed after label opening.
- Model inputs are ESM-2 embeddings and independently sequence-predicted monomer
  geometry. Experimental bound coordinates are not model inputs.
- Experimental contacts are used only as case-study measurement labels.
