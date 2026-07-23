# Independent audit: train-only asymmetric retrieval architecture screen

## Verdict

**PASS with provenance caveats.** All nine fold runs are internally valid, and the predeclared architecture-selection rule selects `antiberty_calm_mean`. The caveats weaken bitwise launch-source archival, but they do not invalidate the independently recomputed metrics, fold macro-average, or selected winner.

This audit covers only train-source, antigen-isolated model selection. It did not evaluate the fixed development holdout or the frozen test set, and it is not evidence of double-remote antibody generalization.

## Architecture selection

| Rank | Variant | Fold MRRs | Macro MRR | Exact random macro MRR | Multiple | Worst fold | Directional gate |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `antiberty_calm_mean` | 0.071493, 0.056428, 0.056192 | **0.061371** | 0.019822 | **3.096×** | **0.056192** | PASS |
| 2 | `antiberty_chain_balanced` | 0.071304, 0.054725, 0.055914 | 0.060648 | 0.019822 | 3.060× | 0.054725 | PASS |
| 3 | `esm2_calm_mean` | 0.063746, 0.029457, 0.042574 | 0.045259 | 0.019822 | 2.283× | 0.029457 | FAIL |

The strict and chain-balanced AntiBERTy variants both pass the early predeclared random-baseline gates. The ESM-2 control fails because fold 1 antibody-to-antigen MRR is 0.017929 versus an exact random expectation of 0.022520. The difference between the two AntiBERTy variants is small (macro MRR 0.000723) even though strict mean is higher in all three folds; this screen selects the winner but does not establish a statistically meaningful superiority claim between those two variants.

## Independent checks

- All 9/9 Slurm tasks finished `COMPLETED` with exit code `0:0`.
- Every saved best checkpoint SHA-256 matches its training summary.
- Every checkpoint, history, summary, and provenance object agrees on its canonical configuration SHA-256.
- Every history starts at epoch 1 and is contiguous, has no duplicate epoch, and exactly matches the history embedded in `last.pt`.
- All nine selected epochs are also the absolute raw maximum validation MRR epochs; no `min_delta` ambiguity occurred.
- Saved directional MRR and R@1/R@5/R@10/R@20 values were recomputed from per-anchor ranks.
- Exact multi-positive random expectations were recomputed per anchor using its native candidate count and known-positive count:
  - `P(best rank = r) = C(N-r, m-1) / C(N, m)`
  - `P(best rank <= k) = 1 - C(N-k, m) / C(N, m)`
- The random-baseline implementation was separately brute-force checked for all 36 `(N,m)` cases with `N <= 8`.
- No non-finite numeric value was found in any history.
- The 18 hashed Slurm stdout/stderr logs contain no matched `NaN`, `Inf`, OOM, traceback, exception, killed-process, segmentation-fault, or non-zero-status evidence.
- Output directories are unique. There is no structural evidence of resume contamination.
- Every run authorizes only source `train`, reports zero intersection with sealed source `validation` and `test`, and declares `frozen_test_used=false`.

## Hash and launch evidence

- Audit JSON: `91d94092fafbc5fd4110e52de8f75712638b114371fe6468a08af8e963c62a3e`
- Audit program: `81c559c7621702225d99780548a1624b1d4e693e13bdccd9045e4c8882beddf2`
- Locked protocol JSON: `602a5fe74ba3797d19027d8893711fc530cab16aadca740d36f1cc98a16b599e`
- Fold index: `feca6a1ec6761aa76baae847a1932fbf910568466dee7a34f133df27566fbc2d`
- Authority split: `e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b`
- Submission wrapper: `b52f88823c0896e823c8211e8ed46d22d15fdccffb119502e23475ee5d0c109d`
- Pre-training Needleman-Wunsch fold audit: `00ccd7dd3220673eec3b7938c05e04160a1415d8bf7a07124a81fadf47c3cc4d`
- Run-time entrypoint hash embedded by all nine checkpoints: `ee3707f61525a9abbb1d451e71fa0b4a07847b0b9915d5a44d1870c673ce61dc`

The machine-readable audit contains every checkpoint/history/summary/provenance hash, all 18 Slurm-log hashes, and SHA-256 hashes of the three distinct Slurm submit lines. Each submit line pins the reviewed fold-index hash. The reviewed submission wrapper independently recomputes and rejects a mismatched index hash, resolves an index entry, and passes that entry's fold-manifest hash to the training entrypoint.

## Provenance caveats

1. The protocol JSON hash is not checkpoint-native. Each run is an exact semantic match to the reviewed protocol, but the protocol-manifest SHA-256 itself was not written into the checkpoints.
2. The fold-index hash is present in Slurm submit-line evidence and enforced by the wrapper, but is not checkpoint-native. Each individual fold-manifest SHA-256 is checkpoint-native and was verified against the physical file and reviewed index.
3. The current remote entrypoint was updated after these runs. All nine checkpoints carry one common run-time source hash set, but the old entrypoint bytes are no longer at the live source path. This is source-archival drift, not metric or checkpoint contamination.
4. Model-weight digests match the reviewed locks and artifact metadata enforced during embedding loads. The audit does not reinterpret the projects' custom model-weight hashing procedures.

These caveats do not change the architecture-selection result. They should be repaired before publication-grade final evaluation by embedding the protocol SHA, fold-index SHA, submission-wrapper SHA, and an immutable source snapshot or commit identifier directly in future checkpoints.
