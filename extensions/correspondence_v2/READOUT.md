# Post-freeze correspondence extension: completed development readout

All 9 GPU tasks completed, followed by a successful independent identity-parity check and complete-campaign aggregation. This is a new exploratory validation analysis, not a new blind test, a new trained model, or a benchmark-improvement claim.

## Cohort and computation

217 development-validation complexes, 68 existing authority components, three fixed training seeds for each of ESM-IF1, AntiBERTy and ProteinMPNN. Twelve conditions per task yielded 23,436 complex-condition records. All cases had positive contacts; no zero-positive cases were excluded. There are 864 small retrieval panels, each with 16 candidates. These are repeated measurements, not 23,436 independent biological samples.

## 1. Amino-acid-preserving corruption still damages contact localization

The following values are equal-complex macro pair AP, averaged over the same three fixed seeds. All arms retain the same base ESM-2 inputs; the encoder name identifies the appended feature source, not an encoder-only model. Each comparison is within an arm. AntiBERTy affects the antibody side only, while ESM-IF1 and ProteinMPNN cover both sides. Widths, affected sides and historical training runs differ; do not treat the table as a parameter-matched ranking of encoders or compare degradation magnitudes as a clean structural-versus-sequence contrast.

| Existing feature arm | Aligned identity | Same-AA shuffle | Chain shift +8 | Within-chain global shuffle | Known-inverse restore |
|---|---:|---:|---:|---:|---:|
| ESM-2 + ESM-IF1 | 0.027995 | 0.011100 | 0.009683 | 0.009399 | 0.027995 |
| ESM-2 + AntiBERTy | 0.021404 | 0.017459 | 0.016658 | 0.015796 | 0.021404 |
| ESM-2 + ProteinMPNN | 0.023018 | 0.014943 | 0.016553 | 0.015349 | 0.023018 |

Same-AA shuffling reassigns the extra feature vectors only among residues of the same amino-acid type within the same biological chain. ESM-2 features, presence flags, the feature multiset and VH/VL boundaries are preserved. The observed degradation therefore cannot be attributed solely to changing the amino-acid type assigned to an extra feature. It still does not isolate spatial neighbourhoods, prove a universal necessity theorem, or demonstrate a learned repair mechanism.

Primary descriptive contrasts below use a DIFFERENT, explicitly registered estimand: equal-cluster/equal-seed paired effects. Their estimates need not equal subtraction of the equal-complex table above. Intervals resample seeds and clusters, are pointwise, and are exploratory rather than multiplicity-corrected confirmatory tests.

| Same-AA minus identity | Equal-cluster AP difference | Pointwise crossed-bootstrap 95% interval |
|---|---:|---|
| ESM-2 + ESM-IF1 | -0.017151 | [-0.022688, -0.012434] |
| ESM-2 + AntiBERTy | -0.005721 | [-0.008787, -0.003089] |
| ESM-2 + ProteinMPNN | -0.008499 | [-0.011953, -0.005752] |

## 2. Contact quality and the current partner-ranking score can move in opposite directions

For the ESM-IF1 arm, within-chain global shuffling reduces mean contact AP from 0.027995 to 0.009399. In the bounded antigen-swap diagnostic, however, mean known-cognate MRR rises from 0.216643 to 0.331886; antibody-swap MRR rises from 0.176119 to 0.226915.

This is a warning against using this small-panel complex score as automatic evidence of interaction specificity. It is not proof that shuffled features are better, that the model cannot bind, or that all co-folding methods fail. Only 16 fixed anchors were used; the same anchors recur across three seeds, alternatives are unlabelled rather than experimentally confirmed non-binders, and these ranking means have no confirmatory interval. Candidate count, pooling function and conditions were fixed before this read.

## 3. Engineering and provenance checks

- 34 standalone extension contract tests passed; four additional synthetic identity-auditor checks passed.
- All 9 complete task results have zero identity-repeat and known-inverse-restore probability discrepancies.
- 27 comparisons to historical identity validation means passed a predeclared 1e-6 tolerance. Maximum observed absolute difference: 5.9274449341895075e-09.
- All 1,045 selected input files and pinned source/checkpoint digests were verified; model state remained unchanged.
- Smoke 1639862, array 1639865_0 through _8, and analysis 1639880 all completed with exit 0:0.
- CPU-only analysis was rejected by the partition before allocation; the original rejection is retained. The accepted analysis used the partition-required one-GPU allocation, although the statistical calculation itself is CPU work.

## Evidence and reproduction

`evidence/attempt1/ANALYSIS.json` contains all 144 predeclared metric-condition contrasts. `PER_TASK_METRICS.csv` has 108 per-task condition summaries. `full/task_*/RESULT.json.gz` preserves every original GPU result, with original and compressed SHA-256 digests in `FULL_RESULT_ARCHIVE_MANIFEST.json`. Decompress into a new working directory before running the unchanged aggregator. No checkpoints, embeddings or raw structure files are redistributed here.

`PLAN.json`, `IDENTITY_PARITY.json`, `SACCT_COMPLETED.txt`, `UNIT_TESTS.txt`, and `EXECUTION_COMPLETE.json` record the inputs, correctness checks and terminal scheduler evidence.

## What remains unfinished

No SaProt/ProSST extraction or new training, learned realignment, layer sweep, new untouched OOD cohort, large all-vs-all benchmark, RePaRank replay, co-folder evaluation or affinity experiment was executed in this extension. Existing PCGrad/GradNorm and old corruption-ladder work are not counted as new. The frozen paper evidence and permanently closed tests were not reopened.

The immediate scientific question exposed by these measurements is why the existing partner score can improve as contact localization deteriorates. Any expanded diagnostic or scoring intervention needs a new declared development protocol; this read does not authorize further reuse of closed test data.
