# Baseline loss/evaluation evidence handoff (2026-08-27)

This package recovers complete scalar training histories from authenticated
archived runs. It does not open or rescore raw held-out labels.

## Deliverables

Training curves and machine-readable histories:

```text
/ssd/scxi253/model_deployment/results/development/baseline_training_evidence_v2
```

- `epiformer/history.csv`: epochs 1--57, including total train/validation loss
  and the recorded loss components.
- `epiformer/loss_curves.svg` and `epitope_loss_components.svg`.
- `mipe/paper_appendix_lr1e4/`: five authenticated 800-epoch folds and their
  fold-mean curve.
- `mipe/released_code_lr1e3/`: five authenticated 800-epoch folds and their
  fold-mean curve. The existing MIPE held-out summary binds only to this track.
- `sppider_seq/`: train/validation curves for the 12 selected ensemble members
  (three seeds, two tasks, two members per task).
- `BASELINE_TRAINING_EVIDENCE_V1.json`: data-only provenance summary.
- `BASELINE_TRAINING_EVIDENCE_V1_RECEIPT.json`: 53 artifact hashes.

Result SHA-256:

```text
b95afeeda288f17fcf8a8c6cd78f3fe7d1fb786138928fcb5046a2649ac6968d
```

Receipt SHA-256:

```text
eaab67258cb1b85dfe87bea58906f990db1f12fa0322d143128451431dc4bd3c
```

All 53 receipt entries were re-hashed successfully and all 26 SVG files were
parsed as valid XML.

Archived evaluation-summary bindings:

```text
/ssd/scxi253/model_deployment/results/development/baseline_archived_evaluation_bindings_v2
```

Result SHA-256:

```text
731f4653cab32d23d0ba5cfda2fa9252ed802868f94d9d17d21ce601f68e2e2a
```

This binding explicitly records each method's split identity, aggregation,
threshold policy, and source-summary hash. It authorizes per-method archived
reporting, but not a single same-protocol comparison table: the archived
EpiFormer, MIPE, and SPPIDER-seq evaluations use different instances/splits,
aggregation rules, and threshold policies.

## DiscoTope-3.0

The official source and fitted model ensemble are bound at:

```text
/ssd/scxi253/model_deployment/results/development/discotope3_official_binding_v1
```

- official repository: `https://github.com/Magnushhoie/DiscoTope-3.0`
- revision: `35d9f2e55f97eaba2a7acefbc394db58fb9670bc`
- source archive SHA-256:
  `bb7295c76d2b231ea04375c961fc746a938743cf1d6fc293c5f6c975475c46f3`
- `models.zip` SHA-256:
  `fa627cd66cb9d3fc7892eabd5962b324ffbfa3b68a54718aae34cc3e839b4f4d`
- exact XGBoost member count: 100
- binding result SHA-256:
  `aa41aee9e583b51983ea567f18675bed09e0d36daf4839fc0fa83d347253914d`

DiscoTope-3.0 is a released fitted XGBoost ensemble rather than an archived
neural-training run, so a neural per-epoch train/validation loss curve is not
applicable. The appropriate reproducibility evidence is the exact source
revision, fitted-model hash/member count, inference output, and evaluation
receipt. No loss trajectory has been invented.

## Reporting rules

1. Do not relabel archived metrics as a unified same-protocol benchmark.
2. SPPIDER-seq F1/MCC in the archived final summary use validation-selected
   thresholds, not a universal fixed threshold of 0.5.
3. The MIPE paper-hyperparameter (`lr=1e-4`) track has no bound held-out result;
   the existing held-out result binds only to the released-code-compatible
   (`lr=1e-3`) track.
4. The EpiFormer curve is its archived internal epitope-group training run; it
   is not the current FrameContact 2,565/549 development protocol.
5. This recovery opened no old SAbDab2 test or temporal holdout and performed no
   model or threshold selection.
