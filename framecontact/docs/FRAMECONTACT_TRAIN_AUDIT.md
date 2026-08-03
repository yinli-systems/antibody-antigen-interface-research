# FrameContact train-fit audit

This audit answers a narrow question raised during development: did a model
fail because it could not fit the training complexes, or because what it fit did
not transfer across antigen folds?

It is deliberately post-selection. The evaluator accepts only a completed
development run whose receipt says both test roles remain unopened. It loads the
checkpoint, temperatures, and operating thresholds already selected on
validation, then evaluates the complete authenticated `train` role. Train
metrics never select a checkpoint or fit calibration.

```bash
python scripts/evaluate_framecontact_train_audit.py \
  --run-root /path/to/completed/development/run \
  --structure-cache-root /path/to/structure-cache \
  --structure-manifest-sha256 SHA256 \
  --contact-cache-root /path/to/contact-cache \
  --contact-manifest-sha256 SHA256 \
  --embedding-root /path/to/esm-cache \
  --embedding-manifest-sha256 SHA256 \
  --label-root /path/to/train-labels \
  --split-manifest /path/to/antigen-fold-v3.csv \
  --split-manifest-sha256 SHA256 \
  --training-partner-panel /path/to/train-reciprocal-panel.json \
  --training-partner-panel-sha256 SHA256 \
  --output-root /new/write-once/train-audit \
  --device cuda --precision bf16
```

For each seed and condition, compare train and selected-checkpoint validation
using the same metric definition. The diagnostic interpretation is:

- high train and low validation: memorization/generalization or regularization
  bottleneck;
- low train and low validation: optimization, representation, supervision, or
  class-imbalance bottleneck;
- low partner margin on both roles: the model has not learned partner
  specificity even on observed antigen folds;
- high train partner margin but low validation margin: partner-specific
  interactions are memorized but do not transfer across antigen folds.

The authority is
[`framecontact_train_audit_v1.json`](../protocols/framecontact_train_audit_v1.json).
This audit does not open, replace, or make claims about a frozen/external test.
