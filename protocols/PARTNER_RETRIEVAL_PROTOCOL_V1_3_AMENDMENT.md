# Partner-specific retrieval protocol v1.3 amendment

Locked: 2026-07-23, after train-only fold selection and before either fixed
development boundary was scored with the selected model.

This amendment fixes the final-refit rule and records the selection outcome permitted
by v1. Earlier protocol files remain immutable.

## Train-only architecture selection

The three preregistered variants were evaluated on the same three exact-NW-audited,
antigen-remote folds. The locked selection statistic was the unweighted mean of fold
bidirectional MRR, followed by worst-fold MRR and then trainable parameter count.

| Variant | Fold MRRs | Macro mean | Worst fold | Mean exact-random MRR |
|---|---:|---:|---:|---:|
| AntiBERTy + ESM-2, strict length mean | 0.071493, 0.056428, 0.056192 | 0.061371 | 0.056192 | 0.019822 |
| AntiBERTy + ESM-2, chain-balanced mean | 0.071304, 0.054725, 0.055914 | 0.060648 | 0.054725 | 0.019822 |
| ESM-2 + ESM-2, strict length mean | 0.063746, 0.029457, 0.042574 | 0.045259 | 0.029457 | 0.019822 |

The selected architecture is **AntiBERTy + ESM-2 with strict projection-before-
length-mean pooling**. The small difference from chain-balanced pooling is not treated
as evidence that strict mean is intrinsically superior; it is only the deterministic
outcome of the preregistered rule.

The selected architecture's fold best epochs were 58, 60, and 44. The fixed refit
epoch count is their median: **58 epochs**.

## Fixed-boundary firewall

The final seed-42 refit for each development boundary must:

- train for exactly 58 epochs with the already locked optimizer, scheduler, batch size,
  architecture, temperature, and seed;
- never compute validation metrics, early-stop on the fixed holdout, or select among
  epoch checkpoints;
- load pair rows and frozen embeddings only for the training membership during refit;
- save one deterministic final epoch checkpoint plus provenance; and
- score the fixed validation membership once, after refit completion, using the
  evaluation command that also writes the full score artifact for statistical tests.

Any run that evaluates the fixed holdout during training is invalid and must not be
reported. The four-view and CALM-comparable boundaries are separate refits because
their training memberships differ.

## Refreshed explicit split schema

The CALM-comparable split was deterministically regenerated with identical 3,200/245
membership and an explicit `concat_antibody_and_antigen` constraint declaration:

- split SHA-256:
  `c586fbd1ec83effe1208746d64950feddef280a95a938cf241293dda0b396b64`
- scientific payload SHA-256:
  `70b63488c1bebb2a01ef3807539d916effeeda961ee21012660737e2442d600f`
- independent verifier SHA-256:
  `73464041aa6aebd1733259a9e662f44bee1f512ed18338959738c0f25c2b1d4d`

The four-view stress-test split remains:

- split SHA-256:
  `5d92edd03e3b96edfa727823b2a5bbdf49e80614ef759fe0c6b14cf6c2b6b92b`
- independent verifier SHA-256:
  `f724e7c8cb99a7f79d76241df077539474977d2946a7c714c956dbc42d822c32`

Both evaluation commands must receive the expected constraint mode explicitly and
reject a mismatch. The frozen SAbDab2 test remains sealed.
