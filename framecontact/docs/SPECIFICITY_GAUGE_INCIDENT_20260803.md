# FrameContact specificity-gauge repair and execution receipt

Date: 2026-08-03

## Scope and evidence boundary

This receipt covers development-only SAbDab2 antigen-fold train/validation
work. No frozen SAbDab2 test prediction, external test prediction, checkpoint
choice, calibration choice, or benchmark claim was made from this incident.

## Incident 1: invalid centered noisy-OR

- Slurm array: `1349055`.
- Observed tasks: 0--4, the five `framecontact_gauge_marginal` seeds.
- Runtime before cancellation: approximately 10 h 38 min per task.
- Symptom: all five seeds returned exactly the same validation epitope and
  paratope AUPRC despite non-identical contact metrics.
- Root cause: a doubly centered interaction residual has zero meaning “no
  pair-specific deviation,” not Bernoulli contact probability 0.5. Applying
  noisy-OR over hundreds of such values reaches its +30 numerical clamp and
  removes all residue ranking information.
- Reproduction: on a centered `220 x 512` random map, the old pooling produced
  one unique residue logit; centered log-mean-exp retained 479 unique finite
  values in the ParaCloud training runtime.
- Disposition: job `1349055` was cancelled. Its contact-side traces remain
  failed-run diagnostics; its residue metrics are ineligible for selection.

The repair keeps noisy-OR for raw calibrated contact probabilities, but uses
masked log-mean-exp for a zero-centered interaction residual. Evaluation now
records, per complex, the exact unique-logit count, raw-logit standard
deviation, and the fraction at `|logit| >= 29`.

## Incident 2: incomplete frozen source and authority schema

The first v2 upload omitted the `slurm/` directory. Remote execution of the
same targeted tests therefore produced 65 passes and three missing-file
errors. No Slurm job was submitted from that source tree. A second immutable
source snapshot included `scripts`, `src`, `protocols`, `tests`, and `slurm`;
68 targeted tests then passed remotely.

Slurm job `1349889` exposed a separate fail-closed protocol error before
dataset construction or an optimizer step: v2 flattened the reciprocal panel
size and hashes, whereas the trainer requires the authenticated
`data.reciprocal_partner_panel` object. The array was cancelled and preserved
as execution-contract failure evidence; it produced no valid epoch or
validation prediction.

Protocol v3 restores the already-frozen panel schema and the complete
implementation and condition-specific loss contracts. It makes no scientific
factor change relative to the repaired v2 hypothesis.

## V3 pre-submission evidence

- Local full suite: 361 passed, 5 optional Boltz-runtime skips.
- Remote targeted suite: 73 passed.
- Remote source manifest:
  `0e33aafed2be4babf0cb2f14247848d88402b2c89aafdd0a6582d59e9545766d`.
- Remote tree digest:
  `d4c6a363a2f60cf27c95de8f9d4f6700040cf863970eb9217277ac0752e86482`.
- Protocol receipt digest:
  `c8002d36b848542d538c3221e17abf2763a7f86248f2e7e3e214bdcd911e1246`.
- Authority preflight: valid; 2,565 train and 549 validation complexes,
  reciprocal-panel v2 on both roles, 2,096,587 model parameters, no output
  directory created, and no test role loaded.
- Slurm test-only preflight: accepted.
- Submitted development array: `1349969`, three conditions by five seeds,
  at most five concurrent RTX 4090 tasks.
- Initial allocation: tasks 0--4 running; all five stderr files empty.

## Promotion boundary

The v3 result is not promoted merely because training succeeds. Promotion
still requires five-seed non-degenerate residue logits, preservation of
contact/epitope/paratope ranking and calibration, and an improvement in the
minimum reciprocal partner margin beyond measured epoch noise. Frozen and
external tests remain closed until every development choice is frozen.
