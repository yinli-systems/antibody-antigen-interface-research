# FrameContact live development audit at 2026-08-01T14:51:57Z

This is a read-only snapshot of the two immutable Slurm arrays on ParaCloud.
The runs use RTX 4090 workers. No job was modified, restarted, promoted, or
selected by this audit.

## Coverage

- `framecontact_unary_only`: synchronized through epoch 21; individual seeds
  through epochs 21--22.
- `framecontact_distogram`: synchronized through epoch 18; seed 17 through
  epoch 19.
- `framecontact_contact_marginal`, `framecontact_contact_hybrid`, and
  `framecontact_distogram_triangle`: not started because each array has a
  five-task concurrency limit.
- Slurm jobs `1345042` and `1345108` were running; queued tasks reported
  `JobArrayTaskLimit` rather than an error.

## Matched epoch 18: distogram minus unary-only

All values are five-seed development-validation means. Exact two-sided
sign-flip p-values have coarse resolution (`0.0625` is the smallest possible
at five seeds) and are descriptive only.

| Metric | Unary-only | Distogram | Paired delta | Positive seeds | Exact p |
|---|---:|---:|---:|---:|---:|
| Pair contact AUPRC | 0.010457 | 0.010897 | +0.000441 | 3/5 | 0.5625 |
| Fixed-logit precision | 0.012025 | 0.014661 | +0.002636 | 5/5 | 0.0625 |
| Fixed-logit recall | 0.307414 | 0.229110 | -0.078304 | 0/5 | 0.0625 |
| Fixed-logit F1 | 0.023078 | 0.027508 | +0.004430 | 5/5 | 0.0625 |
| Epitope AUPRC | 0.121149 | 0.125032 | +0.003883 | 3/5 | 0.4375 |
| Paratope AUPRC | 0.689280 | 0.659057 | -0.030223 | 1/5 | 0.1250 |
| Worst-direction partner margin | -0.002432 | -0.001053 | +0.001379 | 3/5 | 0.9375 |

The distogram predicts only 1.72% of candidate pairs as positive versus 2.85%
for unary-only. Its recall is 74.5% of unary-only recall at the same fixed raw
logit threshold. This is evidence of a precision--recall/sparsity shift, not a
stable ranking or specificity improvement.

Across its own epochs 1--18, distogram precision rose from 0.008667 to
0.014661 while recall fell from 0.838175 to 0.229110 and pooled contact AUPRC
fell from 0.014842 to 0.010897. The current training dynamics therefore do not
justify choosing the latest epoch.

## Evidence boundary

- Source split: older antigen-fold v3.1 development train/validation roles.
- This is not the new strict double-cold split.
- No SAbDab2 frozen-test prediction or metric was loaded.
- No AsEP official-test prediction or metric was loaded.
- No benchmark, external-generalization, A100, or quality-improvement claim is
  authorized.
- The preregistered triangle promotion test is **triangle versus distogram**;
  it cannot run until the queued triangle condition completes. The 74.5%
  recall ratio above is distogram versus unary-only and must not be substituted
  for that gate.

`SHA256SUMS` uses the original remote absolute paths. The copied local files
were rehashed by basename and all five JSON digests matched exactly.

## Files

- `residue-head-full.json` and `distogram-full.json`: full five-seed epoch
  trajectories after metric extraction; no per-complex predictions.
- `residue-head-compact.json` and `distogram-compact.json`: bounded monitoring
  views with descriptive slopes.
- `distogram-minus-unary.json`: paired-seed matched-epoch comparison.
- `render_trajectory.py`: deterministic renderer for the included figure.
- `matched-development-trajectories.png`: mean plus/minus sample-SD trajectories.
