# Partner-negative ablation — running development study

Status at `2026-08-21T18:17:16+08:00`: ten no-partner-negative jobs are
running concurrently on `gpu_4090`, covering seeds 17, 42, 73, 101, and 137
for both AsEP and SAbDab2. No final comparison is reported yet.

The matched comparison is:

- negative arm: standard wrong-partner specificity supervision with one
  negative and specificity loss weight 0.25;
- no-negative arm: the same model, split, seed, optimizer, learning rate,
  epoch limit, early stopping, partner panels, calibration path, and wrong-
  partner execution path, but with specificity loss weight exactly 0.0.

The study intentionally does not compare the arms' total training objectives,
because the objectives contain different terms. Curves are defined on shared
quantities: train contact BCE, train cognate objective
`contact + 0.25 * epitope + 0.25 * paratope`, and fixed raw-logit validation
contact Brier. Final development validation also reports pair contact AUPRC,
AUROC, ranking metrics, calibration, and five-seed paired bootstrap intervals.

The frozen AsEP protocol SHA-256 is
`45e3b1ccbdb22191f1c27f04c0a2abc41e6459336381200ae0cac540a9bc46bb`;
its receipt SHA-256 is
`1f630ce2d92aea5f642ecbd5656686502952e5dc9ba1821a05a23a5433fab9d0`.
The frozen SAbDab2 protocol SHA-256 is
`19b9138bbdfa75f65099579cc7e94b2ed73369897cebc3e8bc8f56c720de952f`;
its receipt SHA-256 is
`742a2a4c3f2c3ea1965ee099fa0a1eb5cdd7391b2559c19194409fe5e4b407bb`.

At the recorded snapshot, all five AsEP seeds had completed six or seven
epochs with no stderr. SAbDab2 was still within its first long epoch. This is
expected from the measured dataset runtime and is not interpreted as failure.

Test and external-test roles remain sealed. They may be opened only after the
five-seed development/validation comparison is complete, audited, and accepted
for progression.
