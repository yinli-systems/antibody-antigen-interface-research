# Live development audit — 2026-08-01T13:40:59Z

This immutable local copy mirrors the read-only remote audit at
`/ssd/scxi253/model_deployment/data/framecontact_20260730/runs/live-audits/20260801T134059Z`.
It contains only in-flight train/validation log summaries and Slurm state. No
checkpoint, frozen-test prediction, external-test artifact, or test label was
loaded.

All five unary-only seeds reached synchronized epoch 20; seed 73 had written
epoch 21. All five distogram seeds reached synchronized epoch 17; seed 17 had
written epoch 18. Contact-marginal, contact-hybrid, and distogram-triangle had
not started because both arrays remained at their five-task concurrency limit.

At matched epoch 17, distogram minus unary-only was mixed and fails any promotion
interpretation:

- worst-direction partner margin: `-0.005354`, positive in `1/5` seeds,
  exact two-sided sign-flip `p=0.375`;
- pooled contact AUPRC: `+0.000304`, positive in `4/5`, `p=0.75`;
- fixed raw-logit-zero contact precision: `+0.001132`, positive in `4/5`,
  `p=0.125`;
- fixed raw-logit-zero contact recall: `-0.034781`, positive in `2/5`,
  `p=0.3125`;
- pooled epitope AUPRC: `+0.004884`, positive in `3/5`, `p=0.4375`;
- epitope precision/recall: `+0.015445` / `-0.087582`;
- pooled paratope AUPRC: `-0.035300`, worse in `5/5`, at the minimum attainable
  two-sided five-seed sign-flip `p=0.0625`;
- paratope precision/recall: `-0.032857` / `-0.033219`.

The earlier epoch-15 positive worst-direction margin did not persist at epoch
17. This is useful negative evidence of trajectory instability, not a reason to
select epoch 15: both runs remain unfinished and checkpoint selection is frozen
to the registered validation rule. The comparison also changes pair
factorization and residue aggregation simultaneously, so the queued
contact-hybrid condition is still required for a causal binary-contact versus
distogram comparison.

These arrays use the older v3.1 development split and cannot substitute for
double-cold retraining or any frozen/external result. Conditional patience times
in the JSON assume no future validation improvement and are not completion
guarantees.

SHA-256 digests of the copied remote files are:

- `DISTOGRAM_COMPACT.json`: `aab83fdcf232951ee6819f169cbb5fa32dba856c5379edb34d6f516da9c4de68`
- `DISTOGRAM_PATIENCE.json`: `2c4e8e3641629146205f931bd7e9e0682f7b1c30fb543b4f5bcd5e0df572a7e5`
- `MATCHED_EPOCH_UNARY_VS_DISTOGRAM.json`: `59835d471b6ca316afdbc673adf24cdb3543404937e5606b505ea027f7696b32`
- `RESIDUE_HEAD_COMPACT.json`: `0dc464da792bd479604e39b587387456a5f7ad22b9c77c378543a702236c715a`
- `RESIDUE_HEAD_PATIENCE.json`: `e309ede0d8b421512dca5e0e2bf78bfef514790928432e09df9804c3d9641ac4`
- `SQUEUE.txt`: `0794a162860d7cb93e035cdbc5a9d005c2d4dfe4d20d62c0d084b871a5951658`
