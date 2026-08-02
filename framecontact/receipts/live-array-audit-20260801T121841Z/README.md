# Live development audit — 2026-08-01T12:18:41Z

This immutable local copy mirrors the read-only remote audit at
`/ssd/scxi253/model_deployment/data/framecontact_20260730/runs/live-audits/20260801T121841Z`.
It contains only in-flight train/validation log summaries and Slurm state. No
checkpoint, frozen-test prediction, external-test artifact, or test label was
loaded.

At this snapshot, all five unary-only seeds had reached epoch 19. Four of five
distogram seeds had reached epoch 16 and one remained at epoch 15, so the latest
matched comparison is still epoch 15 and is byte-identical to the previous
comparison. Contact-marginal, contact-hybrid, and distogram-triangle conditions
had not started because both arrays were still at their concurrency limit.

The epoch-15 comparison changes both pair factorization and residue aggregation,
so it is not a causal distogram promotion test. The decisive binary-contact versus
distogram comparison still requires the queued contact-hybrid condition. These
arrays use the older v3.1 development split and cannot substitute for double-cold
retraining or any frozen/external result.

SHA-256 digests of the copied remote files are:

- `DISTOGRAM_COMPACT.json`: `8c3a814d64086653f8d9c4e9187df35978773b9fb74bce83a2b9b5b145013939`
- `DISTOGRAM_PATIENCE.json`: `0bfb2a0e8c5adccd1cbb6c12451384cd4070143feb2eb0ca27a9acbab7f1b29e`
- `MATCHED_EPOCH_UNARY_VS_DISTOGRAM.json`: `2fe699b1590d7ca4ab459b1caee978387bbfe72aaf40833d239dd3b4f995fa86`
- `RESIDUE_HEAD_COMPACT.json`: `dc0515aef319f472c5917f56128107044820852fbc53be1f8553d182c76843ae`
- `RESIDUE_HEAD_PATIENCE.json`: `e64cc1ebf7499bc8b980205e2be5d5e76c71f24b8bc0a885d88be784bbbc939c`
- `SQUEUE.txt`: `51159e5c8a2edfb36343164822e11debfac3ea32e807ba298fff6d505005f0af`
