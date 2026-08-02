# Live development audit — 2026-08-01T11:35Z

This immutable local copy mirrors the read-only remote audit at
`/ssd/scxi253/model_deployment/data/framecontact_20260730/runs/live-audits/20260801T113500Z`.
It contains only in-flight train/validation log summaries and Slurm state. No
checkpoint, frozen-test prediction, external-test artifact, or test label was
loaded.

At the snapshot, all five unary-only seeds had reached epoch 18. All five
distogram seeds had reached epoch 15 and one had reached epoch 16. The matched
comparison therefore uses epoch 15. The two comparisons change both pair
factorization and residue aggregation and are not a causal geometry or
distogram promotion test.

The decisive binary-contact versus distogram comparison requires the queued
`framecontact_contact_hybrid` condition. The current arrays use the older v3.1
development split and cannot substitute for double-cold retraining.

SHA-256 digests of the copied remote files are:

- `DISTOGRAM_COMPACT.json`: `1a3f60c5f4ebecd6b06008f61ddc38a759d4fbe0a31ddaf4f098ee5fcf66c23d`
- `DISTOGRAM_PATIENCE.json`: `e29984cdbd10fe457880d6030ee3d2a8641048a3dc7083243285bad39872254b`
- `MATCHED_EPOCH_UNARY_VS_DISTOGRAM.json`: `2fe699b1590d7ca4ab459b1caee978387bbfe72aaf40833d239dd3b4f995fa86`
- `RESIDUE_HEAD_COMPACT.json`: `396f85b87c6bcddf55cf024d54273cd03c8014534075927459fb01bf6675efa4`
- `RESIDUE_HEAD_PATIENCE.json`: `a697f76a52f452dca5021a24c7166d85f15704c698d5d80eab768d9060148e25`
- `SQUEUE.txt`: `f83bde4c88fb97d96c50d775541d6fd3b44ccde36312c150eef99b1afa5b975e`
