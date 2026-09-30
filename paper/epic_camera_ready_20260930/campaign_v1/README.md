# EPIC reviewer-closure campaign v1

This directory preregisters and executes the matched multi-seed ablation campaign requested by the AI4DD reviewers. The campaign is deliberately bounded: it uses the frozen post-submission official-455 SAbDab2 authority with independent sequence-predicted AF3 monomer geometry. It is **not** the unavailable submitted OpenDDE checkpoint, and the official 455 split has historical exposure, so these runs are post-submission diagnostic evidence rather than fresh blind evidence.

## Frozen authority

- Campaign protocol SHA-256: `d6647c4421eebb910d8adae73f41c814690e50f5b41ea1d6264179dcea7f0bc7`
- Parent `BUNDLE.json` SHA-256: `0312e79e306e42c087cd653ac79a7a14459691dab0376bc33f1fdbe1275b55a4`
- Full parent verification receipt SHA-256: `22758cac5289c70393711a6e17456ee8747aa5341ecdb8384c5d2372e7f79130`
- Verified parent inventory: 6,597 files / 123,904,952 bytes
- Split sizes: 2,887 train / 321 validation / 455 held-out test
- Seeds: 17, 29, 43
- GPU: one RTX 4090 per condition/seed
- Test policy: no campaign test scoring until every condition/seed checkpoint and validation receipt is frozen.

## Reviewer concerns covered

The 12-condition matrix isolates the full learned contact-gated reference against: content-only cross-attention, no cross-attention, no geometry branch, unary-only pair scoring, rank-1 pair scoring, no contact-derived residue readout, contact-only supervision, residue-only supervision, no epitope auxiliary loss, Dice-only loss, and proportion-only loss. This directly covers the missing geometry, contact-gating, cross-attention, low-rank decoder, shared contact-to-residue coupling, joint-supervision, and epitope-auxiliary ablations. The resulting evidence will report three-seed deltas, per-complex metrics, fixed-threshold precision/recall, calibration, ranked contact cutoffs, parameter counts, and wall-clock.

The campaign does **not** claim to solve the exact OpenDDE-vs-alternative-predictor comparison because no hash-bound OpenDDE authority or submitted checkpoint is available. Existing predicted-vs-bound and no-geometry controls remain sensitivity evidence, not an exact predictor substitution.

## Execution ledger

- Full authority audit: complete and passed.
- Condition smoke array: Slurm `1643888`, 12/12 complete, all stderr files empty.
- Frozen smoke manifest SHA-256: `f9eb682edd0dbff8559b71df708b8f3d31989329ed199db9a4839c8b15f92534`.
- Main training array: Slurm `1643900`, 36 tasks (`12 conditions × 3 seeds`), submitted and running.
- ParaCloud root: `/ssd/scxi253/frontier_multimodel_v1/epic_reviewer_closure_v1_20260930T220257Z`.

`parent_*_reference.py` and `PARENT_PROTOCOL.json` are small frozen provenance snapshots. Executable authority is the parent ParaCloud bundle identified in `CAMPAIGN_PROTOCOL.json`; the snapshots are not used as runtime substitutes.
## Frozen post-training analysis plan

Before any full 20-epoch result existed, `ANALYSIS_PROTOCOL.json` was frozen with SHA-256 `163245a45aee020b71c9f995a997ba7e08913286a95d8b10d7cf8a2a7c581012`. It preregisters paired three-seed deltas, crossed seed×complex bootstrap intervals, contact ranking budgets, calibration, and strict interpretation boundaries. The authority-freeze and aggregation pipeline passed a six-run synthetic integration test (`SYNTHETIC_PIPELINE_TEST_RECEIPT.json`, SHA-256 `5a5e4bb6d2fa52504d8a840ec7af0f6aa074d64184854403c87a65fdfc35e167`).

Slurm job `1643939` is queued with `afterok:1643900`. It will verify and hash every checkpoint/result/validation/log, freeze all 36 runs into a training manifest, and only then compute the validation analysis. It does not open or score the 455-complex test authority.
