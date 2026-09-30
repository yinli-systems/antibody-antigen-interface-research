# EPIC reviewer-concern closure package

This directory is the auditable camera-ready response package for the accepted AI4DD paper **EPIC: Epitope–Paratope Interleaved Contact Learning for Antibody–Antigen Interface Prediction**.

## Start here

- [`REVIEWER_RESPONSE.md`](REVIEWER_RESPONSE.md) — concern-by-concern response.
- [`CAMERA_READY_PATCHES.md`](CAMERA_READY_PATCHES.md) — exact textual replacements for the unavailable manuscript source.
- [`SUPPLEMENTARY_EVIDENCE.md`](SUPPLEMENTARY_EVIDENCE.md) — additional results and evidence boundaries.
- [`AUTHOR_RESPONSE_SHORT.md`](AUTHOR_RESPONSE_SHORT.md) — concise response suitable for a camera-ready note.
- [`BASELINE_PROTOCOL_TABLE.md`](BASELINE_PROTOCOL_TABLE.md) — corrected modality/comparability table.
- [`CLAIM_BOUNDARIES.md`](CLAIM_BOUNDARIES.md) — claims that are and are not authorized.
- [`EVIDENCE_MATRIX.json`](EVIDENCE_MATRIX.json) — machine-readable concern matrix.
- [`REVIEWER_CONCERN_CLOSURE.json`](REVIEWER_CONCERN_CLOSURE.json) — all 19 dispositions and exact empirical boundaries.
- [`EXACT_SUBMISSION_AUTHORITY_AUDIT.json`](EXACT_SUBMISSION_AUTHORITY_AUDIT.json) — what exact submission artifacts were and were not found.
- [`OPEN_ITEMS.md`](OPEN_ITEMS.md) — remaining empirical boundaries, not unanswered reviewer points.

## Reproducible artifacts

- `evidence/rank_calibration_v2/ANALYSIS.json` — official-455 post-submission practical ranking/calibration analysis.
- `evidence/pair_decoder_v1/ANALYSIS.json` — three-seed full-pair versus marginal/unary evidence.
- `evidence/auxiliary_loss_screen/` — frozen count/proportion-loss screen.
- `evidence/icg_successor_v1/` — compact successor contact-gate audit, receipts, and preserved M2 timeout provenance.
- `evidence/original455_authority/` — protocol/build/split audit receipts.
- `scripts/` — exact analysis scripts.

## Boundary

The original LaTeX source for the uploaded anonymous EPIC PDF was not present in the connected repository or GPU snapshot. Accordingly, this branch provides exact camera-ready patches and evidence rather than pretending that the PDF itself was a safely editable source. No new model training is hidden in the package; practical ranking and decoder analyses reuse completed raw outputs.
