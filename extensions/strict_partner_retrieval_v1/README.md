# strict_partner_retrieval_v1

Fail-closed implementation support for the strict dual-axis partner-retrieval experiment.

- `h0.py`: no-training k-mer and NW homology-transfer controls.
- `metrics.py`: finite-library MRR/R@K, exact chance, grouped and crossed uncertainty.
- `authority.py`: refuses execution until both split builder and independent audit authorize training.
- `contract_models.py`: independent L1 CDR late interaction and parameter-identical R1/R2 matched-budget contract wrapper.
- `models.py`: original split/training-line model interfaces. These are separate experimental implementations, not interchangeable checkpoints.

The canonical machine contract is `protocols/strict_dual_axis_partner_retrieval_v1.json`; the human protocol is `protocols/STRICT_DUAL_AXIS_PARTNER_RETRIEVAL_V1.md`.

No current file authorizes execution before the still-running strict split solve and its independent audit produce a passing authority receipt.
