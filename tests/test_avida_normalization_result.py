"""Evidence tests for the audited AVIDa Tier-1 normalization result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_ROOT = (
    ROOT / "data_registry/normalization_receipts/avida_hil6_2024_11"
)


def _read(name: str) -> tuple[dict[str, object], str]:
    raw = (RECEIPT_ROOT / name).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def test_normalization_receipt_binds_all_outputs_and_accounting() -> None:
    receipt, digest = _read("normalization_receipt.json")
    assert digest == "aee48d4cfa6ebf07e95326ddc06833747913e0260ff45abbfef1d973bffbf273"
    assert receipt["status"] == "normalized_tier1_with_explicit_quarantine"
    assert receipt["pairbench_training_eligible"] is True
    assert receipt["counts"] == {
        "admitted_antibodies": 37_200,
        "admitted_antigens": 31,
        "admitted_binders": 20_275,
        "admitted_nonbinders": 535_666,
        "admitted_pairs": 555_941,
        "quarantined_antibodies": 1_399,
        "quarantined_binders": 705,
        "quarantined_nonbinders": 17_245,
        "quarantined_pairs": 17_950,
    }
    assert {row["path"] for row in receipt["outputs"]} == {
        "AUDIT_SUMMARY.json",
        "assays.parquet",
        "duplicate_groups.parquet",
        "entities_antibody.parquet",
        "entities_antigen.parquet",
        "entity_rekey.parquet",
        "pairs.parquet",
        "quarantined_antibodies.parquet",
        "quarantined_pairs.parquet",
    }


def test_independent_audit_replays_every_pair_without_tier2_or_unknown_pairs() -> None:
    audit, digest = _read("independent_audit.json")
    assert digest == "1e45110b343a083327c90f31f3a824f3dd2d8573d0570706c9bd7932739a2a9d"
    assert audit["normalization_receipt_sha256"] == (
        "aee48d4cfa6ebf07e95326ddc06833747913e0260ff45abbfef1d973bffbf273"
    )
    assert audit["status"] == "pass_tier1_normalization_independent_audit"
    assert audit["semantic_comparison"] == (
        "independent_entity_rekey_and_exact_source_order_pair_assay_replay"
    )
    assert audit["tier2_admitted"] is False
    assert audit["unknown_cross_pairs_created"] == 0
    assert audit["counts"]["admitted_pairs"] + audit["counts"][
        "quarantined_pairs"
    ] == 573_891
