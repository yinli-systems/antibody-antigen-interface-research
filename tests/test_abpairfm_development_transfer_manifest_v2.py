from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_protocol_preserves_primary_sealed_test() -> None:
    protocol = json.loads(
        (ROOT / "protocols" / "abpairfm_development_transfer_matrix_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["primary_sealed_role_authority"] == "standard"
    assert protocol["authorization"]["test_access"] is False
    assert protocol["test_access"] is False
    assert protocol["split_slices"]["double_cold_validation"]["expected_count"] == 324
    assert protocol["split_slices"]["temporal_validation"]["expected_count"] == 337


def test_v2_builder_subtracts_standard_sealed_ids_before_publication() -> None:
    text = (
        ROOT / "scripts" / "build_abpairfm_development_transfer_manifest_v2.py"
    ).read_text(encoding="utf-8")
    assert "raw_double_validation - standard_sealed" in text
    assert "raw_temporal_validation - standard_sealed" in text
    assert '"primary_sealed_role_ids_retained": False' in text
    assert '"test_access": False' in text
