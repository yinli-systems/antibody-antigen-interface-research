"""Tests for the fail-closed PAIRScale data and license registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from interaction_first.data_registry import (
    DataRegistryError,
    decide_registry_admission,
    load_registry_collection,
    load_source_registry,
)


def _registry_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data_registry" / "sources"


def test_registered_sources_validate_and_only_audited_avida_is_admitted() -> None:
    registries = load_registry_collection(_registry_root())
    assert [value.source_id for value in registries] == [
        "avida_hil6_2024_11",
        "chimera_bench_2026_05",
        "ppiref_zenodo_14845086",
        "sabdab2_ml_zenodo_20083995",
    ]
    for registry in registries:
        decision = decide_registry_admission(
            registry,
            use_channel="noncommercial_research",
            purpose="training",
        )
        if registry.source_id == "avida_hil6_2024_11":
            assert decision.admitted
            assert decision.reasons == ()
            assert "raw_bytes_not_materialized" not in decision.reasons
        else:
            assert not decision.admitted
            assert "normalization_not_complete" in decision.reasons
            assert "duplicate_groups_not_complete" in decision.reasons
            if registry.source_id == "sabdab2_ml_zenodo_20083995":
                assert "raw_bytes_not_materialized" not in decision.reasons
            else:
                assert "raw_bytes_not_materialized" in decision.reasons


def test_asset_level_license_routing_separates_data_from_code() -> None:
    registries = {
        value.source_id: value for value in load_registry_collection(_registry_root())
    }
    avida = registries["avida_hil6_2024_11"]
    commercial = decide_registry_admission(
        avida,
        use_channel="commercial",
        purpose="training",
    )
    assert "data_license_blocks_commercial_use" in commercial.reasons
    assert any(
        asset["asset_scope"] == "code"
        and asset["declared_license_spdx"] == "MIT"
        for asset in avida.license["assets"]
    )
    assert any(
        asset["asset_scope"] == "data"
        and asset["declared_license_spdx"] == "CC-BY-NC-4.0"
        for asset in avida.license["assets"]
    )


def test_chimera_is_reserved_from_training_independent_of_license() -> None:
    registry = load_source_registry(
        _registry_root() / "chimera_bench_2026_05"
    )
    training = decide_registry_admission(
        registry,
        use_channel="noncommercial_research",
        purpose="training",
    )
    assert "source_reserved_for_external_confirmatory_evaluation" in (
        training.reasons
    )
    confirmatory = decide_registry_admission(
        registry,
        use_channel="noncommercial_research",
        purpose="external_confirmatory_evaluation",
    )
    assert "source_reserved_for_external_confirmatory_evaluation" not in (
        confirmatory.reasons
    )


def test_extra_file_and_unknown_license_fail_closed(tmp_path: Path) -> None:
    source = _registry_root() / "avida_hil6_2024_11"
    copied = tmp_path / source.name
    copied.mkdir()
    for path in source.iterdir():
        (copied / path.name).write_bytes(path.read_bytes())
    (copied / "unexpected.json").write_text("{}\n")
    with pytest.raises(DataRegistryError, match="source files differ"):
        load_source_registry(copied)
    (copied / "unexpected.json").unlink()
    license_path = copied / "license_receipt.json"
    payload = json.loads(license_path.read_text())
    data_asset = next(
        asset for asset in payload["assets"] if asset["asset_scope"] == "data"
    )
    data_asset["license_route"] = "unresolved"
    data_asset["commercial_use_permitted"] = False
    license_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    registry = load_source_registry(copied)
    decision = decide_registry_admission(
        registry,
        use_channel="academic",
        purpose="training",
    )
    assert "data_license_unresolved" in decision.reasons


def test_complete_registry_requires_raw_binding_and_can_be_admitted(
    tmp_path: Path,
) -> None:
    source = _registry_root() / "avida_hil6_2024_11"
    copied = tmp_path / source.name
    copied.mkdir()
    for path in source.iterdir():
        (copied / path.name).write_bytes(path.read_bytes())
    source_path = copied / "source_manifest.json"
    source_payload = json.loads(source_path.read_text())
    source_payload["duplicate_groups_status"] = "complete"
    source_path.write_text(json.dumps(source_payload, sort_keys=True) + "\n")
    raw_path = copied / "raw_sha256.json"
    raw_payload = json.loads(raw_path.read_text())
    raw_payload.update(
        {
            "status": "materialized",
            "materialized_at_utc": "2026-07-27T15:00:00Z",
            "files": [
                {
                    "path": "raw/train.csv",
                    "size_bytes": 7,
                    "sha256": hashlib.sha256(b"payload").hexdigest(),
                }
            ],
        }
    )
    raw_path.write_text(json.dumps(raw_payload, sort_keys=True) + "\n")
    normalization_path = copied / "normalization_receipt.json"
    normalization_payload = json.loads(normalization_path.read_text())
    normalization_payload.update(
        {
            "status": "complete",
            "normalized_at_utc": "2026-07-27T15:01:00Z",
            "input_raw_inventory_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
            "normalizer_source_sha256": hashlib.sha256(b"normalizer").hexdigest(),
            "outputs": [
                {
                    "path": "normalized/pairs.jsonl",
                    "size_bytes": 10,
                    "sha256": hashlib.sha256(b"normalized").hexdigest(),
                }
            ],
        }
    )
    normalization_path.write_text(
        json.dumps(normalization_payload, sort_keys=True) + "\n"
    )
    registry = load_source_registry(copied)
    decision = decide_registry_admission(
        registry,
        use_channel="noncommercial_research",
        purpose="training",
    )
    assert decision.admitted
    raw_payload["files"][0]["size_bytes"] = 8
    raw_path.write_text(json.dumps(raw_payload, sort_keys=True) + "\n")
    with pytest.raises(DataRegistryError, match="not bound to the raw inventory"):
        load_source_registry(copied)
