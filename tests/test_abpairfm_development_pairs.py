from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "pretraining" / "abpairfm_development_pair_panel_v1.json"


def module():
    spec = importlib.util.spec_from_file_location(
        "freeze_abpairfm_development_pairs",
        ROOT / "scripts/freeze_abpairfm_development_pairs.py",
    )
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_pair_hashes_are_chain_framed_and_endpoint_hash_is_explicit() -> None:
    value = module()
    assert value.framed_antibody_sha256("AC", "DE") != value.sha256_text("ACDE")
    assert value.canonical_pair_sha256("AC", "DE", "FG") != value.canonical_pair_sha256(
        "ACD", "E", "FG"
    )
    record = value.pair_record(
        dataset="asep",
        identifier="example",
        heavy="AC",
        light="DE",
        antigen="FG",
        source_antibody_sha256=value.framed_antibody_sha256("AC", "DE"),
        source_antigen_sha256=value.sha256_text("FG"),
        antibody_cluster_id="ab",
        antigen_cluster_id="ag",
        panel_entry={
            "wrong_antibody_instance_ids": ["other"] * 31,
            "wrong_antigen_instance_ids": ["other"] * 31,
        },
    )
    assert record["role"] == "development_validation"
    assert record["antibody_endpoint_sequence"] == "ACDE"
    assert record["antibody_endpoint_sha256"] == value.sha256_text("ACDE")


def test_decoy_validation_rejects_exact_endpoint_reuse() -> None:
    value = module()
    base = {
        "dataset": "asep",
        "source_antibody_sequence_sha256": "same-antibody",
        "source_antigen_sequence_sha256": "antigen-a",
        "wrong_antibody_instance_ids": ["b"],
        "wrong_antigen_instance_ids": ["b"],
    }
    records = [
        {**base, "instance_id": "a"},
        {
            **base,
            "instance_id": "b",
            "source_antigen_sequence_sha256": "antigen-b",
            "wrong_antibody_instance_ids": ["a"],
            "wrong_antigen_instance_ids": ["a"],
        },
    ]
    try:
        value.validate_decoy_endpoint_exclusion(records)
    except ValueError as error:
        assert "exact endpoint" in str(error)
    else:
        raise AssertionError("exact antibody endpoint reuse was accepted")


def test_sequence_validation_allows_missing_light_only_when_explicit() -> None:
    value = module()
    assert value.checked_sequence("", "light", optional=True) == ""
    try:
        value.checked_sequence("ACX", "bad")
    except ValueError as error:
        assert "noncanonical" in str(error)
    else:
        raise AssertionError("noncanonical sequence was accepted")


def test_frozen_pair_panel_summary_has_no_contact_or_test_authority() -> None:
    import json

    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["counts"]["records"] == 719
    assert result["counts"]["unique_exact_pairs"] == 583
    assert result["authorization"]["development_retrieval"] is True
    assert result["authorization"]["development_contact_labels"] is False
    assert result["test_access"] is False
