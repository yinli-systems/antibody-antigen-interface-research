"""Tests for the V6-B create-only row-shard production controller."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_v6_b_row_shard_telemetry.py"
SPEC = importlib.util.spec_from_file_location("v6_b_row_shard", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_verified_read_rejects_a_byte_change(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(b"{}\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert SCRIPT.read_verified(path, digest, "input") == b"{}\n"
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(SCRIPT.V6BRowShardError, match="SHA-256 differs"):
        SCRIPT.read_verified(path, digest, "input")


def test_verified_read_rejects_a_symbolic_path_component(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "input.json"
    payload.write_bytes(b"{}\n")
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    with pytest.raises(SCRIPT.V6BRowShardError, match="symbolic link"):
        SCRIPT.read_verified(link / "input.json", digest, "input")


def test_create_only_publication_is_immutable_and_non_overwriting(
    tmp_path: Path,
) -> None:
    receipt = {
        "format_version": SCRIPT.FORMAT_VERSION,
        "terminal_status": "TELEMETRY_HEADROOM_PASSED",
        "training_authorized": False,
    }
    output = tmp_path / "evidence"
    receipt_path, report_path = SCRIPT.publish_create_only(
        output,
        receipt,
        b"# report\n",
    )
    assert json.loads(receipt_path.read_text(encoding="ascii")) == receipt
    assert report_path.read_bytes() == b"# report\n"
    assert os.stat(receipt_path).st_mode & 0o222 == 0
    assert os.stat(report_path).st_mode & 0o222 == 0
    assert os.stat(output).st_mode & 0o222 == 0
    with pytest.raises(SCRIPT.V6BRowShardError, match="already exists"):
        SCRIPT.publish_create_only(output, receipt, b"# second\n")


def test_markdown_repeats_non_authorizing_boundary() -> None:
    core = {
        "terminal_status": "TELEMETRY_HEADROOM_PASSED",
        "telemetry": {
            "full_variable_count": 10,
            "full_row_count": 20,
            "full_nonzero_count": 30,
            "row_selection": {"selected_row_count": 2},
            "serialized_full_model_bytes": 40,
            "observed_peak_rss_bytes": 50,
            "projected_peak_bytes": 60,
            "available_memory_bytes": 100,
            "resource_headroom_passed": True,
        },
        "partition": {
            "master": {"row_count": 5},
            "inner_row_counts": [5, 5, 5],
        },
    }
    report = SCRIPT._markdown(core).decode("ascii")
    assert "invoked no solver" in report
    assert "does not" in report
    assert "authorize feasibility" in report


def _published_authority() -> bytes:
    pair_rows = [
        {
            "pair_id": "f" * 64,
            "antibody_id": "antibody-a",
            "antigen_id": "antigen-a",
            "source_instance_ids": ["source-a"],
        },
        {
            "pair_id": "0" * 64,
            "antibody_id": "antibody-b",
            "antigen_id": "antigen-b",
            "source_instance_ids": ["source-b"],
        },
    ]
    scientific = {
        "format_version": "score_free_authority_v2",
        "input_sha256": {},
        "authority_membership_sha256": "0" * 64,
        "implementation_source_sha256": {},
        "counts": {},
        "antibodies": [],
        "antigens": [],
        "pairs": pair_rows,
        "quantile_schemes": [],
        "forbidden_outcome_or_model_fields_accepted": False,
    }
    scientific["scientific_payload_sha256"] = hashlib.sha256(
        SCRIPT.canonical_json_bytes(scientific)
    ).hexdigest()
    published = {
        **scientific,
        "verified_file_receipts": {
            "train_authority": {},
            "canonical_manifest": {},
            "edge_ledger_vh": {},
            "edge_ledger_vl": {},
            "edge_ledger_concatenated_antibody": {},
            "edge_ledger_antigen": {},
        },
        "publication_controller_sha256": "1" * 64,
    }
    published["publication_sha256"] = hashlib.sha256(
        SCRIPT.canonical_json_bytes(published)
    ).hexdigest()
    return SCRIPT.canonical_json_bytes(published)


def test_publication_envelope_is_verified_before_scientific_extraction() -> None:
    scientific, envelope = SCRIPT.extract_scientific_authority(
        _published_authority()
    )
    assert json.loads(scientific)["format_version"] == "score_free_authority_v2"
    assert envelope["publication_controller_sha256"] == "1" * 64
    assert envelope["scientific_file_sha256"] == hashlib.sha256(scientific).hexdigest()


def test_publication_envelope_rejects_unknown_or_tampered_metadata() -> None:
    unknown = json.loads(_published_authority())
    unknown["unexpected"] = True
    with pytest.raises(SCRIPT.V6BRowShardError, match="schema differs"):
        SCRIPT.extract_scientific_authority(SCRIPT.canonical_json_bytes(unknown))
    tampered = json.loads(_published_authority())
    tampered["publication_controller_sha256"] = "2" * 64
    with pytest.raises(SCRIPT.V6BRowShardError, match="publication digest differs"):
        SCRIPT.extract_scientific_authority(SCRIPT.canonical_json_bytes(tampered))


def test_pair_order_normalization_changes_only_record_order() -> None:
    scientific, _ = SCRIPT.extract_scientific_authority(_published_authority())
    normalized, receipt = SCRIPT.normalize_scientific_pair_order(scientific)
    before = json.loads(scientific)["pairs"]
    after = json.loads(normalized)["pairs"]
    assert [row["pair_id"] for row in before] == ["f" * 64, "0" * 64]
    assert [row["pair_id"] for row in after] == ["0" * 64, "f" * 64]
    assert sorted(before, key=lambda row: row["pair_id"]) == after
    assert receipt["pair_count"] == 2
    assert receipt["semantic_fields_changed"] is False


def test_pair_order_normalization_rejects_noncanonical_endpoint_order() -> None:
    scientific, _ = SCRIPT.extract_scientific_authority(_published_authority())
    payload = json.loads(scientific)
    payload["pairs"].reverse()
    payload.pop("scientific_payload_sha256")
    payload["scientific_payload_sha256"] = hashlib.sha256(
        SCRIPT.canonical_json_bytes(payload)
    ).hexdigest()
    with pytest.raises(SCRIPT.V6BRowShardError, match="canonical endpoint order"):
        SCRIPT.normalize_scientific_pair_order(SCRIPT.canonical_json_bytes(payload))
