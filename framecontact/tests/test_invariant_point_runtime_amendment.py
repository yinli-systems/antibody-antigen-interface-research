"""Evidence-lock checks for the invariant-point algebraic memory rewrite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "protocols/framecontact_invariant_point_memory_amendment_v1.json"
)
RECEIPT = (
    REPOSITORY_ROOT
    / "protocols/FRAMECONTACT_INVARIANT_POINT_MEMORY_AMENDMENT_V1_RECEIPT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_invariant_point_runtime_amendment_is_hash_bound_and_test_sealed() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert _sha256(PROTOCOL) == receipt["protocol_file_sha256"]
    assert _canonical_sha256(protocol) == receipt["protocol_canonical_sha256"]
    implementation = protocol["implementation_change"]
    assert _sha256(REPOSITORY_ROOT / implementation["recovery_source_path"]) == (
        implementation["recovery_source_sha256"]
    )
    # This is a historical runtime amendment.  Later source additions must not
    # rewrite it, so authenticate the exact source through the preserved V7
    # source manifest instead of requiring today's mutable development path to
    # retain the old digest forever.
    source_manifest = json.loads(
        (
            REPOSITORY_ROOT
            / "receipts/FRAMECONTACT_FRONTIER_SOURCE_V7_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    records = {record["path"]: record for record in source_manifest["records"]}
    assert (
        records[implementation["main_development_source_path"]]["sha256"]
        == implementation["main_development_source_sha256"]
    )
    assert _sha256(REPOSITORY_ROOT / protocol["validation"]["validator_path"]) == (
        protocol["validation"]["validator_sha256"]
    )
    assert len(protocol["validation"]["main_regression_test_source_sha256"]) == 64
    for key in (
        "model_parameters_changed",
        "attention_bias_definition_changed",
        "optimizer_or_loss_changed",
        "dataset_split_or_labels_changed",
        "selection_rule_changed",
    ):
        assert implementation[key] is False
    assert receipt["frozen_test_authorized"] is False
    assert receipt["affected_seeds"] == [17, 42, 73, 101, 137]
