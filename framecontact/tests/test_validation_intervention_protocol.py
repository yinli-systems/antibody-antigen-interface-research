from __future__ import annotations

import json
from pathlib import Path


PROTOCOL = Path(__file__).resolve().parents[1] / "protocols" / "framecontact_validation_interventions_v1.json"
RECEIPT = Path(__file__).resolve().parents[1] / "protocols" / "FRAMECONTACT_VALIDATION_INTERVENTIONS_V1_RECEIPT.json"


def test_validation_intervention_protocol_is_frozen_and_test_free() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["protocol_status"] == "frozen_before_any_intervention_metric_was_read"
    assert payload["reporting"]["selection_performed_by_audit"] is False
    assert payload["reporting"]["test_loaded"] is False
    assert payload["reporting"]["external_test_loaded"] is False


def test_validation_intervention_protocol_registers_negative_and_causal_controls() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    controls = {item["name"]: item for item in payload["interventions"]}
    assert controls["independent_monomer_rigid_transform"]["role"] == "negative_control"
    assert controls["geometry_shuffle"]["role"] == "causal_control"
    assert controls["local_frame_corruption"]["role"] == "causal_control"
    assert controls["degree_matched_graph_rewiring"]["role"] == "causal_control"
    assert controls["coordinate_noise"]["angstrom_standard_deviations"] == [0.1, 0.25, 0.5]


def test_validation_intervention_receipt_binds_protocol_and_evaluator() -> None:
    import hashlib

    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for file_key, hash_key in (
        ("protocol_file", "protocol_file_sha256"),
        ("evaluator_file", "evaluator_file_sha256"),
    ):
        observed = hashlib.sha256((root / receipt[file_key]).read_bytes()).hexdigest()
        assert observed == receipt[hash_key]
    for path, expected in receipt["test_files"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
    assert receipt["intervention_metrics_accessed_before_freeze"] is False
    assert receipt["test_loaded"] is False
    assert receipt["external_test_loaded"] is False
