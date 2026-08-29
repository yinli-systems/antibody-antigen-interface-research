"""Frozen authority checks for the distogram and local-triangle experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "protocols/framecontact_distogram_exploratory_v1.json"
RECEIPT = (
    REPOSITORY_ROOT
    / "protocols/FRAMECONTACT_DISTOGRAM_EXPLORATORY_V1_RECEIPT.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    raw = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def test_distogram_protocol_is_hash_bound_and_development_only() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert _sha(PROTOCOL) == receipt["protocol_file_sha256"]
    assert _canonical(protocol) == receipt["protocol_canonical_sha256"]
    assert protocol["development_matrix"] == [
        "framecontact_distogram",
        "framecontact_distogram_triangle",
    ]
    assert protocol["model_selection"]["seeds"] == [17, 42, 73, 101, 137]
    assert protocol["data"]["pair_geometry_records"] == 3114
    assert protocol["data"]["test_role_forbidden"] is True
    boundary = protocol["evidence_boundary_at_freeze"]
    assert boundary["current_sabdab2_test_is_confirmatory"] is False
    assert boundary["current_sabdab2_test_model_predictions_accessed"] is False
    assert boundary["current_sabdab2_test_metrics_accessed"] is False
    assert protocol["access_audit"]["confirmatory_replacement_required"] is True


def test_triangle_condition_changes_only_the_registered_loss_weight() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    weights = protocol["training_objective"]["loss_weights_by_condition"]
    plain = dict(weights["framecontact_distogram"])
    triangle = dict(weights["framecontact_distogram_triangle"])
    assert plain.pop("triangle") == 0.0
    assert triangle.pop("triangle") == 0.1
    assert plain == triangle
    assert "intra-chain epitope-unary logits" in protocol["controlled_architecture"][
        "triangle_region"
    ]
