import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/antibody_ddp_communication_optimization_v1.json"
PARENT = ROOT / "protocols/antibody_efficiency_gate_v1.json"


def test_ddp_communication_protocol_is_bound_to_frozen_parent() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    parent_sha256 = hashlib.sha256(PARENT.read_bytes()).hexdigest()
    assert protocol["format_version"] == (
        "antibody_ddp_communication_optimization_v1"
    )
    assert protocol["parent_efficiency_protocol_sha256"] == parent_sha256
    assert protocol["promotion"]["thresholds_may_be_relaxed"] is False
    assert protocol["promotion"]["paired_gate_updates_per_mode"] == 500


def test_ddp_communication_grid_is_complete_and_score_free() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    candidates = protocol["candidates"]
    assert len(candidates) == 4
    assert {
        (item["communication_dtype"], item["batched_grad_copy"])
        for item in candidates
    } == {
        ("float32", False),
        ("float32", True),
        ("bfloat16", False),
        ("bfloat16", True),
    }
    assert protocol["correctness"] == {
        "downstream_labels_permitted": False,
        "exact_eight_rank_gradient_parity_required_every_update": True,
        "exact_eight_rank_model_and_optimizer_parity_required": True,
        "inherit_parent_numerical_thresholds": True,
        "same_membership_required": True,
        "updates": 20,
    }
    assert protocol["authorization"]["pilot_16_gpu"] is False
    assert protocol["authorization"]["pilot_32_gpu"] is False
