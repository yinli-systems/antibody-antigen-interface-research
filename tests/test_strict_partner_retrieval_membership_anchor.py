from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "extensions/strict_partner_retrieval_v1/freeze_membership_anchor.py"
spec = importlib.util.spec_from_file_location("membership_anchor", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _scientific(payload: dict) -> dict:
    result = dict(payload)
    result["scientific_payload_sha256"] = module.canonical_payload_sha256(payload)
    return result


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    attempt = "attempt_001"
    policy = "a" * 64
    selector = {
        "selected_binary_membership_sha256": "b" * 64,
        "matrix_sha256": "c" * 64,
        "exact_joint_policy_sha256": policy,
        "certified_core_feasibility_assignment": True,
        "solver_solution_present": True,
        "exact_integer_row_audit_passed": True,
        "maximum_integrality_error": 0.0,
        "accepted_integrality_tolerance": 1e-7,
    }
    assignment = _scientific({
        "format_version": module.PROVISIONAL_FORMAT,
        "artifact_role": module.PROVISIONAL_ROLE,
        "stage": "completed_provisional_unpinned_core_feasibility_discovery",
        "construction_invariants_passed": True,
        "core_feasibility_constraints_passed": True,
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "authorized_for_nested_model_selection": False,
        "split_domain_gate0_requirement_satisfied": False,
        "workflow_identity": {
            "joint_assignment_mode": "provisional_discovery",
            "provisional_attempt_id": attempt,
            "pinned_replay_id": None,
            "source_provisional_attempt_id": None,
        },
        "construction": {"selector": selector},
    })
    start = _scientific({
        "format_version": module.LEDGER_FORMAT,
        "ledger_entry_kind": "attempt_started",
        "attempt_status": "started_no_outcome_yet",
        "provisional_attempt_id": attempt,
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "authorized_for_nested_model_selection": False,
        "split_domain_gate0_requirement_satisfied": False,
        "externally_pinned_assignment_identity": False,
        "eligible_for_independent_pinned_replay_audit": False,
    })
    ap = tmp_path / "assignment.json"
    sp = tmp_path / "started.json"
    op = tmp_path / "outcome.json"
    _write(ap, assignment)
    _write(sp, start)
    solver = {
        "solver_solution_present": True,
        "solver_certified_optimal_for_constant_zero_objective": True,
        "solver_reported_mip_gap_is_finite_zero": True,
        "exact_integer_row_audit_passed": True,
        "certified_core_feasibility_assignment": True,
    }
    outcome = _scientific({
        "format_version": module.LEDGER_FORMAT,
        "ledger_entry_kind": "attempt_outcome",
        "attempt_status": "candidate_published",
        "provisional_attempt_id": attempt,
        "authorized_for_training": False,
        "authorized_for_model_selection": False,
        "authorized_for_nested_model_selection": False,
        "split_domain_gate0_requirement_satisfied": False,
        "externally_pinned_assignment_identity": False,
        "eligible_for_independent_pinned_replay_audit": False,
        "provisional_assignment_file_sha256": hashlib.sha256(ap.read_bytes()).hexdigest(),
        "provisional_assignment_scientific_payload_sha256": assignment["scientific_payload_sha256"],
        "start_ledger_scientific_payload_sha256": start["scientific_payload_sha256"],
        "selected_binary_membership_sha256": selector["selected_binary_membership_sha256"],
        "matrix_sha256": selector["matrix_sha256"],
        "exact_joint_policy_sha256": policy,
        "core_feasibility_constraints_passed": True,
        "solver_evidence": solver,
    })
    _write(op, outcome)
    return ap, sp, op, attempt, policy


def test_valid_anchor_is_replay_only(tmp_path: Path) -> None:
    ap, sp, op, attempt, policy = fixture(tmp_path)
    result = module.verify(ap, sp, op, attempt, policy)
    assert result["status"] == "FROZEN_FOR_SEPARATE_PINNED_REPLAY"
    assert result["replay_authorized"] is True
    assert result["training_authorized"] is False
    assert result["outer_evaluation_authorized"] is False


def test_membership_mismatch_fails(tmp_path: Path) -> None:
    ap, sp, op, attempt, policy = fixture(tmp_path)
    value = json.loads(op.read_text())
    value["selected_binary_membership_sha256"] = "d" * 64
    material = dict(value)
    material.pop("scientific_payload_sha256")
    value["scientific_payload_sha256"] = module.canonical_payload_sha256(material)
    _write(op, value)
    try:
        module.verify(ap, sp, op, attempt, policy)
    except ValueError as error:
        assert "membership" in str(error).lower()
    else:
        raise AssertionError("membership mismatch was accepted")
