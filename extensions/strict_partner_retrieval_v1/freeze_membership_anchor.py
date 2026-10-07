#!/usr/bin/env python3
"""Externally freeze one successful provisional split membership for replay only."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROVISIONAL_FORMAT = "train_only_nested_core_feasibility_provisional_v2"
LEDGER_FORMAT = "train_only_nested_core_feasibility_provisional_attempt_v2"
ANCHOR_FORMAT = "strict_partner_retrieval_membership_anchor_v1"
PROVISIONAL_ROLE = "provisional_unpinned_core_feasibility_discovery"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_embedded_scientific_hash(payload: Mapping[str, Any], label: str) -> str:
    claimed = payload.get("scientific_payload_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError(f"{label} lacks a valid scientific_payload_sha256")
    material = dict(payload)
    material.pop("scientific_payload_sha256", None)
    actual = canonical_payload_sha256(material)
    if actual != claimed:
        raise ValueError(f"{label} scientific payload hash mismatch")
    return actual


def require_false(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        if payload.get(key) is not False:
            raise ValueError(f"{label}.{key} must be false")


def verify(
    assignment_path: Path,
    start_ledger_path: Path,
    outcome_ledger_path: Path,
    expected_attempt_id: str,
    expected_joint_policy_sha256: str,
) -> dict[str, Any]:
    assignment = load_json(assignment_path)
    start = load_json(start_ledger_path)
    outcome = load_json(outcome_ledger_path)

    if assignment.get("format_version") != PROVISIONAL_FORMAT:
        raise ValueError("Unexpected provisional assignment format")
    if assignment.get("artifact_role") != PROVISIONAL_ROLE:
        raise ValueError("Assignment is not a provisional unpinned discovery")
    if assignment.get("stage") != "completed_provisional_unpinned_core_feasibility_discovery":
        raise ValueError("Assignment stage is not complete provisional discovery")
    if assignment.get("construction_invariants_passed") is not True:
        raise ValueError("Construction invariants did not pass")
    if assignment.get("core_feasibility_constraints_passed") is not True:
        raise ValueError("Core feasibility constraints did not pass")
    require_false(
        assignment,
        (
            "authorized_for_training",
            "authorized_for_model_selection",
            "authorized_for_nested_model_selection",
            "split_domain_gate0_requirement_satisfied",
        ),
        "assignment",
    )
    assignment_scientific_sha = verify_embedded_scientific_hash(
        assignment, "assignment"
    )

    workflow = assignment.get("workflow_identity")
    if not isinstance(workflow, dict):
        raise ValueError("Assignment lacks workflow_identity")
    if workflow.get("joint_assignment_mode") != "provisional_discovery":
        raise ValueError("Assignment mode is not provisional_discovery")
    if workflow.get("provisional_attempt_id") != expected_attempt_id:
        raise ValueError("Assignment provisional attempt ID mismatch")
    if workflow.get("pinned_replay_id") is not None:
        raise ValueError("Provisional assignment already has a replay ID")
    if workflow.get("source_provisional_attempt_id") is not None:
        raise ValueError("Provisional assignment already claims a replay source")

    selector = assignment.get("construction", {}).get("selector")
    if not isinstance(selector, dict):
        raise ValueError("Assignment lacks selector receipt")
    selected_digest = selector.get("selected_binary_membership_sha256")
    matrix_digest = selector.get("matrix_sha256")
    policy_digest = selector.get("exact_joint_policy_sha256")
    for name, value in (
        ("selected membership", selected_digest),
        ("matrix", matrix_digest),
        ("joint policy", policy_digest),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Invalid {name} digest")
    if policy_digest != expected_joint_policy_sha256:
        raise ValueError("Joint policy digest differs from the externally expected policy")
    if selector.get("certified_core_feasibility_assignment") is not True:
        raise ValueError("Selector did not certify a core-feasible assignment")
    if selector.get("solver_solution_present") is not True:
        raise ValueError("Selector has no solver solution")
    if selector.get("exact_integer_row_audit_passed") is not True:
        raise ValueError("Selector integer row audit did not pass")
    if selector.get("maximum_integrality_error", 1.0) > selector.get(
        "accepted_integrality_tolerance", 0.0
    ):
        raise ValueError("Selector integrality error exceeds tolerance")

    if start.get("format_version") != LEDGER_FORMAT:
        raise ValueError("Unexpected start-ledger format")
    if start.get("ledger_entry_kind") != "attempt_started":
        raise ValueError("Start ledger has wrong kind")
    if start.get("attempt_status") != "started_no_outcome_yet":
        raise ValueError("Start ledger has unexpected state")
    if start.get("provisional_attempt_id") != expected_attempt_id:
        raise ValueError("Start-ledger attempt ID mismatch")
    require_false(
        start,
        (
            "authorized_for_training",
            "authorized_for_model_selection",
            "authorized_for_nested_model_selection",
            "split_domain_gate0_requirement_satisfied",
            "externally_pinned_assignment_identity",
            "eligible_for_independent_pinned_replay_audit",
        ),
        "start",
    )
    start_scientific_sha = verify_embedded_scientific_hash(start, "start")

    if outcome.get("format_version") != LEDGER_FORMAT:
        raise ValueError("Unexpected outcome-ledger format")
    if outcome.get("ledger_entry_kind") != "attempt_outcome":
        raise ValueError("Outcome ledger has wrong kind")
    if outcome.get("attempt_status") != "candidate_published":
        raise ValueError("Outcome ledger does not report candidate_published")
    if outcome.get("provisional_attempt_id") != expected_attempt_id:
        raise ValueError("Outcome-ledger attempt ID mismatch")
    require_false(
        outcome,
        (
            "authorized_for_training",
            "authorized_for_model_selection",
            "authorized_for_nested_model_selection",
            "split_domain_gate0_requirement_satisfied",
            "externally_pinned_assignment_identity",
            "eligible_for_independent_pinned_replay_audit",
        ),
        "outcome",
    )
    outcome_scientific_sha = verify_embedded_scientific_hash(outcome, "outcome")

    assignment_file_sha = sha256(assignment_path)
    if outcome.get("provisional_assignment_file_sha256") != assignment_file_sha:
        raise ValueError("Outcome ledger does not bind assignment file bytes")
    if (
        outcome.get("provisional_assignment_scientific_payload_sha256")
        != assignment_scientific_sha
    ):
        raise ValueError("Outcome ledger does not bind assignment scientific payload")
    if outcome.get("start_ledger_scientific_payload_sha256") != start_scientific_sha:
        raise ValueError("Outcome ledger does not bind the start ledger")
    if outcome.get("selected_binary_membership_sha256") != selected_digest:
        raise ValueError("Outcome membership digest differs from assignment")
    if outcome.get("matrix_sha256") != matrix_digest:
        raise ValueError("Outcome matrix digest differs from assignment")
    if outcome.get("exact_joint_policy_sha256") != policy_digest:
        raise ValueError("Outcome policy digest differs from assignment")
    if outcome.get("core_feasibility_constraints_passed") is not True:
        raise ValueError("Outcome ledger does not confirm core feasibility")
    solver = outcome.get("solver_evidence")
    if not isinstance(solver, dict):
        raise ValueError("Outcome lacks solver evidence")
    for key in (
        "solver_solution_present",
        "solver_certified_optimal_for_constant_zero_objective",
        "solver_reported_mip_gap_is_finite_zero",
        "exact_integer_row_audit_passed",
        "certified_core_feasibility_assignment",
    ):
        if solver.get(key) is not True:
            raise ValueError(f"Outcome solver evidence failed: {key}")

    return {
        "format_version": ANCHOR_FORMAT,
        "status": "FROZEN_FOR_SEPARATE_PINNED_REPLAY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provisional_attempt_id": expected_attempt_id,
        "assignment": {
            "path": str(assignment_path.resolve()),
            "file_sha256": assignment_file_sha,
            "scientific_payload_sha256": assignment_scientific_sha,
        },
        "start_ledger": {
            "path": str(start_ledger_path.resolve()),
            "file_sha256": sha256(start_ledger_path),
            "scientific_payload_sha256": start_scientific_sha,
        },
        "outcome_ledger": {
            "path": str(outcome_ledger_path.resolve()),
            "file_sha256": sha256(outcome_ledger_path),
            "scientific_payload_sha256": outcome_scientific_sha,
        },
        "selected_binary_membership_sha256": selected_digest,
        "matrix_sha256": matrix_digest,
        "exact_joint_policy_sha256": policy_digest,
        "replay_authorized": True,
        "training_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
        "test_access": False,
        "ood_access": False,
        "required_next_step": "separate_create_only_externally_pinned_replay_then_independent_audit",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--start-ledger", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path, required=True)
    parser.add_argument("--expected-provisional-attempt-id", required=True)
    parser.add_argument("--expected-joint-policy-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = verify(
        args.assignment,
        args.start_ledger,
        args.outcome_ledger,
        args.expected_provisional_attempt_id,
        args.expected_joint_policy_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "selected_binary_membership_sha256": payload["selected_binary_membership_sha256"],
        "replay_authorized": True,
        "training_authorized": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
