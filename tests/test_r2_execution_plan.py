"""Validate the frozen train-only R2 execution lattice."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from interaction_first.r2_execution_plan import (
    FROZEN_SEEDS,
    INNER_CANDIDATES,
    MONOMER_VARIANTS,
    R2ExecutionPlanError,
    build_train_only_execution_plan,
    canonical_json_bytes,
    validate_train_only_execution_plan,
)


ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "protocols" / "sabdab2_partner_geometry_r2_v1.json"


def _protocol() -> tuple[dict, str]:
    payload = PROTOCOL_PATH.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def test_complete_five_seed_train_only_lattice_is_deterministic() -> None:
    protocol, protocol_sha256 = _protocol()
    first = build_train_only_execution_plan(
        protocol,
        protocol_file_sha256=protocol_sha256,
    )
    second = build_train_only_execution_plan(
        dict(reversed(list(protocol.items()))),
        protocol_file_sha256=protocol_sha256,
    )

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert validate_train_only_execution_plan(first) == first["plan_sha256"]
    assert first["fixed_seeds"] == list(FROZEN_SEEDS)
    assert first["counts"] == {
        "monomer_confirmation_runs": 20,
        "nested_inner_runs": 135,
        "total_materialized_train_only_runs": 155,
        "outer_runs": 0,
        "external_test_runs": 0,
    }
    assert {
        item["variant"] for item in first["monomer_confirmation_runs"]
    } == set(MONOMER_VARIANTS)
    assert {
        item["candidate_id"] for item in first["nested_inner_runs"]
    } == set(INNER_CANDIDATES)
    assert {
        item["seed"] for item in first["nested_inner_runs"]
    } == set(FROZEN_SEEDS)
    assert {
        (item["outer_fold"], item["inner_fold"])
        for item in first["nested_inner_runs"]
    } == {(outer, inner) for outer in range(3) for inner in range(3)}
    assert all(
        item["outer_or_test_inference"] is False
        for item in first["nested_inner_runs"]
    )
    assert first["post_inner_templates"]["outer_refit"]["materialized"] is False
    assert first["post_inner_templates"]["external_test"]["materialized"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("seed", "seeds differ"),
        ("early_stopping", "early stopping"),
        ("test_access", "external_test_mount_or_read"),
        ("candidate_budget", "budget or checkpoint"),
        ("pair_layers", "two pair-update"),
    ),
)
def test_protocol_mutations_fail_closed(mutation: str, message: str) -> None:
    protocol, protocol_sha256 = _protocol()
    altered = copy.deepcopy(protocol)
    if mutation == "seed":
        altered["fixed_seeds"][-1] = 999
    elif mutation == "early_stopping":
        altered["candidate_ladder"][1]["early_stopping"] = True
    elif mutation == "test_access":
        altered["authorization"]["external_test_mount_or_read"] = True
    elif mutation == "candidate_budget":
        altered["candidate_ladder"][2]["epochs"] = 41
    elif mutation == "pair_layers":
        altered["candidate_ladder"][2]["num_pair_update_layers"] = 3
    else:
        raise AssertionError(f"Unhandled mutation {mutation}")
    with pytest.raises(R2ExecutionPlanError, match=message):
        build_train_only_execution_plan(
            altered,
            protocol_file_sha256=protocol_sha256,
        )


def test_plan_tamper_and_authority_escalation_fail_closed() -> None:
    protocol, protocol_sha256 = _protocol()
    plan = build_train_only_execution_plan(
        protocol,
        protocol_file_sha256=protocol_sha256,
    )

    tampered = copy.deepcopy(plan)
    tampered["nested_inner_runs"][0]["epochs"] += 1
    with pytest.raises(R2ExecutionPlanError, match="digest differs"):
        validate_train_only_execution_plan(tampered)

    escalated = copy.deepcopy(plan)
    escalated["authorization"]["scientific_training"] = True
    unsigned = dict(escalated)
    unsigned.pop("plan_sha256")
    escalated["plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(R2ExecutionPlanError, match="grants authority"):
        validate_train_only_execution_plan(escalated)


def test_outer_and_test_runs_cannot_be_smuggled_into_inner_inventory() -> None:
    protocol, protocol_sha256 = _protocol()
    plan = build_train_only_execution_plan(
        protocol,
        protocol_file_sha256=protocol_sha256,
    )
    altered = copy.deepcopy(plan)
    altered["nested_inner_runs"][0]["outer_or_test_inference"] = True
    unsigned = dict(altered)
    unsigned.pop("plan_sha256")
    altered["plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(R2ExecutionPlanError, match="outer/test inference"):
        validate_train_only_execution_plan(altered)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", 999),
        ("epochs", 39),
        ("global_batch_size", 64),
        ("checkpoint_rule", "validation_best"),
        ("candidate_id", "R3"),
    ),
)
def test_rehashed_inner_run_tampering_fails_closed(
    field: str,
    value: object,
) -> None:
    protocol, protocol_sha256 = _protocol()
    plan = build_train_only_execution_plan(
        protocol,
        protocol_file_sha256=protocol_sha256,
    )
    altered = copy.deepcopy(plan)
    altered["nested_inner_runs"][0][field] = value
    unsigned = dict(altered)
    unsigned.pop("plan_sha256")
    altered["plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(R2ExecutionPlanError, match="inner"):
        validate_train_only_execution_plan(altered)


def test_rehashed_monomer_run_tampering_fails_closed() -> None:
    protocol, protocol_sha256 = _protocol()
    plan = build_train_only_execution_plan(
        protocol,
        protocol_file_sha256=protocol_sha256,
    )
    altered = copy.deepcopy(plan)
    altered["monomer_confirmation_runs"][0]["checkpoint_rule"] = (
        "validation_best"
    )
    unsigned = dict(altered)
    unsigned.pop("plan_sha256")
    altered["plan_sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(R2ExecutionPlanError, match="monomer run fields"):
        validate_train_only_execution_plan(altered)
