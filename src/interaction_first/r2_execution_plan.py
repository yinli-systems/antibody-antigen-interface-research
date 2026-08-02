"""Build the frozen train-only run lattice for partner-geometry R2.

The plan is prospective metadata. It never reads a dataset, starts training,
selects a checkpoint, or grants outer/test access. A separate authenticated
Gate 0 decision and post-freeze authorities are required by the launcher.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


PROTOCOL_FORMAT = "sabdab2_partner_geometry_r2_protocol_v1"
PLAN_FORMAT = "sabdab2_partner_geometry_r2_train_only_plan_v1"
FROZEN_SEEDS = (17, 42, 73, 101, 137)
MONOMER_VARIANTS = (
    "dense_sequence",
    "topology_only",
    "distance",
    "distance_direction",
)
INNER_CANDIDATES = ("D0", "R1", "R2")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class R2ExecutionPlanError(ValueError):
    """Raised when the frozen R2 execution contract is malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite JSON with one stable, ASCII-only representation."""
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise R2ExecutionPlanError(
            "R2 execution metadata must be finite ASCII JSON"
        ) from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise R2ExecutionPlanError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _require_exact_keys(
    value: Any,
    required: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R2ExecutionPlanError(f"{name} must be a mapping")
    observed = set(value)
    if observed != required:
        missing = sorted(required - observed)
        unexpected = sorted(observed - required)
        raise R2ExecutionPlanError(
            f"{name} fields differ: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return value


def _require_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise R2ExecutionPlanError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return int(value)


def _validate_protocol_authority(protocol: Mapping[str, Any]) -> None:
    if protocol.get("format_version") != PROTOCOL_FORMAT:
        raise R2ExecutionPlanError("unsupported R2 protocol format")
    if protocol.get("status") != (
        "implementation_authorized_training_blocked_at_gate_0"
    ):
        raise R2ExecutionPlanError("R2 protocol status is not frozen and blocked")
    authorization = protocol.get("authorization")
    if not isinstance(authorization, Mapping):
        raise R2ExecutionPlanError("R2 protocol lacks authorization metadata")
    required_false = (
        "scientific_training",
        "inner_model_selection",
        "outer_fold_inference",
        "external_test_mount_or_read",
        "external_test_schema_access",
        "external_test_inference",
    )
    if authorization.get("source_implementation") is not True:
        raise R2ExecutionPlanError("source implementation is not authorized")
    if authorization.get("synthetic_and_train_only_engineering_tests") is not True:
        raise R2ExecutionPlanError("train-only engineering tests are not authorized")
    for name in required_false:
        if authorization.get(name) is not False:
            raise R2ExecutionPlanError(
                f"prospective protocol must keep {name} explicitly false"
            )
    firewall = protocol.get("test_firewall")
    if not isinstance(firewall, Mapping):
        raise R2ExecutionPlanError("R2 protocol lacks the test firewall")
    expected_firewall = {
        "current_state": "sealed",
        "access_before_final_freeze": False,
        "schema_inspection_before_final_freeze": False,
        "performance_scoring_before_final_freeze": False,
    }
    for name, expected in expected_firewall.items():
        if firewall.get(name) != expected:
            raise R2ExecutionPlanError(
                f"test firewall field {name} differs from the frozen value"
            )


def _validate_monomer_contract(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    encoder = protocol.get("frozen_monomer_encoder")
    if not isinstance(encoder, Mapping):
        raise R2ExecutionPlanError("R2 protocol lacks the monomer encoder")
    if encoder.get("antibody_and_antigen_parameters_shared") is not False:
        raise R2ExecutionPlanError("monomer encoders must not share parameters")
    if encoder.get("geometry_feature_mode") != "distance_direction":
        raise R2ExecutionPlanError("monomer geometry must be distance_direction")
    confirmation = encoder.get("confirmation")
    if not isinstance(confirmation, Mapping):
        raise R2ExecutionPlanError("R2 protocol lacks monomer confirmation")
    if tuple(confirmation.get("seeds", ())) != FROZEN_SEEDS:
        raise R2ExecutionPlanError("monomer confirmation seeds differ")
    if confirmation.get("checkpoint_rule") != "fixed_final_epoch_20":
        raise R2ExecutionPlanError("monomer checkpoint rule differs")
    if confirmation.get("early_stopping") is not False:
        raise R2ExecutionPlanError("monomer early stopping must be disabled")
    if confirmation.get("validation_checkpoint_selection") is not False:
        raise R2ExecutionPlanError(
            "monomer validation checkpoint selection must be disabled"
        )
    if tuple(confirmation.get("parameter_matched_controls", ())) != (
        "dense_sequence",
        "topology_only",
        "distance",
    ):
        raise R2ExecutionPlanError("monomer controls differ")
    expected_scalars = {
        "epochs": 20,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "batch_size_per_gpu": 4,
        "precision": "bf16",
    }
    for name, expected in expected_scalars.items():
        if confirmation.get(name) != expected:
            raise R2ExecutionPlanError(
                f"monomer confirmation field {name} differs"
            )
    return confirmation


def _validate_candidate_ladder(
    protocol: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw_ladder = protocol.get("candidate_ladder")
    if not isinstance(raw_ladder, list) or len(raw_ladder) != 3:
        raise R2ExecutionPlanError("candidate ladder must contain three entries")
    ladder: dict[str, Mapping[str, Any]] = {}
    for entry in raw_ladder:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            raise R2ExecutionPlanError("candidate ladder entry is malformed")
        identifier = entry["id"]
        if identifier in ladder:
            raise R2ExecutionPlanError("candidate ladder contains duplicate IDs")
        ladder[identifier] = entry
    if tuple(ladder) != INNER_CANDIDATES:
        raise R2ExecutionPlanError("candidate ladder order or IDs differ")
    expected = {
        "D0": (58, 1024, "fixed_final_epoch_58"),
        "R1": (40, 32, "fixed_final_epoch_40"),
        "R2": (40, 32, "fixed_final_epoch_40"),
    }
    for identifier, (epochs, batch, checkpoint_rule) in expected.items():
        entry = ladder[identifier]
        if (
            entry.get("epochs"),
            entry.get("global_batch_size"),
            entry.get("checkpoint_rule"),
        ) != (epochs, batch, checkpoint_rule):
            raise R2ExecutionPlanError(
                f"candidate {identifier} budget or checkpoint rule differs"
            )
        if entry.get("early_stopping") is not False:
            raise R2ExecutionPlanError(
                f"candidate {identifier} early stopping must be disabled"
            )
    r2 = ladder["R2"]
    if r2.get("num_pair_update_layers") != 2:
        raise R2ExecutionPlanError("R2 must contain two pair-update layers")
    if tuple(r2.get("pair_update_order", ())) != (
        "antigen_axis_within_each_antibody_row",
        "antibody_axis_within_each_antigen_column",
        "feed_forward",
    ):
        raise R2ExecutionPlanError("R2 pair-update order differs")
    return ladder


def build_train_only_execution_plan(
    protocol: Mapping[str, Any],
    *,
    protocol_file_sha256: str,
) -> dict[str, Any]:
    """Build every preregistered monomer and nested inner run exactly once."""
    if not isinstance(protocol, Mapping):
        raise R2ExecutionPlanError("protocol must be a mapping")
    protocol_file_sha256 = _require_sha256(
        protocol_file_sha256,
        "protocol_file_sha256",
    )
    _validate_protocol_authority(protocol)
    confirmation = _validate_monomer_contract(protocol)
    ladder = _validate_candidate_ladder(protocol)
    if tuple(protocol.get("fixed_seeds", ())) != FROZEN_SEEDS:
        raise R2ExecutionPlanError("global fixed seeds differ")
    nested = protocol.get("nested_evaluation")
    if not isinstance(nested, Mapping):
        raise R2ExecutionPlanError("R2 protocol lacks nested evaluation")
    outer_count = _require_integer(
        nested.get("outer_fold_count"),
        "outer_fold_count",
    )
    inner_per_outer = _require_integer(
        nested.get("dependent_inner_fold_count_per_outer"),
        "dependent_inner_fold_count_per_outer",
    )
    if outer_count != 3 or inner_per_outer != 3:
        raise R2ExecutionPlanError("R2 nested fold cardinality differs")
    if nested.get("test_use") != "one_shot_after_final_freeze":
        raise R2ExecutionPlanError("R2 test-use rule differs")

    monomer_runs = []
    for variant in MONOMER_VARIANTS:
        for seed in FROZEN_SEEDS:
            monomer_runs.append(
                {
                    "run_id": f"monomer/{variant}/seed-{seed}",
                    "variant": variant,
                    "seed": seed,
                    "epochs": confirmation["epochs"],
                    "checkpoint_rule": confirmation["checkpoint_rule"],
                    "early_stopping": False,
                    "validation_checkpoint_selection": False,
                    "optimizer": confirmation["optimizer"],
                    "learning_rate": confirmation["learning_rate"],
                    "weight_decay": confirmation["weight_decay"],
                    "batch_size_per_gpu": confirmation["batch_size_per_gpu"],
                    "precision": confirmation["precision"],
                    "data_scope": "gate0_authorized_train_only",
                }
            )

    inner_runs = []
    for outer_fold in range(outer_count):
        for inner_fold in range(inner_per_outer):
            for candidate_id in INNER_CANDIDATES:
                candidate = ladder[candidate_id]
                for seed in FROZEN_SEEDS:
                    inner_runs.append(
                        {
                            "run_id": (
                                f"inner/o{outer_fold}/i{inner_fold}/"
                                f"{candidate_id}/seed-{seed}"
                            ),
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "candidate_id": candidate_id,
                            "seed": seed,
                            "epochs": candidate["epochs"],
                            "global_batch_size": candidate["global_batch_size"],
                            "checkpoint_rule": candidate["checkpoint_rule"],
                            "early_stopping": False,
                            "data_scope": (
                                "isolated_corresponding_outer_train_inner_fold"
                            ),
                            "outer_or_test_inference": False,
                        }
                    )

    run_ids = [item["run_id"] for item in monomer_runs + inner_runs]
    if len(run_ids) != len(set(run_ids)):
        raise AssertionError("generated R2 run IDs are not unique")
    plan: dict[str, Any] = {
        "format_version": PLAN_FORMAT,
        "protocol_file_sha256": protocol_file_sha256,
        "status": "prospective_train_only_plan_blocked_at_gate_0",
        "authorization": {
            "scientific_training": False,
            "model_selection": False,
            "outer_fold_inference": False,
            "external_test_mount_or_read": False,
            "external_test_schema_access": False,
            "external_test_inference": False,
        },
        "test_firewall": {
            "state": "sealed",
            "outer_runs_in_plan": 0,
            "external_test_runs_in_plan": 0,
        },
        "fixed_seeds": list(FROZEN_SEEDS),
        "monomer_confirmation_runs": monomer_runs,
        "nested_inner_runs": inner_runs,
        "post_inner_templates": {
            "outer_refit": {
                "materialized": False,
                "requires_frozen_inner_selection": True,
                "seeds": list(FROZEN_SEEDS),
                "checkpoint_rule": "candidate_fixed_final_epoch",
            },
            "external_test": {
                "materialized": False,
                "requires_frozen_final_refit_registry": True,
                "one_shot": True,
            },
        },
        "counts": {
            "monomer_confirmation_runs": len(monomer_runs),
            "nested_inner_runs": len(inner_runs),
            "total_materialized_train_only_runs": (
                len(monomer_runs) + len(inner_runs)
            ),
            "outer_runs": 0,
            "external_test_runs": 0,
        },
    }
    plan["plan_sha256"] = _sha256_json(plan)
    return plan


def validate_train_only_execution_plan(plan: Mapping[str, Any]) -> str:
    """Validate one generated plan and return its authenticated digest."""
    if not isinstance(plan, Mapping):
        raise R2ExecutionPlanError("plan must be a mapping")
    expected_keys = {
        "format_version",
        "protocol_file_sha256",
        "status",
        "authorization",
        "test_firewall",
        "fixed_seeds",
        "monomer_confirmation_runs",
        "nested_inner_runs",
        "post_inner_templates",
        "counts",
        "plan_sha256",
    }
    _require_exact_keys(plan, expected_keys, "plan")
    if plan["format_version"] != PLAN_FORMAT:
        raise R2ExecutionPlanError("unsupported R2 plan format")
    _require_sha256(plan["protocol_file_sha256"], "protocol_file_sha256")
    observed_digest = _require_sha256(plan["plan_sha256"], "plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    expected_digest = _sha256_json(unsigned)
    if observed_digest != expected_digest:
        raise R2ExecutionPlanError("R2 plan digest differs")
    if plan["status"] != "prospective_train_only_plan_blocked_at_gate_0":
        raise R2ExecutionPlanError("R2 plan status differs")
    authorization = _require_exact_keys(
        plan["authorization"],
        {
            "scientific_training",
            "model_selection",
            "outer_fold_inference",
            "external_test_mount_or_read",
            "external_test_schema_access",
            "external_test_inference",
        },
        "plan.authorization",
    )
    if any(value is not False for value in authorization.values()):
        raise R2ExecutionPlanError("prospective R2 plan grants authority")
    if tuple(plan["fixed_seeds"]) != FROZEN_SEEDS:
        raise R2ExecutionPlanError("R2 plan seeds differ")
    expected_counts = {
        "monomer_confirmation_runs": 20,
        "nested_inner_runs": 135,
        "total_materialized_train_only_runs": 155,
        "outer_runs": 0,
        "external_test_runs": 0,
    }
    if plan["counts"] != expected_counts:
        raise R2ExecutionPlanError("R2 plan run counts differ")
    if plan["test_firewall"] != {
        "state": "sealed",
        "outer_runs_in_plan": 0,
        "external_test_runs_in_plan": 0,
    }:
        raise R2ExecutionPlanError("R2 plan test firewall differs")
    if plan["post_inner_templates"] != {
        "outer_refit": {
            "materialized": False,
            "requires_frozen_inner_selection": True,
            "seeds": list(FROZEN_SEEDS),
            "checkpoint_rule": "candidate_fixed_final_epoch",
        },
        "external_test": {
            "materialized": False,
            "requires_frozen_final_refit_registry": True,
            "one_shot": True,
        },
    }:
        raise R2ExecutionPlanError("R2 plan post-inner templates differ")

    monomer_runs = plan["monomer_confirmation_runs"]
    if not isinstance(monomer_runs, list) or len(monomer_runs) != 20:
        raise R2ExecutionPlanError("R2 monomer run inventory differs")
    expected_monomer_keys = {
        "run_id",
        "variant",
        "seed",
        "epochs",
        "checkpoint_rule",
        "early_stopping",
        "validation_checkpoint_selection",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size_per_gpu",
        "precision",
        "data_scope",
    }
    observed_monomer_coordinates: set[tuple[str, int]] = set()
    for index, item in enumerate(monomer_runs):
        item = _require_exact_keys(
            item,
            expected_monomer_keys,
            f"monomer_confirmation_runs[{index}]",
        )
        coordinate = (item["variant"], item["seed"])
        if coordinate not in {
            (variant, seed)
            for variant in MONOMER_VARIANTS
            for seed in FROZEN_SEEDS
        }:
            raise R2ExecutionPlanError("R2 monomer coordinate differs")
        variant, seed = coordinate
        expected_item = {
            "run_id": f"monomer/{variant}/seed-{seed}",
            "variant": variant,
            "seed": seed,
            "epochs": 20,
            "checkpoint_rule": "fixed_final_epoch_20",
            "early_stopping": False,
            "validation_checkpoint_selection": False,
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "batch_size_per_gpu": 4,
            "precision": "bf16",
            "data_scope": "gate0_authorized_train_only",
        }
        if item != expected_item:
            raise R2ExecutionPlanError("R2 monomer run fields differ")
        observed_monomer_coordinates.add(coordinate)
    if len(observed_monomer_coordinates) != 20:
        raise R2ExecutionPlanError("R2 monomer coordinates are duplicated")

    inner_runs = plan["nested_inner_runs"]
    if not isinstance(inner_runs, list) or len(inner_runs) != 135:
        raise R2ExecutionPlanError("R2 inner run inventory differs")
    expected_inner_keys = {
        "run_id",
        "outer_fold",
        "inner_fold",
        "candidate_id",
        "seed",
        "epochs",
        "global_batch_size",
        "checkpoint_rule",
        "early_stopping",
        "data_scope",
        "outer_or_test_inference",
    }
    candidate_budgets = {
        "D0": (58, 1024, "fixed_final_epoch_58"),
        "R1": (40, 32, "fixed_final_epoch_40"),
        "R2": (40, 32, "fixed_final_epoch_40"),
    }
    expected_inner_coordinates = {
        (outer, inner, candidate, seed)
        for outer in range(3)
        for inner in range(3)
        for candidate in INNER_CANDIDATES
        for seed in FROZEN_SEEDS
    }
    observed_inner_coordinates: set[tuple[int, int, str, int]] = set()
    for index, item in enumerate(inner_runs):
        item = _require_exact_keys(
            item,
            expected_inner_keys,
            f"nested_inner_runs[{index}]",
        )
        coordinate = (
            item["outer_fold"],
            item["inner_fold"],
            item["candidate_id"],
            item["seed"],
        )
        if coordinate not in expected_inner_coordinates:
            raise R2ExecutionPlanError("R2 inner coordinate differs")
        if item["outer_or_test_inference"] is not False:
            raise R2ExecutionPlanError(
                "R2 inner plan contains outer/test inference"
            )
        outer_fold, inner_fold, candidate_id, seed = coordinate
        epochs, global_batch, checkpoint_rule = candidate_budgets[candidate_id]
        expected_item = {
            "run_id": (
                f"inner/o{outer_fold}/i{inner_fold}/"
                f"{candidate_id}/seed-{seed}"
            ),
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "candidate_id": candidate_id,
            "seed": seed,
            "epochs": epochs,
            "global_batch_size": global_batch,
            "checkpoint_rule": checkpoint_rule,
            "early_stopping": False,
            "data_scope": "isolated_corresponding_outer_train_inner_fold",
            "outer_or_test_inference": False,
        }
        if item != expected_item:
            raise R2ExecutionPlanError("R2 inner run fields differ")
        observed_inner_coordinates.add(coordinate)
    if observed_inner_coordinates != expected_inner_coordinates:
        raise R2ExecutionPlanError(
            "R2 inner coordinates are missing or duplicated"
        )

    all_runs = list(monomer_runs) + list(inner_runs)
    run_ids = [item.get("run_id") for item in all_runs]
    if (
        any(not isinstance(value, str) or not value for value in run_ids)
        or len(run_ids) != len(set(run_ids))
    ):
        raise R2ExecutionPlanError("R2 plan run IDs are malformed or duplicated")
    if any(
        item.get("outer_or_test_inference") is not False
        for item in inner_runs
    ):
        raise R2ExecutionPlanError("R2 inner plan contains outer/test inference")
    return observed_digest


__all__ = [
    "FROZEN_SEEDS",
    "INNER_CANDIDATES",
    "MONOMER_VARIANTS",
    "PLAN_FORMAT",
    "PROTOCOL_FORMAT",
    "R2ExecutionPlanError",
    "build_train_only_execution_plan",
    "canonical_json_bytes",
    "validate_train_only_execution_plan",
]
