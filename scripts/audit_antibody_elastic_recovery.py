#!/usr/bin/env python3
"""Audit the real-OAS 4-to-2-to-4 elastic DCP correctness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


RUN_FORMAT = "antibody_elastic_recovery_run_v1"
GATE_FORMAT = "antibody_elastic_recovery_gate_v1"
AUDIT_FORMAT = "antibody_elastic_recovery_audit_v1"
AUTHORIZATION = {
    "elastic_recovery_gate": True,
    "production_pretraining": False,
    "scientific_claim": False,
    "test_access": False,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
COORDINATE_KEYS = (
    "consumed_residue_tokens",
    "learning_rate",
    "residue_tokens",
    "selected_targets",
    "sequence_count",
    "update",
)


class ElasticRecoveryAuditError(RuntimeError):
    """Report an invalid artifact or failed elastic numerical gate."""


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _load(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ElasticRecoveryAuditError(f"Evidence path is invalid: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ElasticRecoveryAuditError(
            f"Evidence is not strict ASCII JSON: {path}"
        ) from error
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ElasticRecoveryAuditError(f"Evidence is not canonical: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _scaled_error(
    observed: float,
    expected: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    if not math.isfinite(observed) or not math.isfinite(expected):
        raise ElasticRecoveryAuditError("Trace contains a non-finite value")
    return abs(observed - expected) / (
        absolute_tolerance + relative_tolerance * abs(expected)
    )


def _load_run(root: Path, run_key: str, protocol_sha256: str) -> dict[str, Any]:
    receipt, receipt_sha256 = _load(root / "receipt.json")
    state, state_sha256 = _load(root / "trainer_state.json")
    elastic = receipt.get("elastic_resume")
    if (
        receipt.get("format_version") != RUN_FORMAT
        or receipt.get("authorization") != AUTHORIZATION
        or receipt.get("test_access") != TEST_ACCESS
        or receipt.get("gate_protocol_sha256") != protocol_sha256
        or not isinstance(elastic, dict)
        or elastic.get("enabled") is not True
        or elastic.get("run_key") != run_key
        or receipt.get("world_size") != state.get("world_size")
        or receipt.get("implementation") is None
        or receipt.get("corpus_open", {}).get(
            "array_file_hashes_verified_during_gate_freeze"
        )
        is not True
        or receipt.get("corpus_open", {}).get("stage_revalidation")
        != "immutable_tree_manifest_size_dtype_shape"
    ):
        raise ElasticRecoveryAuditError(f"Elastic run identity differs: {run_key}")
    return {
        "receipt": receipt,
        "receipt_file_sha256": receipt_sha256,
        "root": str(root),
        "state": state,
        "state_file_sha256": state_sha256,
    }


def _validate_checkpoint_link(
    source: Mapping[str, Any],
    destination: Mapping[str, Any],
) -> None:
    source_receipt = source["receipt"]
    source_state = source["state"]
    destination_receipt = destination["receipt"]
    elastic = destination_receipt["elastic_resume"]
    required_hashes = (
        source_receipt.get("checkpoint_model_sha256"),
        source_receipt.get("checkpoint_optimizer_sha256"),
        source_receipt.get("checkpoint_rng_manifest_sha256"),
        source_receipt.get("final_model_sha256"),
        source_receipt.get("final_optimizer_sha256"),
        source_receipt.get("final_semantic_optimizer_sha256"),
        destination_receipt.get("resume_payload_model_sha256"),
        destination_receipt.get("resume_payload_optimizer_sha256"),
        destination_receipt.get("resume_loaded_model_sha256"),
        destination_receipt.get("resume_loaded_optimizer_sha256"),
        destination_receipt.get("resume_loaded_semantic_optimizer_sha256"),
        destination_receipt.get("resume_rng_manifest_sha256"),
    )
    if not all(_is_sha256(value) for value in required_hashes):
        raise ElasticRecoveryAuditError("Source checkpoint digests are invalid")
    if (
        source_receipt.get("checkpoint_model_sha256")
        != source_receipt.get("final_model_sha256")
        or source_receipt.get("checkpoint_state_hash_domains")
        != {
            "model": "torch_distributed_checkpoint_canonical_state_dict_v1",
            "optimizer": (
                "torch_distributed_checkpoint_canonical_optimizer_state_dict_v1"
            ),
            "rng": "authenticated_rank_rng_manifest_v1",
        }
        or source_receipt.get("final_state_hash_domains")
        != {
            "model": "unwrapped_module_state_dict_v1",
            "optimizer": "native_optimizer_state_dict_v1",
        }
        or destination_receipt.get("resume_payload_model_sha256")
        != source_receipt.get("checkpoint_model_sha256")
        or destination_receipt.get("resume_payload_optimizer_sha256")
        != source_receipt.get("checkpoint_optimizer_sha256")
        or destination_receipt.get("resume_loaded_model_sha256")
        != source_receipt.get("final_model_sha256")
        or destination_receipt.get("resume_loaded_optimizer_sha256")
        != source_receipt.get("final_optimizer_sha256")
        or destination_receipt.get("resume_loaded_semantic_optimizer_sha256")
        != source_receipt.get("final_semantic_optimizer_sha256")
        or destination_receipt.get("resume_rng_manifest_sha256")
        != source_receipt.get("checkpoint_rng_manifest_sha256")
        or elastic.get("source_world_size") != source_state.get("world_size")
        or elastic.get("source_semantic_state_sha256")
        != source_receipt.get("elastic_resume", {}).get(
            "destination_semantic_state_sha256"
        )
        or elastic.get("cross_world") is not True
        or elastic.get("rng_policy")
        != "authenticate_source_then_topology_reseed"
    ):
        raise ElasticRecoveryAuditError("Elastic checkpoint chain differs")
    authentication = elastic.get("source_rng_authentication")
    if (
        not isinstance(authentication, dict)
        or authentication.get("manifest_sha256")
        != source_receipt.get("checkpoint_rng_manifest_sha256")
        or authentication.get("source_world_size")
        != source_state.get("world_size")
        or len(authentication.get("source_rank_state_sha256", []))
        != source_state.get("world_size")
    ):
        raise ElasticRecoveryAuditError("Source RNG authentication differs")


def _validate_optimizer_diagnostic(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    continuous_root: Path,
    stage3_root: Path,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = _load(path)
    unsigned = dict(value)
    recorded = unsigned.pop("diagnostic_sha256", None)
    expected_roots = {
        "left": str((continuous_root / "checkpoint").resolve()),
        "right": str((stage3_root / "checkpoint").resolve()),
    }
    if (
        recorded != hashlib.sha256(_canonical(unsigned)).hexdigest()
        or value.get("format_version")
        != "antibody_optimizer_state_diagnostic_v1"
        or value.get("implementation_sha256")
        != protocol.get("optimizer_diagnostic_sha256")
        or value.get("authorization")
        != {
            "diagnostic_only": True,
            "production_pretraining": False,
            "scientific_claim": False,
        }
        or value.get("test_access") != TEST_ACCESS
        or value.get("checkpoint_roots") != expected_roots
        or value.get("model")
        != {
            "different_tensors": 0,
            "exact": True,
            "tensor_differences": {},
        }
    ):
        raise ElasticRecoveryAuditError("Optimizer diagnostic identity differs")
    optimizer = value.get("optimizer")
    if (
        not isinstance(optimizer, dict)
        or optimizer.get("param_groups_equal") is not True
        or optimizer.get("non_tensor_differences") != []
        or int(optimizer.get("named_parameter_count", 0)) <= 0
    ):
        raise ElasticRecoveryAuditError("Optimizer structural state differs")
    inventories = value.get("checkpoint_inventories")
    if not isinstance(inventories, dict) or set(inventories) != {"left", "right"}:
        raise ElasticRecoveryAuditError("Checkpoint inventory evidence differs")
    for inventory in inventories.values():
        files = inventory.get("files") if isinstance(inventory, dict) else None
        if (
            not isinstance(files, list)
            or not files
            or inventory.get("inventory_sha256")
            != hashlib.sha256(_canonical(files)).hexdigest()
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or isinstance(item.get("bytes"), bool)
                or not isinstance(item.get("bytes"), int)
                or item["bytes"] < 0
                or not _is_sha256(item.get("sha256"))
                for item in files
            )
        ):
            raise ElasticRecoveryAuditError("Checkpoint inventory is invalid")
    summaries = value.get("field_summaries")
    gate = protocol.get("numerical_gate", {}).get("optimizer_state", {})
    if not isinstance(summaries, dict) or set(summaries) != {
        "exp_avg",
        "exp_avg_sq",
        "step",
    }:
        raise ElasticRecoveryAuditError("Optimizer moment inventory differs")
    step = summaries["step"]
    if (
        not isinstance(step, dict)
        or step.get("different_elements") != 0
        or step.get("different_tensors") != 0
        or step.get("maximum_absolute_difference") != 0.0
        or step.get("maximum_relative_difference") != 0.0
        or step.get("maximum_ulp_distance") != 0
        or int(step.get("tensor_count", 0)) <= 0
        or gate.get("step_exact") is not True
        or gate.get("param_groups_exact") is not True
        or gate.get("terminal_model_exact") is not True
    ):
        raise ElasticRecoveryAuditError("Optimizer step state differs")
    for field in ("exp_avg", "exp_avg_sq"):
        observed = summaries[field]
        threshold = gate.get(field)
        if not isinstance(observed, dict) or not isinstance(threshold, dict):
            raise ElasticRecoveryAuditError("Optimizer moment gate is invalid")
        for key in (
            "maximum_absolute_difference",
            "maximum_relative_difference",
        ):
            value_float = float(observed.get(key, math.inf))
            threshold_float = float(threshold.get(key, -1.0))
            if (
                not math.isfinite(value_float)
                or not math.isfinite(threshold_float)
                or value_float < 0.0
                or threshold_float < 0.0
                or value_float > threshold_float
            ):
                raise ElasticRecoveryAuditError(
                    f"Optimizer {field} {key} gate failed"
                )
        observed_ulp = observed.get("maximum_ulp_distance")
        threshold_ulp = threshold.get("maximum_ulp_distance")
        if (
            isinstance(observed_ulp, bool)
            or not isinstance(observed_ulp, int)
            or observed_ulp < 0
            or isinstance(threshold_ulp, bool)
            or not isinstance(threshold_ulp, int)
            or threshold_ulp < 0
            or observed_ulp > threshold_ulp
        ):
            raise ElasticRecoveryAuditError(f"Optimizer {field} ULP gate failed")
    return value, file_sha256


def audit(
    *,
    gate_protocol: Path,
    continuous_root: Path,
    stage1_root: Path,
    stage2_root: Path,
    stage3_root: Path,
    optimizer_diagnostic: Path,
) -> dict[str, Any]:
    """Validate evidence, transition chain, coordinates, and BF16 trace parity."""
    protocol, protocol_sha256 = _load(gate_protocol)
    auditor_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        protocol.get("format_version") != GATE_FORMAT
        or protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
        or protocol.get("auditor_sha256") != auditor_sha256
    ):
        raise ElasticRecoveryAuditError("Elastic gate protocol identity differs")
    roots = {
        "continuous_4": continuous_root,
        "stage1_4": stage1_root,
        "stage2_2": stage2_root,
        "stage3_4": stage3_root,
    }
    runs = {
        key: _load_run(root, key, protocol_sha256)
        for key, root in roots.items()
    }
    expected_world_sizes = {
        "continuous_4": 4,
        "stage1_4": 4,
        "stage2_2": 2,
        "stage3_4": 4,
    }
    for key, expected_world_size in expected_world_sizes.items():
        if runs[key]["state"].get("world_size") != expected_world_size:
            raise ElasticRecoveryAuditError(f"World size differs for {key}")
        if runs[key]["receipt"].get("implementation") != protocol.get(
            "implementation"
        ):
            raise ElasticRecoveryAuditError(f"Implementation differs for {key}")
    _validate_checkpoint_link(runs["stage1_4"], runs["stage2_2"])
    _validate_checkpoint_link(runs["stage2_2"], runs["stage3_4"])

    continuous_trace = runs["continuous_4"]["receipt"].get("trace")
    segmented_trace = []
    for key in ("stage1_4", "stage2_2", "stage3_4"):
        trace = runs[key]["receipt"].get("trace")
        if not isinstance(trace, list):
            raise ElasticRecoveryAuditError(f"Trace is absent for {key}")
        segmented_trace.extend(trace)
    if not isinstance(continuous_trace, list) or len(continuous_trace) != 4:
        raise ElasticRecoveryAuditError("Continuous reference trace differs")
    if len(segmented_trace) != len(continuous_trace):
        raise ElasticRecoveryAuditError("Segmented trace length differs")
    gate = protocol.get("numerical_gate", {})
    absolute_tolerance = float(gate.get("absolute_tolerance", 0.0))
    relative_tolerance = float(gate.get("relative_tolerance", 0.0))
    maximum_loss_scaled_error = 0.0
    maximum_gradient_scaled_error = 0.0
    for index, (expected, observed) in enumerate(
        zip(continuous_trace, segmented_trace)
    ):
        if any(observed.get(key) != expected.get(key) for key in COORDINATE_KEYS):
            raise ElasticRecoveryAuditError(
                f"Global training coordinate differs at update {index + 1}"
            )
        loss_error = _scaled_error(
            float(observed["loss"]),
            float(expected["loss"]),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        gradient_error = _scaled_error(
            float(observed["gradient_norm"]),
            float(expected["gradient_norm"]),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        maximum_loss_scaled_error = max(maximum_loss_scaled_error, loss_error)
        maximum_gradient_scaled_error = max(
            maximum_gradient_scaled_error,
            gradient_error,
        )
        if index < 2 and (loss_error != 0.0 or gradient_error != 0.0):
            raise ElasticRecoveryAuditError("Same-world trace prefix is not exact")
        if loss_error > 1.0 or gradient_error > 1.0:
            raise ElasticRecoveryAuditError(
                f"Cross-world numerical gate failed at update {index + 1}"
            )

    terminal_fields = (
        "consumed_residue_tokens",
        "next_update",
        "sample_cursor",
        "target_residue_tokens",
        "training_universe_count",
    )
    continuous_state = runs["continuous_4"]["state"]
    segmented_state = runs["stage3_4"]["state"]
    if any(
        continuous_state.get(field) != segmented_state.get(field)
        for field in terminal_fields
    ):
        raise ElasticRecoveryAuditError("Terminal training coordinate differs")
    optimizer_diagnostic_value, optimizer_diagnostic_file_sha256 = (
        _validate_optimizer_diagnostic(
            optimizer_diagnostic,
            protocol=protocol,
            continuous_root=continuous_root,
            stage3_root=stage3_root,
        )
    )
    terminal_model_equal = (
        _is_sha256(
            runs["continuous_4"]["receipt"].get("final_model_sha256")
        )
        and runs["continuous_4"]["receipt"].get("final_model_sha256")
        == runs["stage3_4"]["receipt"].get("final_model_sha256")
    )
    terminal_semantic_optimizer_equal = (
        _is_sha256(
            runs["continuous_4"]["receipt"].get(
                "final_semantic_optimizer_sha256"
            )
        )
        and runs["continuous_4"]["receipt"].get(
            "final_semantic_optimizer_sha256"
        )
        == runs["stage3_4"]["receipt"].get(
            "final_semantic_optimizer_sha256"
        )
    )
    if not terminal_model_equal:
        raise ElasticRecoveryAuditError("Terminal model state differs")
    return {
        "auditor_sha256": auditor_sha256,
        "authorization": {
            "production_pretraining": False,
            "scientific_claim": False,
            "test_access": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "evidence": {
            key: {
                "receipt_file_sha256": value["receipt_file_sha256"],
                "root": value["root"],
                "state_file_sha256": value["state_file_sha256"],
            }
            for key, value in runs.items()
        },
        "optimizer_diagnostic": {
            "file_sha256": optimizer_diagnostic_file_sha256,
            "path": str(optimizer_diagnostic),
        },
        "format_version": AUDIT_FORMAT,
        "gate_protocol_file_sha256": protocol_sha256,
        "gates": {
            "checkpoint_chain_authenticated": True,
            "global_coordinates_exact": True,
            "same_world_prefix_exact": True,
            "source_rng_inventories_authenticated": True,
            "terminal_coordinate_exact": True,
            "terminal_model_exact": terminal_model_equal,
            "terminal_optimizer_moments_within_gate": True,
            "trace_numerical_noninferiority": True,
        },
        "numerical_result": {
            "absolute_tolerance": absolute_tolerance,
            "maximum_gradient_scaled_error": maximum_gradient_scaled_error,
            "maximum_loss_scaled_error": maximum_loss_scaled_error,
            "optimizer_field_summaries": optimizer_diagnostic_value[
                "field_summaries"
            ],
            "relative_tolerance": relative_tolerance,
            "terminal_semantic_optimizer_exact": (
                terminal_semantic_optimizer_equal
            ),
        },
        "status": "passed",
        "test_access": TEST_ACCESS,
        "transitions": [[4, 2], [2, 4]],
    }


def _publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ElasticRecoveryAuditError("Audit output must be create-only")
    value = dict(payload)
    value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(value))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--stage3", type=Path, required=True)
    parser.add_argument("--optimizer-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _publish(
        args.output,
        audit(
            gate_protocol=args.gate_protocol,
            continuous_root=args.continuous,
            stage1_root=args.stage1,
            stage2_root=args.stage2,
            stage3_root=args.stage3,
            optimizer_diagnostic=args.optimizer_diagnostic,
        ),
    )


if __name__ == "__main__":
    main()
