#!/usr/bin/env python3
"""Independently audit a five-arm Dynamo Transformer-region bisection."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_compile_backend_attribution as parent
import audit_antibody_production_batch_efficiency as artifacts


FORMAT_VERSION = "antibody_dynamo_region_bisection_result_v1"
PROTOCOL_FORMAT = "antibody_dynamo_region_bisection_protocol_v1"
SIDECAR_FORMAT = "antibody_dynamo_region_bisection_receipt_v1"
MINIFIER_PROTOCOL_FORMAT = "antibody_compile_accuracy_minifier_protocol_v1"
MINIFIER_AUDIT_FORMAT = "antibody_compile_accuracy_minifier_audit_v1"
RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("dynamo_first_half", "dynamo_first_half_gc_off"),
    ("dynamo_second_half", "dynamo_second_half_gc_off"),
    ("dynamo_full_control", "dynamo_full_control_gc_off"),
    ("safe_eager_r2", "safe_eager"),
)
REFERENCE_RUNS = ("safe_eager_r1", "safe_eager_r2")
ATTRIBUTION_RUNS = (
    "dynamo_first_half",
    "dynamo_second_half",
    "dynamo_full_control",
)
PROMOTABLE_RUNS: tuple[str, ...] = ()
AUTHORIZATION = {
    "aot_minifier": False,
    "configuration_promotion": False,
    "production_pretraining": False,
    "scale_16_gpu": False,
    "scale_32_gpu": False,
    "scientific_claim": False,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}


class DynamoRegionBisectionAuditError(RuntimeError):
    """Reject drifted, mutable, or unauthorized bisection evidence."""


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = artifacts._load(path, canonical=False)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(artifacts._canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise DynamoRegionBisectionAuditError(
            f"Receipt self-hash differs: {path}"
        )
    return value, hashlib.sha256(raw).hexdigest()


def _validate_minifier_predecessor(
    protocol: Mapping[str, Any],
    minifier_protocol_path: Path,
    minifier_audit_path: Path,
) -> dict[str, str]:
    minifier_protocol, protocol_raw = artifacts._load(
        minifier_protocol_path,
        canonical=False,
    )
    minifier_protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    minifier_audit, minifier_audit_file_sha256 = _self_hashed(
        minifier_audit_path
    )
    predecessor = protocol.get("failed_minifier", {})
    if (
        minifier_protocol.get("format_version") != MINIFIER_PROTOCOL_FORMAT
        or minifier_audit.get("format_version") != MINIFIER_AUDIT_FORMAT
        or minifier_audit.get("boundary") != "dynamo"
        or minifier_audit.get("boundary_localized") is not False
        or minifier_audit.get("exact_predecessor_replay") is not True
        or minifier_audit.get("minifier_artifacts") != []
        or minifier_audit.get("next_stage")
        != "dynamo_boundary_not_localized_manual_bisection_required"
        or minifier_audit.get("authorization") != protocol.get("authorization")
        or minifier_audit.get("test_access") != protocol.get("test_access")
        or minifier_audit.get("inputs", {}).get("protocol_file_sha256")
        != minifier_protocol_sha256
        or predecessor.get("protocol_file_sha256")
        != minifier_protocol_sha256
        or predecessor.get("audit_file_sha256")
        != minifier_audit_file_sha256
        or predecessor.get("audit_receipt_sha256")
        != minifier_audit.get("receipt_sha256")
        or protocol.get("inputs", {}).get("minifier_protocol_file_sha256")
        != minifier_protocol_sha256
        or protocol.get("inputs", {}).get("minifier_audit_file_sha256")
        != minifier_audit_file_sha256
    ):
        raise DynamoRegionBisectionAuditError(
            "The failed minifier predecessor differs"
        )
    return {
        "minifier_audit_file_sha256": minifier_audit_file_sha256,
        "minifier_audit_receipt_sha256": minifier_audit["receipt_sha256"],
        "minifier_protocol_file_sha256": minifier_protocol_sha256,
    }


def _localization(result: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = result["comparisons"]
    failed = [
        name
        for name in ("dynamo_first_half", "dynamo_second_half")
        if not comparisons[name]["gates"]["evaluation_noninferiority"]
    ]
    full_failed = not comparisons["dynamo_full_control"]["gates"][
        "evaluation_noninferiority"
    ]
    if not full_failed:
        disposition = "full_control_did_not_reproduce_stop"
        next_regions: list[list[int]] = []
    elif failed == ["dynamo_first_half"]:
        disposition = "bisect_first_half"
        next_regions = [list(range(0, 8)), list(range(8, 16))]
    elif failed == ["dynamo_second_half"]:
        disposition = "bisect_second_half"
        next_regions = [list(range(16, 24)), list(range(24, 32))]
    elif len(failed) == 2:
        disposition = "both_halves_fail_bisect_each_independently"
        next_regions = [
            list(range(0, 8)),
            list(range(8, 16)),
            list(range(16, 24)),
            list(range(24, 32)),
        ]
    else:
        disposition = "cross_half_interaction_required"
        next_regions = []
    return {
        "failed_half_runs": failed,
        "full_control_reproduced": full_failed,
        "next_disposition": disposition,
        "next_registered_regions": next_regions,
    }


def _validate_exact_replay(
    run_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    expected = protocol["expected_replay"]
    receipts = {
        name: artifacts._load(run_root / name / "run/receipt.json")[0]
        for name, _ in RUNS
    }
    membership_sha256 = expected["membership_sha256"]
    if any(
        receipt.get("membership_sha256") != membership_sha256
        or receipt.get("preset") != expected["preset"]
        or receipt.get("run_seed") != expected["run_seed"]
        or receipt.get("world_size") != expected["world_size"]
        for receipt in receipts.values()
    ):
        raise DynamoRegionBisectionAuditError(
            "Exact starting state or membership replay differs"
        )
    full_control = receipts["dynamo_full_control"]
    if (
        full_control.get("final_model_sha256")
        != expected["full_dynamo_final_model_sha256"]
        or full_control.get("final_optimizer_sha256")
        != expected["full_dynamo_final_optimizer_sha256"]
    ):
        raise DynamoRegionBisectionAuditError(
            "The full Dynamo control does not exactly replay the predecessor"
        )
    return {
        "full_dynamo_final_model_sha256": full_control["final_model_sha256"],
        "full_dynamo_final_optimizer_sha256": full_control[
            "final_optimizer_sha256"
        ],
        "membership_sha256": membership_sha256,
        "passed": True,
    }


def audit_bisection(
    *,
    run_root: Path,
    protocol_path: Path,
    minifier_protocol_path: Path,
    minifier_audit_path: Path,
) -> dict[str, Any]:
    """Audit exact replay, fixed half membership, and unchanged gates."""
    protocol, protocol_raw = artifacts._load(protocol_path, canonical=False)
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
        or protocol.get("measurement", {}).get("run_order")
        != [name for name, _ in RUNS]
        or protocol.get("measurement", {}).get("promotable_runs") != []
        or artifacts._sha256(run_root / "protocol.json") != protocol_sha256
        or protocol.get("inputs", {}).get("auditor_sha256")
        != artifacts._sha256(Path(__file__))
    ):
        raise DynamoRegionBisectionAuditError("Bisection protocol differs")
    predecessor = _validate_minifier_predecessor(
        protocol,
        minifier_protocol_path,
        minifier_audit_path,
    )
    saved = {
        "ATTRIBUTION_RUNS": parent.ATTRIBUTION_RUNS,
        "FORMAT_VERSION": parent.FORMAT_VERSION,
        "PROMOTABLE_RUNS": parent.PROMOTABLE_RUNS,
        "PROTOCOL_FORMAT": parent.PROTOCOL_FORMAT,
        "REFERENCE_RUNS": parent.REFERENCE_RUNS,
        "RUNS": parent.RUNS,
        "SIDECAR_FORMAT": parent.SIDECAR_FORMAT,
        "__file__": parent.__file__,
    }
    parent.ATTRIBUTION_RUNS = ATTRIBUTION_RUNS
    parent.FORMAT_VERSION = FORMAT_VERSION
    parent.PROMOTABLE_RUNS = PROMOTABLE_RUNS
    parent.PROTOCOL_FORMAT = PROTOCOL_FORMAT
    parent.REFERENCE_RUNS = REFERENCE_RUNS
    parent.RUNS = RUNS
    parent.SIDECAR_FORMAT = SIDECAR_FORMAT
    parent.__file__ = str(Path(__file__))
    try:
        result = parent.audit_backend_attribution(
            run_root=run_root,
            protocol_path=protocol_path,
        )
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)
    exact_replay = _validate_exact_replay(run_root, protocol)
    for run_name in ATTRIBUTION_RUNS:
        mode = dict(RUNS)[run_name]
        flags = protocol["modes"][mode]
        result["comparisons"][run_name]["block_indices"] = flags[
            "block_indices"
        ]
        result["comparisons"][run_name]["compile_scope"] = flags[
            "compile_scope"
        ]
        result["mode_sidecars"][run_name]["block_indices"] = flags[
            "block_indices"
        ]
        result["mode_sidecars"][run_name]["compile_scope"] = flags[
            "compile_scope"
        ]
    localization = _localization(result)
    result["authorization"] = protocol["authorization"]
    result["candidate_found"] = False
    result["claim_boundary"] = protocol["claim_boundary"]
    result["failed_minifier_evidence"] = predecessor
    result["exact_predecessor_replay"] = exact_replay
    result["localization"] = localization
    result["selected_candidate_for_strict_confirmation"] = None
    result["diagnostic_gates"]["full_control_reproduced"] = localization[
        "full_control_reproduced"
    ]
    result["diagnostic_gates"]["exact_predecessor_replay"] = exact_replay[
        "passed"
    ]
    result["diagnostic_valid"] = all(result["diagnostic_gates"].values())
    return result


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(
        artifacts._canonical_json_bytes(payload)
    ).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(artifacts._canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--minifier-protocol", type=Path, required=True)
    parser.add_argument("--minifier-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_bisection(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
        minifier_protocol_path=arguments.minifier_protocol,
        minifier_audit_path=arguments.minifier_audit,
    )
    _publish(arguments.output, result)
    return 0 if result["diagnostic_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
