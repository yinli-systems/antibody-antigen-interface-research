#!/usr/bin/env python3
"""Audit exact eight-GPU recovery for a confirmed real-OAS candidate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_production_batch_efficiency as artifact_parent
import audit_antibody_production_loader_recovery as parent


FORMAT_VERSION = "antibody_production_loader_recovery_audit_8gpu_v1"
RUN_FORMAT = "antibody_production_loader_recovery_run_v1"
GATE_FORMAT = "antibody_production_loader_recovery_gate_v1"
SIDECAR_FORMAT = "antibody_production_candidate_gate_receipt_v1"
CONFIRMATION_FORMAT = "antibody_compile_candidate_confirmation_result_v1"
AUTHORIZATION = {
    "loader_recovery_gate": True,
    "production_pretraining": False,
    "scale_out": False,
    "scientific_claim": False,
}
CONFIRMATION_AUTHORIZATION = {
    "configuration_promotion": True,
    "production_pretraining": False,
    "scale_16_gpu": False,
    "scale_32_gpu": False,
    "scientific_claim": False,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}


def _self_hashed(path: Path) -> dict[str, Any]:
    value, _ = parent._load(path)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(parent._canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise parent.RecoveryAuditError(f"Receipt self-hash differs: {path}")
    return value


def _load_run_8gpu(root: Path, role: str) -> dict[str, Any]:
    receipt, receipt_file_sha256 = parent._load(root / "receipt.json")
    state, state_file_sha256 = parent._load(root / "trainer_state.json")
    if (
        receipt.get("format_version") != RUN_FORMAT
        or receipt.get("run_role") != role
        or receipt.get("test_access") != TEST_ACCESS
        or receipt.get("authorization") != AUTHORIZATION
        or receipt.get("world_size") != 8
        or state.get("world_size") != 8
        or receipt.get("corpus_open", {}).get("node_count") != 1
        or len(receipt.get("corpus_open", {}).get("rank_hosts", [])) != 8
        or len(set(receipt.get("corpus_open", {}).get("rank_hosts", []))) != 1
    ):
        raise parent.RecoveryAuditError(f"{role} eight-GPU identity differs")
    return {
        "receipt": receipt,
        "receipt_file_sha256": receipt_file_sha256,
        "root": str(root),
        "state": state,
        "state_file_sha256": state_file_sha256,
    }


def _validate_confirmation(
    *,
    protocol: Mapping[str, Any],
    confirmation_protocol_path: Path,
    confirmation_audit_path: Path,
) -> dict[str, str]:
    confirmation_protocol, confirmation_protocol_sha256 = parent._load(
        confirmation_protocol_path
    )
    confirmation_audit = _self_hashed(confirmation_audit_path)
    confirmation_audit_file_sha256 = parent._file_sha256(
        confirmation_audit_path
    )
    selection = protocol.get("candidate_selection", {})
    candidate = confirmation_protocol.get("confirmation_selection", {}).get(
        "selected_candidate"
    )
    controls = confirmation_protocol.get("optimization", {}).get(
        "execution_controls"
    )
    if (
        confirmation_audit.get("format_version") != CONFIRMATION_FORMAT
        or confirmation_audit.get("gate_passed") is not True
        or confirmation_audit.get("authorization") != CONFIRMATION_AUTHORIZATION
        or confirmation_audit.get("test_access") != TEST_ACCESS
        or confirmation_audit.get("confirmed_candidate") != candidate
        or confirmation_audit.get("inputs", {}).get("protocol_file_sha256")
        != confirmation_protocol_sha256
        or selection.get("configuration_promotion_verified") is not True
        or selection.get("confirmation_protocol_file_sha256")
        != confirmation_protocol_sha256
        or selection.get("confirmation_audit_file_sha256")
        != confirmation_audit_file_sha256
        or selection.get("confirmation_audit_receipt_sha256")
        != confirmation_audit.get("receipt_sha256")
        or selection.get("selected_candidate") != candidate
        or selection.get("execution_controls") != controls
    ):
        raise parent.RecoveryAuditError("Candidate promotion evidence differs")
    return {
        "confirmation_audit_file_sha256": confirmation_audit_file_sha256,
        "confirmation_audit_receipt_sha256": confirmation_audit[
            "receipt_sha256"
        ],
        "confirmation_protocol_file_sha256": confirmation_protocol_sha256,
    }


def _candidate_sidecar(
    root: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, str]:
    path = root / "production_candidate_gate_receipt.json"
    value = _self_hashed(path)
    selection = protocol["candidate_selection"]
    controls = selection["execution_controls"]
    reduction = value.get("bf16_reduced_precision_reduction", {})
    python_gc = value.get("python_gc", {})
    if (
        value.get("format_version") != SIDECAR_FORMAT
        or value.get("authorization") != AUTHORIZATION
        or value.get("test_access") != TEST_ACCESS
        or value.get("gate_protocol_sha256") != protocol_sha256
        or value.get("selected_candidate") != selection["selected_candidate"]
        or value.get("execution_controls") != controls
        or value.get("wrapper_sha256") != selection["wrapper_sha256"]
        or value.get("base_receipt_file_sha256")
        != parent._file_sha256(root / "receipt.json")
        or value.get("base_trainer_state_file_sha256")
        != parent._file_sha256(root / "trainer_state.json")
        or reduction
        != {
            "after_restore": True,
            "before": True,
            "during_run": not controls["accuracy_first_bf16_reduction"],
        }
        or python_gc.get("enabled_after_restore") is not True
        or python_gc.get("enabled_during_run")
        is not (not controls["disable_python_gc"])
    ):
        raise parent.RecoveryAuditError(f"Candidate sidecar differs: {root}")
    before = python_gc.get("stats_before")
    after = python_gc.get("stats_after")
    delta = python_gc.get("stats_delta")
    if (
        not isinstance(before, list)
        or not isinstance(after, list)
        or not isinstance(delta, list)
        or len(before) != len(after)
        or len(before) != len(delta)
        or any(
            right[key] - left[key] != difference[key]
            for left, right, difference in zip(before, after, delta)
            for key in ("collected", "collections", "uncollectable")
        )
    ):
        raise parent.RecoveryAuditError(f"Candidate GC statistics differ: {root}")
    return {
        "automatic_gc_disabled_during_run": controls["disable_python_gc"],
        "collection_counters_include_explicit_calls": True,
        "file_sha256": parent._file_sha256(path),
        "observed_collection_counter_delta": [
            item["collections"] for item in delta
        ],
        "receipt_sha256": value["receipt_sha256"],
    }


def audit_recovery_8gpu(
    *,
    gate_protocol_path: Path,
    continuous_root: Path,
    stage1_root: Path,
    resumed_root: Path,
    confirmation_protocol_path: Path,
    confirmation_audit_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    """Audit exact recovery, source promotion, and immutable artifacts."""
    protocol, protocol_sha256 = parent._load(gate_protocol_path)
    if (
        protocol.get("format_version") != GATE_FORMAT
        or protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
        or protocol.get("execution", {}).get("world_size") != 8
        or protocol.get("auditor_sha256") != parent._file_sha256(Path(__file__))
    ):
        raise parent.RecoveryAuditError("Eight-GPU gate protocol differs")
    promotion = _validate_confirmation(
        protocol=protocol,
        confirmation_protocol_path=confirmation_protocol_path,
        confirmation_audit_path=confirmation_audit_path,
    )
    original_load_run = parent._load_run
    original_file = parent.__file__
    parent._load_run = _load_run_8gpu
    parent.__file__ = str(Path(__file__))
    try:
        result = parent.audit_recovery(
            gate_protocol_path=gate_protocol_path,
            continuous_root=continuous_root,
            stage1_root=stage1_root,
            resumed_root=resumed_root,
        )
    finally:
        parent._load_run = original_load_run
        parent.__file__ = original_file
    sidecars = {
        "continuous": _candidate_sidecar(
            continuous_root,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        ),
        "stage1": _candidate_sidecar(
            stage1_root,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        ),
        "resumed": _candidate_sidecar(
            resumed_root,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        ),
    }
    artifact_audit = artifact_parent._verify_artifacts(artifact_root)
    if not artifact_audit["manifest_verified"] or artifact_audit["writable_paths"]:
        raise parent.RecoveryAuditError("Recovery artifacts are not immutable")
    result["artifact_audit"] = artifact_audit
    result["authorization"] = {
        "production_pretraining": False,
        "scale_16_gpu": False,
        "scale_32_gpu": False,
        "scientific_claim": False,
    }
    result["candidate_sidecars"] = sidecars
    result["claim_boundary"] = (
        "Exact eight-GPU real-OAS loader and checkpoint recovery for one "
        "confirmed execution candidate only; no production training, scale, "
        "biological, downstream, or SOTA claim"
    )
    result["format_version"] = FORMAT_VERSION
    result["promotion_evidence"] = promotion
    result["world_size"] = 8
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-audit", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_recovery_8gpu(
        gate_protocol_path=arguments.gate_protocol,
        continuous_root=arguments.continuous,
        stage1_root=arguments.stage1,
        resumed_root=arguments.resumed,
        confirmation_protocol_path=arguments.confirmation_protocol,
        confirmation_audit_path=arguments.confirmation_audit,
        artifact_root=arguments.artifact_root,
    )
    parent._publish(arguments.output, result)


if __name__ == "__main__":
    main()
