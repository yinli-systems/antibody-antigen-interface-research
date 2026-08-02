#!/usr/bin/env python3
"""Seal the transparent runtime-only v5 amendment before validation access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--failed-stdout", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, required=True)
    parser.add_argument("--oom-stderr", type=Path, required=True)
    parser.add_argument("--safe-stress-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite frozen receipt: {arguments.receipt}")
    raw = arguments.protocol.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != "framecontact_primary_protocol_v5":
        raise ValueError("Unknown v5 protocol schema")
    parent = protocol["parent_protocol"]
    if sha256_file(arguments.parent_protocol) != parent.get("sha256"):
        raise ValueError("Parent protocol digest differs")
    if sha256_file(arguments.parent_receipt) != parent.get("receipt_sha256"):
        raise ValueError("Parent receipt digest differs")
    evidence = protocol["incident_evidence"]
    for field, path in (
        ("discarded_task_stdout_sha256", arguments.failed_stdout),
        ("discarded_task_stderr_sha256", arguments.failed_stderr),
        ("three_way_ipa_oom_stderr_sha256", arguments.oom_stderr),
        ("memory_safe_stress_receipt_sha256", arguments.safe_stress_receipt),
    ):
        if sha256_file(path) != evidence.get(field):
            raise ValueError(f"Incident evidence differs: {field}")
    if arguments.failed_stdout.stat().st_size != 0:
        raise ValueError("Discarded task emitted output before the amendment")
    failed_text = arguments.failed_stderr.read_text(encoding="utf-8")
    oom_text = arguments.oom_stderr.read_text(encoding="utf-8")
    if "each complex must contain both binary classes" not in failed_text:
        raise ValueError("Discarded task failure is not the frozen schema incident")
    if "CUDA out of memory" not in oom_text:
        raise ValueError("Three-way stress does not prove the frozen OOM incident")
    safe = json.loads(arguments.safe_stress_receipt.read_text(encoding="utf-8"))
    if (
        safe.get("offload_saved_tensors") is not True
        or safe.get("experiment") != "frozen_frame_ipa"
        or int(safe.get("antigen_length", -1)) != 2363
        or int(safe.get("peak_cuda_memory_bytes", -1)) != int(
            evidence["memory_safe_peak_cuda_bytes"]
        )
        or safe.get("checks", {}).get("forward_backward_finite") is not True
        or safe.get("checks", {}).get("optimizer_step_performed") is not False
        or safe.get("checks", {}).get("validation_loaded") is not False
        or safe.get("checks", {}).get("test_loaded") is not False
    ):
        raise ValueError("Memory-safe stress receipt is ineligible")
    boundary = protocol["evidence_boundary_at_freeze"]
    for key in (
        "discarded_run_checkpoint_or_result_persisted",
        "sabdab2_antigen_fold_validation_predictions_accessed",
        "sabdab2_antigen_fold_validation_metrics_accessed",
        "sabdab2_antigen_fold_test_predictions_accessed",
        "sabdab2_antigen_fold_test_metrics_accessed",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"v5 boundary is ineligible: {key}")
    if boundary.get("test_manifest_label_derived_positive_count_accessed_for_schema_qc") is not True:
        raise ValueError("v5 must disclose the limited test-label metadata access")
    if protocol["training_objective"]["specificity_score_source"] != "interaction_only":
        raise ValueError("v5 must preserve interaction-only specificity")
    if protocol["implementation"].get("offload_saved_tensors") is not True:
        raise ValueError("v5 does not require the proven memory-safe path")
    receipt = {
        "schema_version": "framecontact_protocol_freeze_receipt_v5",
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(canonical_bytes(protocol)).hexdigest(),
        "parent_protocol_sha256": parent["sha256"],
        "parent_receipt_sha256": parent["receipt_sha256"],
        "incident_evidence_sha256": {
            key: evidence[key]
            for key in (
                "discarded_task_stdout_sha256",
                "discarded_task_stderr_sha256",
                "three_way_ipa_oom_stderr_sha256",
                "memory_safe_stress_receipt_sha256"
            )
        },
        "prevalidation_amendment_validated": True,
        "memory_safe_path_validated": True,
        "one_class_loss_disclosure_validated": True,
        "limited_test_label_metadata_access_disclosed": True,
        "test_prediction_boundary_validated": True
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
