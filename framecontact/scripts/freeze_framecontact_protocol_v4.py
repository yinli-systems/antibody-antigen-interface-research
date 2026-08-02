#!/usr/bin/env python3
"""Seal the memory-safe v4 amendment without weakening v3 authorities."""

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
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite frozen receipt: {arguments.receipt}")
    raw = arguments.protocol.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != "framecontact_primary_protocol_v4":
        raise ValueError("Unknown v4 protocol schema")
    parent = protocol["parent_protocol"]
    if sha256_file(arguments.parent_protocol) != parent.get("sha256"):
        raise ValueError("Parent protocol digest differs")
    if sha256_file(arguments.parent_receipt) != parent.get("receipt_sha256"):
        raise ValueError("Parent receipt digest differs")
    if sha256_file(arguments.smoke_receipt) != protocol["amendment"].get(
        "smoke_receipt_sha256"
    ):
        raise ValueError("Memory smoke digest differs")
    smoke = json.loads(arguments.smoke_receipt.read_text(encoding="utf-8"))
    checks = smoke.get("checks", {})
    if (
        smoke.get("format_version") != "framecontact_sabdab2_gpu_smoke_v1"
        or smoke.get("runtime", {}).get("gpu") != "NVIDIA GeForce RTX 4090"
        or int(smoke.get("antigen_length", -1)) != 2363
        or int(smoke.get("candidate_pair_count", -1)) != 545853
        or checks.get("all_frozen_conditions_executed") is not True
        or checks.get("forward_backward_finite") is not True
        or checks.get("optimizer_step_performed") is not False
        or checks.get("validation_loaded") is not False
        or checks.get("test_loaded") is not False
        or checks.get("performance_metric_computed") is not False
    ):
        raise ValueError("Memory smoke is not eligible for a pre-fitting amendment")
    peaks = {row["experiment"]: int(row["peak_cuda_memory_bytes"]) for row in smoke["results"]}
    if peaks.get("frozen_frame_ipa", 0) < 16_000_000_000:
        raise ValueError("Smoke does not establish the recorded batch-two memory risk")
    if protocol["amendment"]["changes_only"] != {
        "batch_size": {"from": 2, "to": 1},
        "evaluation_batch_size": {"from": 8, "to": 1},
    }:
        raise ValueError("v4 may only amend train/evaluation batch sizes")
    boundary = protocol["evidence_boundary_at_freeze"]
    if any(value is not False for value in boundary.values()):
        raise ValueError("v4 was not frozen before all SAbDab2 prediction access")
    if protocol["training_objective"]["specificity_score_source"] != "interaction_only":
        raise ValueError("v4 must preserve interaction-only specificity")
    if protocol["model_selection"]["seeds"] != [17, 42, 73, 101, 137]:
        raise ValueError("v4 must preserve the five-seed set")
    receipt = {
        "schema_version": "framecontact_protocol_freeze_receipt_v4",
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(canonical_bytes(protocol)).hexdigest(),
        "parent_protocol_sha256": parent["sha256"],
        "parent_receipt_sha256": parent["receipt_sha256"],
        "smoke_receipt_sha256": protocol["amendment"]["smoke_receipt_sha256"],
        "only_batch_sizes_amended": True,
        "preoptimization_memory_evidence_validated": True,
        "sabdab2_pretraining_boundary_validated": True,
        "test_access_boundary_validated": True
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
