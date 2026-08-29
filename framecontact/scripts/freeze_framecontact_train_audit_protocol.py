#!/usr/bin/env python3
"""Seal the development-only FrameContact train-audit protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite receipt: {arguments.receipt}")
    raw = arguments.protocol.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != "framecontact_train_audit_protocol_v1":
        raise ValueError("Unknown train-audit protocol schema")
    if protocol.get("data", {}).get("role") != "train":
        raise ValueError("Train audit may read only the train role")
    if protocol.get("data", {}).get("test_role_forbidden") is not True:
        raise ValueError("Test role must remain forbidden")
    if protocol.get("calibration", {}).get("fit_on_train") is not False:
        raise ValueError("Train calibration fitting is forbidden")
    eligible = protocol.get("eligible_checkpoint", {})
    if eligible.get("selection_source") != "validation only":
        raise ValueError("Checkpoint selection must be validation-only")
    if eligible.get("train_metrics_available_before_selection") is not False:
        raise ValueError("Train metrics cannot be visible before model selection")
    implementation = protocol.get("implementation", {})
    repository = arguments.protocol.resolve().parents[1]
    for name in ("evaluator", "test"):
        path = repository / implementation[f"{name}_path"]
        if _sha(path) != implementation[f"{name}_sha256"]:
            raise ValueError(f"{name} SHA-256 differs")
    boundary = protocol.get("evidence_boundary_at_freeze", {})
    if any(value is not False for value in boundary.values()):
        raise ValueError("Train-audit protocol was frozen after forbidden evidence")
    receipt = {
        "schema_version": "framecontact_train_audit_receipt_v1",
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(_canonical(protocol)).hexdigest(),
        "implementation_sha256": {
            "evaluator": implementation["evaluator_sha256"],
            "test": implementation["test_sha256"],
        },
        "validation_only_selection_validated": True,
        "validation_frozen_calibration_validated": True,
        "complete_train_role_validated": True,
        "test_and_external_roles_forbidden": True,
        "write_once_output_validated": True,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(_canonical(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
