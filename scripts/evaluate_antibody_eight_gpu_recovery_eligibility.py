#!/usr/bin/env python3
"""Gate an 8-GPU recovery pilot on four-GPU safety and fidelity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FORMAT_VERSION = "antibody_eight_gpu_recovery_eligibility_v1"
SOURCE_FORMAT = "antibody_four_gpu_efficiency_result_v3"
REQUIRED_GATES = (
    "artifact_integrity",
    "evaluation_noninferiority",
    "membership",
    "memory",
    "replicate_stability",
    "replicate_state",
    "short_horizon_numerical",
    "step_stability",
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _load(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ValueError("audit must be a canonical JSON object")
    payload = dict(value)
    receipt_sha256 = payload.pop("receipt_sha256", None)
    if receipt_sha256 != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("audit receipt hash differs")
    return value, hashlib.sha256(raw).hexdigest()


def evaluate(audit: dict[str, object], *, audit_file_sha256: str) -> dict[str, object]:
    if audit.get("format_version") != SOURCE_FORMAT:
        raise ValueError("unexpected four-GPU audit format")
    if audit.get("authorization") != {
        "configuration_promotion": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }:
        raise ValueError("four-GPU audit authorization boundary differs")
    test_access = audit.get("test_access")
    if not isinstance(test_access, dict) or any(test_access.values()):
        raise ValueError("held-out or external test access is not allowed")
    gates = audit.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("four-GPU audit gates are missing")
    required = {name: gates.get(name) is True for name in REQUIRED_GATES}
    failed_efficiency = sorted(
        name for name, passed in gates.items() if passed is not True
    )
    eligible = all(required.values())
    return {
        "authorization": {
            "eight_gpu_recovery_pilot": eligible,
            "production_pretraining": False,
            "scale_promotion": False,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Eligibility for an eight-GPU safe-eager exact-recovery pilot only; "
            "failed compiled-efficiency gates remain failed and production "
            "pretraining remains unauthorized"
        ),
        "eligible": eligible,
        "failed_efficiency_gates_retained": failed_efficiency,
        "format_version": FORMAT_VERSION,
        "inputs": {
            "four_gpu_audit_file_sha256": audit_file_sha256,
            "four_gpu_audit_receipt_sha256": audit["receipt_sha256"],
        },
        "required_recovery_gates": required,
        "test_access": test_access,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit, audit_file_sha256 = _load(args.audit)
    result = evaluate(audit, audit_file_sha256=audit_file_sha256)
    result["receipt_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(result))
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
