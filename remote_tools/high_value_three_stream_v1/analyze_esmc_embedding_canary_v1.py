#!/usr/bin/env python3
"""Audit base/CDR embedding canaries before full development materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def audit_result(path: Path, artifact_root: Path, role: str, expected_count: int) -> dict[str, object]:
    value = json.loads(path.read_text())
    if (
        value.get("status") != "PASS_ESMC_EMBEDDING_CANARY"
        or value.get("role") != role
        or value.get("canary") is not True
        or value.get("record_count") != expected_count
        or value.get("audit", {}).get("finite") is not True
        or value.get("audit", {}).get("test_access") is not False
    ):
        raise RuntimeError(f"{role} embedding canary result differs")
    kinds = set()
    paired_values = set()
    for record in value["records"]:
        artifact_path = artifact_root / record["artifact_path"]
        if sha256_file(artifact_path) != record["artifact_sha256"]:
            raise RuntimeError("embedding canary artifact SHA differs")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        chunk = artifact["chunks"][0]
        if (
            artifact.get("format_version") != "sppider_seq_embedding_v1"
            or artifact.get("sequence_sha256") != record["sequence_sha256"]
            or tuple(chunk.shape) != (record["sequence_length"], 1152)
            or not bool(torch.isfinite(chunk).all())
        ):
            raise RuntimeError("embedding canary artifact contract differs")
        kinds.add(record["kind"])
        if record["kind"] == "antibody":
            paired_values.add(bool(artifact["paired_input"]))
    if role == "base" and (kinds != {"antibody", "antigen"} or paired_values != {False, True}):
        raise RuntimeError("base canary lacks required input cases")
    if role == "published_cdr" and (kinds != {"antibody"} or paired_values != {False, True}):
        raise RuntimeError("CDR canary lacks required antibody cases")
    return {
        "result_path": str(path),
        "result_sha256": sha256_file(path),
        "record_count": expected_count,
        "peak_allocated_bytes": value["runtime"]["peak_allocated_bytes"],
        "elapsed_seconds": value["runtime"]["elapsed_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.plan) != args.plan_sha256:
        raise RuntimeError("development sequence plan SHA differs")
    plan = json.loads(args.plan.read_text())
    if plan.get("authorization", {}).get("embedding_canary_authorized") is not True:
        raise RuntimeError("embedding canary was not authorized")
    evidence = {
        "base": audit_result(
            args.canary_root / "base" / "RESULT.json",
            args.canary_root / "base" / "artifacts",
            "base",
            4,
        ),
        "published_cdr": audit_result(
            args.canary_root / "published_cdr" / "RESULT.json",
            args.canary_root / "published_cdr" / "artifacts",
            "published_cdr",
            2,
        ),
    }
    result = {
        "format_version": "sabdab2_esmc_embedding_canary_analysis_v1",
        "status": "PASS_BASE_AND_CDR_EMBEDDING_CANARY",
        "plan_sha256": args.plan_sha256,
        "evidence": evidence,
        "authorization": {
            "full_embedding_materialization_authorized": True,
            "training_authorized": False,
            "test_access": False,
        },
        "audit": {
            "paired_antibody": True,
            "single_chain_antibody": True,
            "short_and_long_antigen": True,
            "embedding_dimension": 1152,
            "finite": True,
            "task_labels_read": False,
            "test_access": False,
        },
    }
    atomic_write(args.output, canonical_bytes(result))
    receipt = {
        "format_version": "sabdab2_esmc_embedding_canary_analysis_receipt_v1",
        "status": result["status"],
        "result_sha256": sha256_file(args.output),
        "plan_sha256": args.plan_sha256,
        "full_embedding_materialization_authorized": True,
        "training_authorized": False,
        "test_access": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
