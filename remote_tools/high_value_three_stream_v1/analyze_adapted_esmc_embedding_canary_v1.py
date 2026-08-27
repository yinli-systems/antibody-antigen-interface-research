#!/usr/bin/env python3
"""Audit two adapted ESM-C embedding canaries and authorize full development embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


ROLES = ("whole_chain_adapted", "cdr_focused_adapted")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise RuntimeError("adapted embedding protocol SHA differs")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("status") != "FROZEN_AFTER_ADAPTATION_BEFORE_EMBEDDING"
        or protocol.get("authorization", {}).get("canary_embedding_authorized") is not True
        or protocol.get("authorization", {}).get("full_embedding_authorized_before_canary_pass") is not False
        or protocol.get("test_access") is not False
    ):
        raise RuntimeError("adapted canary authority differs")
    per_role = {}
    all_passed = True
    for role in ROLES:
        result_path = args.canary_root / f"{role}.json"
        result = json.loads(result_path.read_text())
        expected_sha = protocol["adapted_antibody_models"][role]["checkpoint_sha256"]
        checks = {
            "status": result.get("status") == "PASS_ADAPTED_ESMC_EMBEDDING_CANARY",
            "protocol": result.get("protocol_sha256") == args.protocol_sha256,
            "role": result.get("role") == role,
            "checkpoint": result.get("checkpoint_sha256") == expected_sha,
            "canary": result.get("canary") is True,
            "record_count": result.get("record_count") == 2,
            "finite": result.get("audit", {}).get("finite") is True,
            "strict_load": result.get("audit", {}).get("strict_model_load") is True,
            "no_task_labels": result.get("audit", {}).get("task_labels_read") is False,
            "no_test": result.get("audit", {}).get("test_access") is False,
            "no_antigen": result.get("audit", {}).get("antigen_embedding_materialized") is False,
        }
        for record in result.get("records", []):
            artifact_path = args.canary_root / role / record["artifact_path"]
            if sha256_file(artifact_path) != record["artifact_sha256"]:
                checks["artifact_sha"] = False
                break
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
            chunk = artifact.get("chunks", [None])[0]
            if (
                artifact.get("model_weight_sha256") != expected_sha
                or artifact.get("embedding_dimension") != 1152
                or not isinstance(chunk, torch.Tensor)
                or tuple(chunk.shape) != (record["sequence_length"], 1152)
                or not bool(torch.isfinite(chunk).all())
            ):
                checks["artifact_contract"] = False
                break
        all_passed = all_passed and all(checks.values())
        per_role[role] = {
            "result_sha256": sha256_file(result_path),
            "checkpoint_sha256": expected_sha,
            "checks": checks,
        }
    result = {
        "format_version": "sabdab2_adapted_esmc_embedding_canary_analysis_v1",
        "status": "PASS_ADAPTED_ESMC_EMBEDDING_CANARY" if all_passed else "FAIL_CLOSED_ADAPTED_ESMC_EMBEDDING_CANARY",
        "protocol_sha256": args.protocol_sha256,
        "per_role": per_role,
        "authorization": {
            "full_development_embedding_materialization_authorized": all_passed,
            "downstream_training_authorized": False,
            "test_access": False,
        },
    }
    atomic_write(args.output, result)
    receipt = {
        "format_version": "sabdab2_adapted_esmc_embedding_canary_receipt_v1",
        "status": result["status"],
        "analysis_sha256": sha256_file(args.output),
        "protocol_sha256": args.protocol_sha256,
        "full_development_embedding_materialization_authorized": all_passed,
        "downstream_training_authorized": False,
        "test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
