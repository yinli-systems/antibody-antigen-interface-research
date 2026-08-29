#!/usr/bin/env python3
"""Fail-closed audit of the two exact ESM-C backbone runtime canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.authority) != args.authority_sha256:
        raise RuntimeError("runtime authority SHA differs")
    authority = json.loads(args.authority.read_text())
    if authority.get("status") != "FROZEN_BEFORE_RUNTIME_CANARY":
        raise RuntimeError("runtime authority status differs")
    canaries = {}
    for role in ("base", "published_cdr"):
        path = args.canary_root / f"{role}.json"
        value = json.loads(path.read_text())
        expected_checkpoint = authority["models"][role]["sha256"]
        if (
            value.get("status") != "PASS_STRICT_ESMC_600M_RUNTIME_CANARY"
            or value.get("role") != role
            or value.get("checkpoint_sha256") != expected_checkpoint
            or value.get("runtime_source_sha256")
            != authority["sources"]["minimal_runtime"]["sha256"]
            or value.get("state_tensor_count") != 362
            or value.get("parameter_count") != 573632640
            or value.get("embedding_dimension") != 1152
            or value.get("separator_token_id") != 30
            or value.get("audit", {}).get("strict_state_dict_load") is not True
            or value.get("audit", {}).get("finite") is not True
            or value.get("audit", {}).get("test_access") is not False
            or value.get("audit", {}).get("training_started") is not False
        ):
            raise RuntimeError(f"{role} runtime canary differs")
        canaries[role] = {"path": str(path), "sha256": sha256_file(path), **value}
    result = {
        "format_version": "paired_esmc_runtime_canary_analysis_v1",
        "status": "PASS_BASE_AND_PUBLISHED_CDR_RUNTIME_VERIFIED",
        "authority_sha256": args.authority_sha256,
        "canaries": canaries,
        "authorization": {
            "embedding_materialization_authorized": True,
            "training_authorized": False,
            "test_access": False,
        },
        "audit": {
            "same_backbone_architecture": True,
            "same_embedding_dimension": True,
            "same_paired_separator_semantics": True,
            "strict_load_both_models": True,
            "finite_both_models": True,
            "test_access": False,
            "training_started": False,
        },
    }
    atomic_write(args.output, canonical_bytes(result))
    receipt = {
        "format_version": "paired_esmc_runtime_canary_analysis_receipt_v1",
        "status": result["status"],
        "result_sha256": sha256_file(args.output),
        "authority_sha256": args.authority_sha256,
        "embedding_materialization_authorized": True,
        "training_authorized": False,
        "test_access": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
