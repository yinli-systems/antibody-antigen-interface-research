#!/usr/bin/env python3
"""Fail-closed audit for the frozen 3B Stage-A0 8-to-4 GPU canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    acceptance = protocol["acceptance"]
    ranks = result.get("ranks", [])
    total_memory = min(int(rank["total_memory_bytes"]) for rank in ranks) if ranks else 0
    peak_allocated = max(int(rank["max_memory_allocated_bytes"]) for rank in ranks) if ranks else 0
    memory_fraction = peak_allocated / total_memory if total_memory else math.inf
    content_sha = result.get("checkpoint", {}).get("content_sha256")
    checks = {
        "checkpoint_content_sha256": isinstance(content_sha, str) and len(content_sha) == 64,
        "input_checkpoint_content_sha256": result.get("input_checkpoint_content_sha256")
        == acceptance["input_checkpoint_content_sha256"],
        "memory_fraction": memory_fraction <= acceptance["maximum_peak_allocated_memory_fraction"],
        "mfu": result.get("throughput", {}).get("mfu_6n_estimate", -1)
        >= acceptance["minimum_mfu_6n_estimate"],
        "nonfinite": result.get("loss", {}).get("nonfinite_updates")
        == acceptance["nonfinite_updates"],
        "status": result.get("status") == acceptance["required_status"],
        "test_access": result.get("test_access") is acceptance["test_access"],
        "world_size": result.get("world_size") == acceptance["world_size"],
    }
    audit = {
        "checks": checks,
        "format_version": "abpairfm_3b_4gpu_dcp_mfu_canary_audit_v1",
        "measurements": {
            "checkpoint_content_sha256": content_sha,
            "mfu_6n_estimate": result.get("throughput", {}).get("mfu_6n_estimate"),
            "peak_allocated_memory_fraction": memory_fraction,
            "residue_tokens_per_second_mean": result.get("throughput", {}).get(
                "residue_tokens_per_second_mean"
            ),
        },
        "protocol_sha256": sha256(args.protocol),
        "result_sha256": sha256(args.result),
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
