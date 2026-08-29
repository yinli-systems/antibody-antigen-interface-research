#!/usr/bin/env python3
"""Fail-closed audit for the repaired Stage-A0 systems canary."""

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
    arguments = parser.parse_args()

    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    result = json.loads(arguments.result.read_text(encoding="utf-8"))
    acceptance = protocol["acceptance"]
    repair = protocol.get("repair", {})
    optimization = result.get("optimization", {})
    ranks = result.get("ranks", [])
    total_memory = min(int(rank["total_memory_bytes"]) for rank in ranks) if ranks else 0
    peak_allocated = max(int(rank["max_memory_allocated_bytes"]) for rank in ranks) if ranks else 0
    memory_fraction = peak_allocated / total_memory if total_memory else math.inf
    content_sha = result.get("checkpoint", {}).get("content_sha256")
    checks = {
        "protocol_version": protocol.get("format_version")
        == "abpairfm_3b_systems_optimization_protocol_v3",
        "repair_parent_stderr": repair.get("failed_stderr_sha256")
        == "1d55df4b2e538ef23fa0f6bfaa054dc51f13da48ed6f6f756252b5285fbf1744",
        "repaired_source": repair.get("repaired_source_aggregate_sha256")
        == "2d741e7aa303f891729179b432a8b3f17a0363f0c3bdb00cbf776df318c018cf",
        "attention_backend": optimization.get("attention_backend")
        == protocol["optimization"]["attention_backend"],
        "attention_batch_vectorized_sdpa": optimization.get("attention_sdpa_calls_per_layer") == 2,
        "checkpoint_content_sha256": isinstance(content_sha, str) and len(content_sha) == 64,
        "input_checkpoint_content_sha256": result.get("input_checkpoint_content_sha256")
        == acceptance["input_checkpoint_content_sha256"],
        "length_bucket_schedule": isinstance(optimization.get("length_bucket_schedule_sha256"), str)
        and len(optimization["length_bucket_schedule_sha256"]) == 64,
        "length_bucketed": optimization.get("length_bucketed_microbatches") is True,
        "memory_fraction": memory_fraction <= acceptance["maximum_peak_allocated_memory_fraction"],
        "mfu": result.get("throughput", {}).get("mfu_6n_estimate", -1)
        >= acceptance["minimum_mfu_6n_estimate"],
        "nonfinite": result.get("loss", {}).get("nonfinite_updates")
        == acceptance["nonfinite_updates"],
        "packed_metrics": optimization.get("scalar_metric_all_reduces_per_update") == 1,
        "status": result.get("status") == acceptance["required_status"],
        "test_access": result.get("test_access") is acceptance["test_access"],
        "world_size": result.get("world_size") == acceptance["world_size"],
    }
    audit = {
        "checks": checks,
        "format_version": "abpairfm_3b_systems_optimization_audit_v3",
        "measurements": {
            "checkpoint_content_sha256": content_sha,
            "mean_padded_tokens": optimization.get("mean_padded_tokens"),
            "mean_query_capacity": optimization.get("mean_query_capacity"),
            "mfu_6n_estimate": result.get("throughput", {}).get("mfu_6n_estimate"),
            "peak_allocated_memory_fraction": memory_fraction,
            "residue_tokens_per_second_mean": result.get("throughput", {}).get("residue_tokens_per_second_mean"),
        },
        "protocol_sha256": sha256(arguments.protocol),
        "result_sha256": sha256(arguments.result),
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    arguments.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
