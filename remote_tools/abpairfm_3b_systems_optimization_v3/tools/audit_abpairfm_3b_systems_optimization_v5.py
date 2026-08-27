#!/usr/bin/env python3
"""Fail-closed audit for the frozen Stage-A0 systems canary v5."""

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


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.output.exists():
        raise FileExistsError(arguments.output)

    protocol_sha256 = sha256(arguments.protocol)
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    result = json.loads(arguments.result.read_text(encoding="utf-8"))
    acceptance = protocol["acceptance"]
    expected_batch = protocol["batch_equivalence"]
    expected_hardware = protocol["hardware"]
    expected_optimization = protocol["optimization"]
    expected_training = protocol["training"]
    optimization = result.get("optimization", {})
    ranks = result.get("ranks", [])
    total_memory = min(int(rank["total_memory_bytes"]) for rank in ranks) if ranks else 0
    peak_allocated = max(int(rank["max_memory_allocated_bytes"]) for rank in ranks) if ranks else 0
    memory_fraction = peak_allocated / total_memory if total_memory else math.inf
    content_sha256 = result.get("checkpoint", {}).get("content_sha256")
    expected_global_pairs = (
        expected_batch["microbatch_pairs_per_rank"]
        * expected_batch["accumulation_steps"]
        * acceptance["world_size"]
    )

    checks = {
        "attention_backend": optimization.get("attention_backend")
        == expected_optimization["attention_backend"],
        "attention_calls_per_layer": optimization.get("attention_sdpa_calls_per_layer")
        == expected_optimization["attention_sdpa_calls_per_layer"],
        "batch_equivalence": optimization.get("microbatch_pairs_per_rank")
        == expected_batch["microbatch_pairs_per_rank"]
        and optimization.get("accumulation_steps") == expected_batch["accumulation_steps"]
        and expected_global_pairs == expected_batch["global_pairs_per_update"]
        and expected_global_pairs == expected_batch["production_8gpu_global_pairs_per_update"],
        "checkpoint_content_sha256": is_sha256(content_sha256),
        "device_count": len(ranks) == acceptance["world_size"],
        "device_model": bool(ranks)
        and all(rank.get("device_name") == expected_hardware["allowed_gpu_model"] for rank in ranks),
        "input_checkpoint_content_sha256": result.get("input_checkpoint_content_sha256")
        == acceptance["input_checkpoint_content_sha256"],
        "length_bucket_schedule_sha256": is_sha256(
            optimization.get("length_bucket_schedule_sha256")
        ),
        "length_bucketed_microbatches": optimization.get("length_bucketed_microbatches")
        is expected_optimization["length_bucketed_microbatches"],
        "memory_fraction": memory_fraction
        <= acceptance["maximum_peak_allocated_memory_fraction"],
        "mfu": result.get("throughput", {}).get("mfu_6n_estimate", -1)
        >= acceptance["minimum_mfu_6n_estimate"],
        "nonfinite_updates": result.get("loss", {}).get("nonfinite_updates")
        == acceptance["nonfinite_updates"],
        "protocol_sha256": protocol_sha256 == arguments.protocol_sha256,
        "protocol_version": protocol.get("format_version")
        == "abpairfm_3b_systems_optimization_protocol_v5",
        "scalar_metric_all_reduces_per_update": optimization.get(
            "scalar_metric_all_reduces_per_update"
        )
        == expected_optimization["scalar_metric_all_reduces_per_update"],
        "source_aggregate_sha256": result.get("source_sha256")
        == protocol["repair"]["source_aggregate_sha256"],
        "status": result.get("status") == acceptance["required_status"],
        "test_access": result.get("test_access") is acceptance["test_access"]
        and result.get("training_state", {}).get("test_access") is acceptance["test_access"],
        "updates": optimization.get("updates") == expected_training["updates"],
        "world_size": result.get("world_size") == acceptance["world_size"],
    }
    audit = {
        "checks": checks,
        "format_version": "abpairfm_3b_systems_optimization_audit_v5",
        "measurements": {
            "checkpoint_content_sha256": content_sha256,
            "mean_padded_tokens": optimization.get("mean_padded_tokens"),
            "mean_query_capacity": optimization.get("mean_query_capacity"),
            "mfu_6n_estimate": result.get("throughput", {}).get("mfu_6n_estimate"),
            "peak_allocated_memory_fraction": memory_fraction,
            "residue_tokens_per_second_mean": result.get("throughput", {}).get(
                "residue_tokens_per_second_mean"
            ),
        },
        "production_switch_authorized": False,
        "protocol_sha256": protocol_sha256,
        "result_sha256": sha256(arguments.result),
        "status": "PASS" if all(checks.values()) else "FAIL_CLOSED",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
