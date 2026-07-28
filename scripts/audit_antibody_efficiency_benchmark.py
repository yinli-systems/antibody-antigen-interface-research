#!/usr/bin/env python3
"""Audit a 500-update same-membership antibody efficiency benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


CANDIDATES = ("eager", "compiled")
EXPECTED_MANIFEST_SHA256 = (
    "0f73ccf7f537d223ca6d94b83f08b40a34c23e557f2a747cc32d798c974450f6"
)
EXPECTED_MODEL_FAMILY_SHA256 = (
    "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
)
EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "c8974cb06350da46ca49ffff61c149061b7e7f46b43c9e686176539bbe223e6b"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bindings(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or name not in CANDIDATES or name in result or not item:
            raise RuntimeError(f"Invalid {label} binding: {value}")
        result[name] = item
    if set(result) != set(CANDIDATES):
        raise RuntimeError(f"{label} bindings must cover {CANDIDATES}")
    return result


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(receipt: dict[str, object], warmup: int) -> dict[str, float | int]:
    trace = receipt["trace"]
    if len(trace) != 500 or warmup >= len(trace):
        raise RuntimeError("Efficiency benchmark must contain 500 updates")
    measured = trace[warmup:]
    seconds = [float(item["seconds"]) for item in measured]
    if not all(math.isfinite(value) and value > 0.0 for value in seconds):
        raise RuntimeError("Efficiency benchmark contains invalid timing")
    residue_tokens = sum(int(item["residue_tokens"]) for item in measured)
    p50 = _quantile(seconds, 0.5)
    p95 = _quantile(seconds, 0.95)
    return {
        "maximum_step_seconds": max(seconds),
        "measured_updates": len(measured),
        "p50_step_seconds": p50,
        "p95_over_p50": p95 / p50,
        "p95_step_seconds": p95,
        "residue_tokens": residue_tokens,
        "residue_tokens_per_second": residue_tokens / sum(seconds),
        "warmup_updates_excluded": warmup,
    }


def _number(value: str) -> float:
    return float(value.strip().split()[0])


def _telemetry(path: Path) -> dict[str, float | int | str]:
    with path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    active: list[tuple[float, float, float]] = []
    for row in rows:
        normalized = {key.strip(): value for key, value in row.items()}
        utilization = _number(normalized["utilization.gpu [%]"])
        memory = _number(normalized["memory.used [MiB]"])
        power = _number(normalized["power.draw [W]"])
        if memory > 100.0:
            active.append((utilization, memory, power))
    if not active:
        raise RuntimeError(f"No active GPU telemetry in {path}")
    return {
        "active_samples": len(active),
        "file_sha256": _sha256(path),
        "gpu_utilization_mean_percent": statistics.fmean(row[0] for row in active),
        "gpu_utilization_median_percent": statistics.median(row[0] for row in active),
        "maximum_memory_mib": max(row[1] for row in active),
        "mean_power_watts": statistics.fmean(row[2] for row in active),
    }


def _validate_receipt(
    name: str,
    receipt: dict[str, object],
    protocol_sha256: str,
    require_disabled_compiled_cudagraphs: bool,
    expected_length_bucket_boundaries: list[int],
) -> None:
    expected = {
        "backend": "ddp",
        "batch_quantum": 128,
        "efficiency_protocol_sha256": protocol_sha256,
        "format_version": "antibody_pretraining_optimized_pilot_v1",
        "micro_batch_per_gpu": 16,
        "model_family_sha256": EXPECTED_MODEL_FAMILY_SHA256,
        "preset": "abf_650m",
        "shard_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "target_residue_tokens": 16384,
        "training_protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
        "world_size": 8,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Efficiency receipt has invalid frozen fields")
    if receipt.get("test_access") != {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }:
        raise RuntimeError("Efficiency receipt used forbidden test inputs")
    if require_disabled_compiled_cudagraphs:
        common = {
            "ddp_bucket_cap_mb": 25,
            "fsdp_output_dtype": "bfloat16",
            "length_bucketing": False,
        }
        expected_optimization = {
            "eager": {
                **common,
                "bucketed_padding": False,
                "compile_blocks": False,
                "compile_prewarm_buckets": False,
                "disable_compiled_cudagraphs": False,
                "fused_adamw": False,
                "pinned_staging": False,
            },
            "compiled": {
                **common,
                "bucketed_padding": True,
                "compile_blocks": True,
                "compile_prewarm_buckets": True,
                "disable_compiled_cudagraphs": True,
                "fused_adamw": True,
                "pinned_staging": True,
            },
        }
        if receipt.get("optimization") != expected_optimization[name]:
            raise RuntimeError(f"Efficiency optimization differs for {name}")
        if receipt.get("length_bucket_boundaries") != (
            expected_length_bucket_boundaries
        ):
            raise RuntimeError(f"Efficiency length buckets differ for {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", default=[])
    parser.add_argument("--telemetry", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument(
        "--require-disabled-compiled-cudagraphs",
        action="store_true",
    )
    parser.add_argument(
        "--expected-length-bucket-boundaries",
        default="96,128,160,194",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        expected_length_bucket_boundaries = [
            int(value) for value in args.expected_length_bucket_boundaries.split(",")
        ]
    except ValueError as error:
        raise RuntimeError("Expected length buckets must be integers") from error
    if (
        not expected_length_bucket_boundaries
        or sorted(set(expected_length_bucket_boundaries))
        != expected_length_bucket_boundaries
        or expected_length_bucket_boundaries[-1] != 194
    ):
        raise RuntimeError("Expected length buckets must increase to 194")
    receipt_bindings = _bindings(args.receipt, "receipt")
    telemetry_bindings = _bindings(args.telemetry, "telemetry")
    job_ids = _bindings(args.job_id, "job ID")
    receipt_paths = {
        name: Path(value) for name, value in receipt_bindings.items()
    }
    telemetry_paths = {
        name: Path(value) for name, value in telemetry_bindings.items()
    }
    receipts = {name: _load(path) for name, path in receipt_paths.items()}
    protocol = _load(args.efficiency_protocol)
    if protocol.get("format_version") != "antibody_efficiency_gate_v1":
        raise RuntimeError("Unexpected efficiency protocol")
    protocol_sha256 = _sha256(args.efficiency_protocol)
    for name, receipt in receipts.items():
        _validate_receipt(
            name,
            receipt,
            protocol_sha256,
            args.require_disabled_compiled_cudagraphs,
            expected_length_bucket_boundaries,
        )

    keys = (
        "consumed_residue_tokens",
        "residue_tokens",
        "selected_targets",
        "sequence_count",
        "update",
    )
    reference_trace = receipts["eager"]["trace"]
    membership_equal = (
        len({receipt["membership_sha256"] for receipt in receipts.values()}) == 1
        and all(
            len(receipt["trace"]) == len(reference_trace)
            and all(
                all(left[key] == right[key] for key in keys)
                for left, right in zip(reference_trace, receipt["trace"])
            )
            for receipt in receipts.values()
        )
    )
    summaries = {name: _summary(receipt, 50) for name, receipt in receipts.items()}
    telemetry = {
        name: _telemetry(path) for name, path in telemetry_paths.items()
    }
    eager_throughput = summaries["eager"]["residue_tokens_per_second"]
    for name, summary in summaries.items():
        summary["throughput_relative_to_eager"] = (
            summary["residue_tokens_per_second"] / eager_throughput
        )
        summary["maximum_memory_fraction"] = (
            max(
                float(receipts[name]["peak_allocated_memory_mib"]),
                float(telemetry[name]["maximum_memory_mib"]),
            )
            / protocol["efficiency_gate"]["total_gpu_memory_mib"]
        )
    candidate = summaries["compiled"]
    gates = {
        "memory": (
            candidate["maximum_memory_fraction"]
            <= protocol["efficiency_gate"]["maximum_memory_fraction"]
        ),
        "membership": membership_equal,
        "stability": (
            candidate["p95_over_p50"]
            <= protocol["efficiency_gate"]["maximum_p95_over_p50_step_time"]
        ),
        "throughput": (
            candidate["throughput_relative_to_eager"]
            >= protocol["efficiency_gate"][
                "minimum_throughput_relative_to_safe_reference"
            ]
        ),
    }
    passed = all(gates.values())
    payload = {
        "authorization": {
            "production_pretraining": False,
            "recovery_required_before_scale": True,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
        },
        "claim_boundary": "Five-hundred-update repeated-shard eight-GPU systems benchmark; no production-corpus, scientific, or state-of-the-art claim",
        "efficiency_protocol_sha256": protocol_sha256,
        "format_version": "antibody_efficiency_benchmark_v1",
        "gate_passed": passed,
        "gates": gates,
        "gpu_telemetry": telemetry,
        "label": args.label,
        "membership_equal": membership_equal,
        "expected_length_bucket_boundaries": expected_length_bucket_boundaries,
        "required_disabled_compiled_cudagraphs": (
            args.require_disabled_compiled_cudagraphs
        ),
        "runs": {
            name: {
                "job_id": job_ids[name],
                "receipt_sha256": _sha256(receipt_paths[name]),
                "slurm_state": "COMPLETED",
            }
            for name in CANDIDATES
        },
        "source": {
            "archive_sha256": args.source_archive_sha256,
            "runner_sha256": args.runner_sha256,
        },
        "summaries": summaries,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
