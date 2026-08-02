#!/usr/bin/env python3
"""Audit an identical-work eight-GPU benchmark across training backends."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


BACKENDS = ("ddp", "fsdp2", "zero2", "zero3")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _named_paths(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or name not in BACKENDS or name in result:
            raise RuntimeError(f"Invalid {label} binding: {value}")
        result[name] = Path(raw_path)
    if tuple(sorted(result)) != tuple(sorted(BACKENDS)):
        raise RuntimeError(f"{label} bindings must cover {BACKENDS}")
    return result


def _named_strings(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or name not in BACKENDS or name in result or not item:
            raise RuntimeError(f"Invalid {label} binding: {value}")
        result[name] = item
    if tuple(sorted(result)) != tuple(sorted(BACKENDS)):
        raise RuntimeError(f"{label} bindings must cover {BACKENDS}")
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
        raise RuntimeError("Benchmark does not contain the frozen 500 updates")
    measured = trace[warmup:]
    seconds = [float(item["seconds"]) for item in measured]
    if not all(math.isfinite(value) and value > 0.0 for value in seconds):
        raise RuntimeError("Benchmark contains invalid step times")
    residue_tokens = sum(int(item["residue_tokens"]) for item in measured)
    median = _quantile(seconds, 0.5)
    p95 = _quantile(seconds, 0.95)
    return {
        "measured_updates": len(measured),
        "p50_step_seconds": median,
        "p95_over_p50": p95 / median,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", default=[])
    parser.add_argument("--telemetry", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_paths = _named_paths(args.receipt, "receipt")
    telemetry_paths = _named_paths(args.telemetry, "telemetry")
    job_ids = _named_strings(args.job_id, "job ID")
    receipts = {backend: _load(path) for backend, path in receipt_paths.items()}
    for backend, receipt in receipts.items():
        if receipt["backend"] != backend or receipt["world_size"] != 8:
            raise RuntimeError(f"Mislabeled receipt for {backend}")
    reference = receipts["ddp"]["trace"]
    membership_keys = (
        "consumed_residue_tokens",
        "residue_tokens",
        "selected_targets",
        "update",
    )
    membership_equal = all(
        len(receipt["trace"]) == len(reference)
        and all(
            all(left[key] == right[key] for key in membership_keys)
            for left, right in zip(reference, receipt["trace"])
        )
        for receipt in receipts.values()
    )
    summaries = {
        backend: _summary(receipts[backend], 50) for backend in BACKENDS
    }
    ddp_throughput = summaries["ddp"]["residue_tokens_per_second"]
    for summary in summaries.values():
        summary["throughput_relative_to_ddp"] = (
            summary["residue_tokens_per_second"] / ddp_throughput
        )
    selected = max(
        summaries,
        key=lambda backend: summaries[backend]["residue_tokens_per_second"],
    )
    payload = {
        "authorization": {
            "production_pretraining": False,
            "scale_32_gpu": False,
            "scale_64_gpu": False,
        },
        "claim_boundary": "35M eight-GPU systems benchmark on repeated pilot data; no scientific, production-corpus, or strong-scaling claim",
        "format_version": "antibody_backend_benchmark_v2",
        "gpu_telemetry": {
            backend: _telemetry(telemetry_paths[backend]) for backend in BACKENDS
        },
        "membership_equal": membership_equal,
        "runs": {
            backend: {
                "job_id": job_ids[backend],
                "receipt_sha256": _sha256(receipt_paths[backend]),
                "slurm_state": "COMPLETED",
            }
            for backend in BACKENDS
        },
        "selected_backend_for_35m": selected,
        "summaries": summaries,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    stable = all(value["p95_over_p50"] <= 1.20 for value in summaries.values())
    if not membership_equal or not stable:
        raise RuntimeError(f"Benchmark audit failed: {payload}")
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
