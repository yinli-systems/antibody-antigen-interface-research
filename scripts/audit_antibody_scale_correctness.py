#!/usr/bin/env python3
"""Audit a fixed-work larger-model correctness pilot across four backends."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


BACKENDS = ("ddp", "fsdp2", "zero2", "zero3")
LIMITS = {
    "ddp_fsdp2_gradient_norm_relative": 1e-3,
    "ddp_fsdp2_loss_absolute": 5e-4,
    "parameter_square_sum_per_parameter": 1e-10,
    "parameter_sum_per_parameter": 1e-10,
    "zero2_zero3_gradient_norm_relative": 1e-6,
    "zero2_zero3_loss_absolute": 1e-7,
}


def _bindings(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        backend, separator, raw_path = value.partition("=")
        if not separator or backend not in BACKENDS or backend in result:
            raise RuntimeError(f"Invalid {label} binding: {value}")
        result[backend] = Path(raw_path)
    if set(result) != set(BACKENDS):
        raise RuntimeError(f"{label} bindings must cover {BACKENDS}")
    return result


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: str) -> float:
    return float(value.strip().split()[0])


def _telemetry(path: Path) -> dict[str, float | int | str]:
    with path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    active: list[tuple[float, float, float]] = []
    for row in rows:
        normalized = {key.strip(): value for key, value in row.items()}
        memory = _number(normalized["memory.used [MiB]"])
        if memory > 100.0:
            active.append(
                (
                    _number(normalized["utilization.gpu [%]"]),
                    memory,
                    _number(normalized["power.draw [W]"]),
                )
            )
    if not active:
        raise RuntimeError(f"No active GPU telemetry in {path}")
    return {
        "active_samples": len(active),
        "file_sha256": _sha256(path),
        "gpu_utilization_mean_percent": statistics.fmean(row[0] for row in active),
        "maximum_memory_mib": max(row[1] for row in active),
        "mean_power_watts": statistics.fmean(row[2] for row in active),
    }


def _maximum_difference(
    left: dict[str, object], right: dict[str, object], key: str
) -> float:
    return max(
        abs(float(a[key]) - float(b[key]))
        for a, b in zip(left["trace"], right["trace"])
    )


def _maximum_relative_difference(
    left: dict[str, object], right: dict[str, object], key: str
) -> float:
    return max(
        abs(float(a[key]) - float(b[key])) / max(abs(float(a[key])), 1e-30)
        for a, b in zip(left["trace"], right["trace"])
    )


def _comparison(
    left: dict[str, object], right: dict[str, object]
) -> dict[str, float]:
    parameter_count = int(left["parameter_moments"]["count"])
    if parameter_count != int(right["parameter_moments"]["count"]):
        raise RuntimeError("Parameter counts differ across compared backends")
    return {
        "gradient_norm_relative_maximum": _maximum_relative_difference(
            left, right, "gradient_norm"
        ),
        "loss_absolute_maximum": _maximum_difference(left, right, "loss"),
        "parameter_square_sum_difference_per_parameter": abs(
            float(left["parameter_moments"]["square_sum"])
            - float(right["parameter_moments"]["square_sum"])
        )
        / parameter_count,
        "parameter_sum_difference_per_parameter": abs(
            float(left["parameter_moments"]["sum"])
            - float(right["parameter_moments"]["sum"])
        )
        / parameter_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", default=[])
    parser.add_argument("--telemetry", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_paths = _bindings(args.receipt, "receipt")
    telemetry_paths = _bindings(args.telemetry, "telemetry")
    receipts = {backend: _load(path) for backend, path in receipt_paths.items()}
    for backend, receipt in receipts.items():
        if (
            receipt["backend"] != backend
            or receipt["preset"] != "abf_300m"
            or receipt["world_size"] != 8
            or len(receipt["trace"]) != 3
            or receipt["parameter_moments"]["count"] != 308_365_312
        ):
            raise RuntimeError(f"Invalid 300M receipt for {backend}")
    keys = ("consumed_residue_tokens", "residue_tokens", "selected_targets", "update")
    reference = receipts["ddp"]["trace"]
    membership_equal = all(
        all(
            all(left[key] == right[key] for key in keys)
            for left, right in zip(reference, receipt["trace"])
        )
        for receipt in receipts.values()
    )
    comparisons = {
        "ddp_fsdp2": _comparison(receipts["ddp"], receipts["fsdp2"]),
        "zero2_zero3": _comparison(receipts["zero2"], receipts["zero3"]),
    }
    common_parameter_pass = all(
        comparison["parameter_square_sum_difference_per_parameter"]
        <= LIMITS["parameter_square_sum_per_parameter"]
        and comparison["parameter_sum_difference_per_parameter"]
        <= LIMITS["parameter_sum_per_parameter"]
        for comparison in comparisons.values()
    )
    numerical_pass = (
        membership_equal
        and common_parameter_pass
        and comparisons["ddp_fsdp2"]["gradient_norm_relative_maximum"]
        <= LIMITS["ddp_fsdp2_gradient_norm_relative"]
        and comparisons["ddp_fsdp2"]["loss_absolute_maximum"]
        <= LIMITS["ddp_fsdp2_loss_absolute"]
        and comparisons["zero2_zero3"]["gradient_norm_relative_maximum"]
        <= LIMITS["zero2_zero3_gradient_norm_relative"]
        and comparisons["zero2_zero3"]["loss_absolute_maximum"]
        <= LIMITS["zero2_zero3_loss_absolute"]
    )
    payload = {
        "authorization": {
            "benchmark_300m": numerical_pass,
            "pilot_650m": False,
            "production_pretraining": False,
            "scale_32_gpu": False,
            "scale_64_gpu": False,
        },
        "claim_boundary": "three-update 300M eight-GPU engineering correctness only; thresholds were fixed for this post-pilot audit and are not a scientific promotion rule",
        "comparisons": comparisons,
        "direct_ddp_zero_numerical_parity": False,
        "format_version": "antibody_scale_correctness_v1",
        "gpu_telemetry": {
            backend: _telemetry(telemetry_paths[backend]) for backend in BACKENDS
        },
        "limits": LIMITS,
        "membership_equal": membership_equal,
        "numerical_pass": numerical_pass,
        "parameter_count": 308_365_312,
        "preset": "abf_300m",
        "receipt_sha256": {
            backend: _sha256(receipt_paths[backend]) for backend in BACKENDS
        },
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "world_size": 8,
    }
    if not numerical_pass or not all(
        math.isfinite(value)
        for comparison in comparisons.values()
        for value in comparison.values()
    ):
        raise RuntimeError(f"300M correctness audit failed: {payload}")
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
