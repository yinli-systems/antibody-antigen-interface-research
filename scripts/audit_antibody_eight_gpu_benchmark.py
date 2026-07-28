#!/usr/bin/env python3
"""Summarize a fixed-work 500-update DDP and FSDP2 benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", type=Path, required=True)
    parser.add_argument("--fsdp2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ddp = _load(args.ddp)
    fsdp2 = _load(args.fsdp2)
    if ddp["backend"] != "ddp" or fsdp2["backend"] != "fsdp2":
        raise RuntimeError("Benchmark backends are mislabeled")
    if len(ddp["trace"]) != len(fsdp2["trace"]):
        raise RuntimeError("Benchmark trace lengths differ")
    membership_equal = all(
        all(left[key] == right[key] for key in ("consumed_residue_tokens", "residue_tokens", "selected_targets", "update"))
        for left, right in zip(ddp["trace"], fsdp2["trace"])
    )
    summaries = {"ddp": _summary(ddp, 50), "fsdp2": _summary(fsdp2, 50)}
    selected = max(
        summaries,
        key=lambda backend: summaries[backend]["residue_tokens_per_second"],
    )
    payload = {
        "claim_boundary": "35M eight-GPU systems benchmark on repeated pilot data; no scientific or scale-efficiency claim",
        "format_version": "antibody_eight_gpu_benchmark_v1",
        "membership_equal": membership_equal,
        "selected_backend_for_35m": selected,
        "summaries": summaries,
        "test_access": {"downstream_labels_used": False, "sabdab2_heldout_accessed": False},
    }
    if not membership_equal or any(value["p95_over_p50"] > 1.20 for value in summaries.values()):
        raise RuntimeError(f"Benchmark stability gate failed: {payload}")
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
