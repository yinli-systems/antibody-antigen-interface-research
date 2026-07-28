#!/usr/bin/env python3
"""Audit the fixed-work 300M eight-GPU backend benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audit_antibody_backend_benchmark import (
    BACKENDS,
    _load,
    _named_paths,
    _named_strings,
    _sha256,
    _summary,
    _telemetry,
)


TOTAL_GPU_MEMORY_MIB = 32_607.0
MAXIMUM_MEMORY_FRACTION = 0.75


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
        if (
            receipt["backend"] != backend
            or receipt["preset"] != "abf_300m"
            or receipt["world_size"] != 8
            or receipt["parameter_moments"]["count"] != 308_365_312
        ):
            raise RuntimeError(f"Invalid 300M benchmark receipt for {backend}")
    keys = ("consumed_residue_tokens", "residue_tokens", "selected_targets", "update")
    reference = receipts["ddp"]["trace"]
    membership_equal = all(
        len(receipt["trace"]) == len(reference)
        and all(
            all(left[key] == right[key] for key in keys)
            for left, right in zip(reference, receipt["trace"])
        )
        for receipt in receipts.values()
    )
    summaries = {
        backend: _summary(receipts[backend], 50) for backend in BACKENDS
    }
    selected = max(
        summaries,
        key=lambda backend: summaries[backend]["residue_tokens_per_second"],
    )
    selected_throughput = summaries[selected]["residue_tokens_per_second"]
    telemetry = {
        backend: _telemetry(telemetry_paths[backend]) for backend in BACKENDS
    }
    for backend, summary in summaries.items():
        summary["throughput_relative_to_selected"] = (
            summary["residue_tokens_per_second"] / selected_throughput
        )
        summary["maximum_memory_fraction"] = (
            telemetry[backend]["maximum_memory_mib"] / TOTAL_GPU_MEMORY_MIB
        )
    stable = all(summary["p95_over_p50"] <= 1.20 for summary in summaries.values())
    headroom = all(
        summary["maximum_memory_fraction"] <= MAXIMUM_MEMORY_FRACTION
        for summary in summaries.values()
    )
    passed = membership_equal and stable and headroom
    payload = {
        "authorization": {
            "pilot_650m": passed,
            "production_pretraining": False,
            "scale_32_gpu": False,
            "scale_64_gpu": False,
        },
        "claim_boundary": "500-update 300M eight-GPU systems benchmark on repeated pilot data; no scientific, production-corpus, or strong-scaling claim",
        "format_version": "antibody_scale_benchmark_v1",
        "gpu_telemetry": telemetry,
        "headroom_gate": {
            "maximum_memory_fraction": MAXIMUM_MEMORY_FRACTION,
            "passed": headroom,
            "total_gpu_memory_mib": TOTAL_GPU_MEMORY_MIB,
        },
        "membership_equal": membership_equal,
        "parameter_count": 308_365_312,
        "preset": "abf_300m",
        "runs": {
            backend: {
                "job_id": job_ids[backend],
                "receipt_sha256": _sha256(receipt_paths[backend]),
                "slurm_state": "COMPLETED",
            }
            for backend in BACKENDS
        },
        "selected_backend_for_300m": selected,
        "stability_gate": {"maximum_p95_over_p50": 1.20, "passed": stable},
        "summaries": summaries,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "world_size": 8,
    }
    if not passed:
        raise RuntimeError(f"300M benchmark audit failed: {payload}")
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
