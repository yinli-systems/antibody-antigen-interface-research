#!/usr/bin/env python3
"""Audit same-membership eager and CUDAGraph-disabled compiled candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


CANDIDATES = ("eager", "compiled_no_cudagraph")
EXPECTED_MANIFEST_SHA256 = (
    "0f73ccf7f537d223ca6d94b83f08b40a34c23e557f2a747cc32d798c974450f6"
)
EXPECTED_MODEL_FAMILY_SHA256 = (
    "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
)
EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "c8974cb06350da46ca49ffff61c149061b7e7f46b43c9e686176539bbe223e6b"
)
COMMON_OPTIMIZATION = {
    "ddp_bucket_cap_mb": 25,
    "fsdp_output_dtype": "bfloat16",
    "length_bucketing": False,
}
EXPECTED_OPTIMIZATION = {
    "eager": {
        **COMMON_OPTIMIZATION,
        "bucketed_padding": False,
        "compile_blocks": False,
        "compile_prewarm_buckets": False,
        "disable_compiled_cudagraphs": False,
        "fused_adamw": False,
        "pinned_staging": False,
    },
    "compiled_no_cudagraph": {
        **COMMON_OPTIMIZATION,
        "bucketed_padding": True,
        "compile_blocks": True,
        "compile_prewarm_buckets": True,
        "disable_compiled_cudagraphs": True,
        "fused_adamw": True,
        "pinned_staging": True,
    },
}


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


def _relative_difference(reference: float, candidate: float) -> float:
    return abs(candidate - reference) / max(abs(reference), 1e-12)


def _summary(receipt: dict[str, object]) -> dict[str, float | int]:
    trace = receipt["trace"]
    if len(trace) != 20:
        raise RuntimeError("Correctness receipt must contain 20 updates")
    measured = trace[5:]
    seconds = [float(item["seconds"]) for item in measured]
    if not all(math.isfinite(value) and value > 0.0 for value in seconds):
        raise RuntimeError("Correctness receipt contains invalid timing")
    residue_tokens = sum(int(item["residue_tokens"]) for item in measured)
    p50 = statistics.median(seconds)
    p95 = sorted(seconds)[math.ceil(0.95 * len(seconds)) - 1]
    return {
        "measured_updates": len(measured),
        "p50_step_seconds": p50,
        "p95_over_p50": p95 / p50,
        "residue_tokens": residue_tokens,
        "residue_tokens_per_second": residue_tokens / sum(seconds),
        "warmup_updates_excluded": 5,
    }


def _validate_receipt(
    name: str,
    receipt: dict[str, object],
    expected_length_bucket_boundaries: list[int],
) -> None:
    expected = {
        "backend": "ddp",
        "batch_quantum": 128,
        "format_version": "antibody_pretraining_optimized_pilot_v1",
        "length_bucket_boundaries": expected_length_bucket_boundaries,
        "micro_batch_per_gpu": 16,
        "model_family_sha256": EXPECTED_MODEL_FAMILY_SHA256,
        "optimization": EXPECTED_OPTIMIZATION[name],
        "preset": "abf_650m",
        "shard_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "target_residue_tokens": 16384,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "training_protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
        "world_size": 8,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Receipt fields differ for {name}")
    arguments = receipt.get("launch_argv")
    if not isinstance(arguments, list):
        raise RuntimeError(f"Launch arguments are not bound for {name}")
    switch_present = "--disable-compiled-cudagraphs" in arguments
    if switch_present != (name == "compiled_no_cudagraph"):
        raise RuntimeError(f"CUDAGraph launch switch differs for {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument(
        "--expected-length-bucket-boundaries",
        default="96,128,160,194",
    )
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
    job_ids = _bindings(args.job_id, "job ID")
    paths = {name: Path(value) for name, value in receipt_bindings.items()}
    receipts = {name: _load(path) for name, path in paths.items()}
    protocol_sha256 = _sha256(args.efficiency_protocol)
    protocol = _load(args.efficiency_protocol)
    if protocol.get("format_version") != "antibody_efficiency_gate_v1":
        raise RuntimeError("Unexpected efficiency protocol")
    for name, receipt in receipts.items():
        _validate_receipt(name, receipt, expected_length_bucket_boundaries)
        if receipt.get("efficiency_protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"Efficiency protocol digest differs for {name}")

    membership_keys = (
        "consumed_residue_tokens",
        "residue_tokens",
        "selected_targets",
        "sequence_count",
        "update",
    )
    reference_trace = receipts["eager"]["trace"]
    candidate_trace = receipts["compiled_no_cudagraph"]["trace"]
    membership_equal = (
        receipts["eager"]["membership_sha256"]
        == receipts["compiled_no_cudagraph"]["membership_sha256"]
        and len(reference_trace) == len(candidate_trace)
        and all(
            all(left[key] == right[key] for key in membership_keys)
            for left, right in zip(reference_trace, candidate_trace)
        )
    )
    summaries = {name: _summary(receipt) for name, receipt in receipts.items()}
    reference = receipts["eager"]
    candidate = receipts["compiled_no_cudagraph"]
    thresholds = protocol["correctness_gate"]
    loss_difference = abs(
        float(candidate["trace"][-1]["loss"])
        - float(reference["trace"][-1]["loss"])
    )
    gradient_difference = _relative_difference(
        float(reference["trace"][-1]["gradient_norm"]),
        float(candidate["trace"][-1]["gradient_norm"]),
    )
    moment_differences = {
        key: _relative_difference(
            float(reference["parameter_moments"][key]),
            float(candidate["parameter_moments"][key]),
        )
        for key in ("absolute_maximum", "square_sum", "sum")
    }
    comparison_passed = (
        loss_difference <= thresholds["maximum_final_loss_absolute_difference"]
        and gradient_difference
        <= thresholds["maximum_gradient_norm_relative_difference"]
        and max(moment_differences.values())
        <= thresholds["maximum_parameter_moment_relative_difference"]
    )
    diagnostic_stability_passed = all(
        summary["p95_over_p50"]
        <= protocol["efficiency_gate"]["maximum_p95_over_p50_step_time"]
        for summary in summaries.values()
    )
    passed = membership_equal and comparison_passed
    eager_throughput = summaries["eager"]["residue_tokens_per_second"]
    for summary in summaries.values():
        summary["throughput_relative_to_eager"] = (
            summary["residue_tokens_per_second"] / eager_throughput
        )
    payload = {
        "authorization": {
            "benchmark_500_updates": passed,
            "production_pretraining": False,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
        },
        "claim_boundary": "Twenty-update repeated-shard CUDAGraph ablation; no production-corpus, biological, or state-of-the-art claim",
        "comparison_to_eager": {
            "final_gradient_norm_relative_difference": gradient_difference,
            "final_loss_absolute_difference": loss_difference,
            "maximum_parameter_moment_relative_difference": max(
                moment_differences.values()
            ),
            "parameter_moment_relative_differences": moment_differences,
            "passed": comparison_passed,
        },
        "efficiency_protocol_sha256": protocol_sha256,
        "format_version": "antibody_cudagraph_ablation_correctness_v1",
        "expected_length_bucket_boundaries": expected_length_bucket_boundaries,
        "membership_equal": membership_equal,
        "runs": {
            name: {
                "job_id": job_ids[name],
                "receipt_sha256": _sha256(paths[name]),
                "slurm_state": "COMPLETED",
            }
            for name in CANDIDATES
        },
        "source": {
            "archive_sha256": args.source_archive_sha256,
            "runner_sha256": args.runner_sha256,
        },
        "diagnostic_stability_passed": diagnostic_stability_passed,
        "summaries": summaries,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    if not passed:
        raise RuntimeError(f"CUDAGraph ablation correctness failed: {payload}")
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
