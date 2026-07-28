#!/usr/bin/env python3
"""Audit same-membership eager and optimized antibody training candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


CANDIDATES = ("eager", "fused_pinned", "compiled")
EXPECTED_MANIFEST_SHA256 = (
    "0f73ccf7f537d223ca6d94b83f08b40a34c23e557f2a747cc32d798c974450f6"
)
EXPECTED_MODEL_FAMILY_SHA256 = (
    "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
)
EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "c8974cb06350da46ca49ffff61c149061b7e7f46b43c9e686176539bbe223e6b"
)
EXPECTED_OPTIMIZATION = {
    "eager": {
        "compile_blocks": False,
        "ddp_bucket_cap_mb": 25,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": False,
        "length_bucketing": False,
        "pinned_staging": False,
    },
    "fused_pinned": {
        "compile_blocks": False,
        "ddp_bucket_cap_mb": 25,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": True,
        "length_bucketing": False,
        "pinned_staging": True,
    },
    "compiled": {
        "compile_blocks": True,
        "ddp_bucket_cap_mb": 25,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": True,
        "length_bucketing": False,
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


def _summary(receipt: dict[str, object], warmup: int) -> dict[str, float | int]:
    trace = receipt["trace"]
    if len(trace) != 20 or warmup >= len(trace):
        raise RuntimeError("Correctness receipt must contain 20 updates")
    measured = trace[warmup:]
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
        "warmup_updates_excluded": warmup,
    }


def _validate_receipt(name: str, receipt: dict[str, object]) -> None:
    expected = {
        "backend": "ddp",
        "batch_quantum": 128,
        "format_version": "antibody_pretraining_optimized_pilot_v1",
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
    if receipt.get("authorization") != {
        "production_pretraining": False,
        "scientific_claim": False,
    }:
        raise RuntimeError(f"Receipt authorization differs for {name}")
    arguments = receipt.get("launch_argv")
    if not isinstance(arguments, list) or "--micro-batch" not in arguments:
        raise RuntimeError(f"Launch arguments are not bound for {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_bindings = _bindings(args.receipt, "receipt")
    job_ids = _bindings(args.job_id, "job ID")
    paths = {name: Path(value) for name, value in receipt_bindings.items()}
    receipts = {name: _load(path) for name, path in paths.items()}
    protocol = _load(args.efficiency_protocol)
    if protocol.get("format_version") != "antibody_efficiency_gate_v1":
        raise RuntimeError("Unexpected efficiency protocol")
    protocol_sha256 = _sha256(args.efficiency_protocol)
    for name, receipt in receipts.items():
        _validate_receipt(name, receipt)
        if receipt.get("efficiency_protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"Efficiency protocol digest differs for {name}")

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
    summaries = {name: _summary(receipt, 5) for name, receipt in receipts.items()}
    reference = receipts["eager"]
    thresholds = protocol["correctness_gate"]
    comparisons = {}
    for name in CANDIDATES[1:]:
        candidate = receipts[name]
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
        comparisons[name] = {
            "final_gradient_norm_relative_difference": gradient_difference,
            "final_loss_absolute_difference": loss_difference,
            "maximum_parameter_moment_relative_difference": max(
                moment_differences.values()
            ),
            "parameter_moment_relative_differences": moment_differences,
            "passed": (
                loss_difference
                <= thresholds["maximum_final_loss_absolute_difference"]
                and gradient_difference
                <= thresholds["maximum_gradient_norm_relative_difference"]
                and max(moment_differences.values())
                <= thresholds["maximum_parameter_moment_relative_difference"]
            ),
        }
    stable = all(
        summary["p95_over_p50"]
        <= protocol["efficiency_gate"]["maximum_p95_over_p50_step_time"]
        for summary in summaries.values()
    )
    passed = membership_equal and stable and all(
        comparison["passed"] for comparison in comparisons.values()
    )
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
        "claim_boundary": "Twenty-update repeated-shard correctness probe; timing is diagnostic and no scientific or production claim is authorized",
        "comparisons_to_eager": comparisons,
        "efficiency_protocol_sha256": protocol_sha256,
        "format_version": "antibody_efficiency_correctness_v1",
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
        "stability_passed": stable,
        "summaries": summaries,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    if not passed:
        raise RuntimeError(f"Efficiency correctness audit failed: {payload}")
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
