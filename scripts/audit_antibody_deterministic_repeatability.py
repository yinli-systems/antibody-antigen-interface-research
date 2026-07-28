#!/usr/bin/env python3
"""Audit deterministic antibody correctness across compiled execution modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


CANDIDATES = ("eager", "compiled_cudagraph", "compiled_no_cudagraph")
EXPECTED_BUCKETS = [96, 128, 144, 194]
EXPECTED_MANIFEST_SHA256 = (
    "0f73ccf7f537d223ca6d94b83f08b40a34c23e557f2a747cc32d798c974450f6"
)
EXPECTED_MODEL_FAMILY_SHA256 = (
    "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
)
EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "c8974cb06350da46ca49ffff61c149061b7e7f46b43c9e686176539bbe223e6b"
)
TRACE_KEYS = (
    "consumed_residue_tokens",
    "gradient_norm",
    "learning_rate",
    "loss",
    "micro_steps",
    "residue_tokens",
    "selected_targets",
    "sequence_count",
    "update",
)
MEMBERSHIP_KEYS = (
    "consumed_residue_tokens",
    "residue_tokens",
    "selected_targets",
    "sequence_count",
    "update",
)
STATE_COORDINATE_KEYS = (
    "backend",
    "batch_quantum",
    "consumed_residue_tokens",
    "efficiency_protocol_sha256",
    "length_bucket_boundaries",
    "micro_batch_per_gpu",
    "model_family_sha256",
    "next_update",
    "preset",
    "sample_cursor",
    "shard_manifest_sha256",
    "target_residue_tokens",
    "training_protocol_sha256",
    "world_size",
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


def _relative_difference(reference: float, candidate: float) -> float:
    return abs(candidate - reference) / max(abs(reference), 1e-12)


def _optimization(name: str) -> dict[str, object]:
    compiled = name != "eager"
    return {
        "bucketed_padding": compiled,
        "compile_blocks": compiled,
        "compile_prewarm_buckets": compiled,
        "ddp_bucket_cap_mb": 25,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": name == "compiled_no_cudagraph",
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": compiled,
        "length_bucketing": False,
        "pinned_staging": compiled,
    }


def _validate(name: str, receipt: dict[str, object]) -> None:
    expected = {
        "backend": "ddp",
        "batch_quantum": 128,
        "deterministic_algorithms": True,
        "format_version": "antibody_pretraining_optimized_pilot_v1",
        "length_bucket_boundaries": EXPECTED_BUCKETS,
        "micro_batch_per_gpu": 16,
        "model_family_sha256": EXPECTED_MODEL_FAMILY_SHA256,
        "optimization": _optimization(name),
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
        raise RuntimeError(f"Deterministic receipt differs for {name}")
    if len(receipt.get("trace", [])) != 20:
        raise RuntimeError(f"Deterministic trace length differs for {name}")


def _summary(receipt: dict[str, object]) -> dict[str, float | int]:
    measured = receipt["trace"][5:]
    seconds = [float(item["seconds"]) for item in measured]
    tokens = sum(int(item["residue_tokens"]) for item in measured)
    return {
        "measured_updates": len(measured),
        "p50_step_seconds": statistics.median(seconds),
        "residue_tokens_per_second": tokens / sum(seconds),
        "warmup_updates_excluded": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", default=[])
    parser.add_argument("--state", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_bindings = _bindings(args.receipt, "receipt")
    state_bindings = _bindings(args.state, "state")
    job_ids = _bindings(args.job_id, "job ID")
    receipt_paths = {name: Path(path) for name, path in receipt_bindings.items()}
    state_paths = {name: Path(path) for name, path in state_bindings.items()}
    receipts = {name: _load(path) for name, path in receipt_paths.items()}
    states = {name: _load(path) for name, path in state_paths.items()}
    protocol = _load(args.efficiency_protocol)
    if protocol.get("format_version") != "antibody_efficiency_gate_v1":
        raise RuntimeError("Unexpected efficiency protocol")
    for name, receipt in receipts.items():
        _validate(name, receipt)

    reference_trace = receipts["eager"]["trace"]
    membership_equal = (
        len({receipt["membership_sha256"] for receipt in receipts.values()}) == 1
        and all(
            all(
                all(left[key] == right[key] for key in MEMBERSHIP_KEYS)
                for left, right in zip(reference_trace, receipt["trace"])
            )
            for receipt in receipts.values()
        )
    )
    compiled = receipts["compiled_cudagraph"]
    repeat = receipts["compiled_no_cudagraph"]
    repeatability = {
        "final_model": compiled["final_model_sha256"] == repeat["final_model_sha256"],
        "final_optimizer": (
            compiled["final_optimizer_sha256"] == repeat["final_optimizer_sha256"]
        ),
        "parameter_moments": compiled["parameter_moments"] == repeat["parameter_moments"],
        "trace": all(
            all(left[key] == right[key] for key in TRACE_KEYS)
            for left, right in zip(compiled["trace"], repeat["trace"])
        ),
        "trainer_state_coordinates": all(
            states["compiled_cudagraph"][key]
            == states["compiled_no_cudagraph"][key]
            for key in STATE_COORDINATE_KEYS
        ),
        "trainer_state_execution_mode": (
            states["compiled_cudagraph"]["optimization"]
            == _optimization("compiled_cudagraph")
            and states["compiled_no_cudagraph"]["optimization"]
            == _optimization("compiled_no_cudagraph")
        ),
    }
    reference = receipts["eager"]
    thresholds = protocol["correctness_gate"]
    loss_difference = abs(
        float(compiled["trace"][-1]["loss"])
        - float(reference["trace"][-1]["loss"])
    )
    gradient_difference = _relative_difference(
        float(reference["trace"][-1]["gradient_norm"]),
        float(compiled["trace"][-1]["gradient_norm"]),
    )
    moment_differences = {
        key: _relative_difference(
            float(reference["parameter_moments"][key]),
            float(compiled["parameter_moments"][key]),
        )
        for key in ("absolute_maximum", "square_sum", "sum")
    }
    correctness_passed = (
        loss_difference <= thresholds["maximum_final_loss_absolute_difference"]
        and gradient_difference
        <= thresholds["maximum_gradient_norm_relative_difference"]
        and max(moment_differences.values())
        <= thresholds["maximum_parameter_moment_relative_difference"]
    )
    passed = membership_equal and correctness_passed and all(repeatability.values())
    summaries = {name: _summary(receipt) for name, receipt in receipts.items()}
    eager_throughput = summaries["eager"]["residue_tokens_per_second"]
    for summary in summaries.values():
        summary["throughput_relative_to_eager"] = (
            summary["residue_tokens_per_second"] / eager_throughput
        )
    payload = {
        "authorization": {
            "benchmark_500_updates": passed,
            "pilot_16_gpu": False,
            "pilot_32_gpu": False,
            "production_pretraining": False,
        },
        "claim_boundary": "Twenty-update deterministic repeated-shard correctness and repeatability probe; no production-corpus, biological, scale, or state-of-the-art claim",
        "comparison_to_eager": {
            "final_gradient_norm_relative_difference": gradient_difference,
            "final_loss_absolute_difference": loss_difference,
            "maximum_parameter_moment_relative_difference": max(
                moment_differences.values()
            ),
            "passed": correctness_passed,
        },
        "format_version": "antibody_deterministic_repeatability_v1",
        "gate_passed": passed,
        "input_sha256": {
            **{f"receipt_{name}": _sha256(path) for name, path in receipt_paths.items()},
            **{f"state_{name}": _sha256(path) for name, path in state_paths.items()},
        },
        "job_ids": job_ids,
        "membership_equal": membership_equal,
        "repeatability": repeatability,
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
    if not passed:
        raise RuntimeError(f"Deterministic repeatability audit failed: {payload}")
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
