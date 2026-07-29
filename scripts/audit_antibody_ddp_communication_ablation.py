#!/usr/bin/env python3
"""Audit the frozen eight-GPU DDP communication candidate grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


CANDIDATES = (
    "fp32_control",
    "fp32_batched_copy",
    "bf16_compress",
    "bf16_compress_batched_copy",
)
EXPECTED_CANDIDATES = {
    "fp32_control": ("float32", False),
    "fp32_batched_copy": ("float32", True),
    "bf16_compress": ("bfloat16", False),
    "bf16_compress_batched_copy": ("bfloat16", True),
}
EXPECTED_MANIFEST_SHA256 = (
    "0f73ccf7f537d223ca6d94b83f08b40a34c23e557f2a747cc32d798c974450f6"
)
EXPECTED_MODEL_FAMILY_SHA256 = (
    "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
)
EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "c8974cb06350da46ca49ffff61c149061b7e7f46b43c9e686176539bbe223e6b"
)
EXPECTED_PARAMETER_COUNT = 634_513_920
EXPECTED_PARAMETER_BYTES = 2_538_055_680
EXPECTED_WORLD_SIZE = 8


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
    if tuple(result) != CANDIDATES:
        raise RuntimeError(f"{label} bindings must use canonical candidate order")
    return result


def _relative_difference(reference: float, candidate: float) -> float:
    return abs(candidate - reference) / max(abs(reference), 1e-12)


def _rank_parity(receipt: dict[str, object]) -> bool:
    checks = (
        ("initial_model_rank_sha256", None),
        ("final_model_rank_sha256", "final_model_sha256"),
        ("final_optimizer_rank_sha256", "final_optimizer_sha256"),
    )
    for rank_key, summary_key in checks:
        values = receipt.get(rank_key)
        if not isinstance(values, list) or len(values) != EXPECTED_WORLD_SIZE:
            return False
        if len(set(values)) != 1:
            return False
        if summary_key is not None and values[0] != receipt.get(summary_key):
            return False
    return True


def _optimization(dtype: str, batched_copy: bool) -> dict[str, object]:
    return {
        "audit_gradient_rank_parity": True,
        "bucketed_padding": True,
        "compile_blocks": True,
        "compile_prewarm_buckets": True,
        "ddp_batched_grad_copy": batched_copy,
        "ddp_bucket_cap_mb": 25,
        "ddp_communication_dtype": dtype,
        "ddp_static_graph": False,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": False,
        "disable_ddp_gradient_as_bucket_view": False,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": True,
        "length_bucketing": False,
        "pinned_staging": True,
    }


def _validate_protocols(
    communication_protocol: dict[str, object],
    communication_protocol_sha256: str,
    efficiency_protocol: dict[str, object],
    efficiency_protocol_sha256: str,
) -> None:
    if communication_protocol.get("format_version") != (
        "antibody_ddp_communication_optimization_v1"
    ):
        raise RuntimeError("Unexpected communication protocol")
    if communication_protocol.get("parent_efficiency_protocol_sha256") != (
        efficiency_protocol_sha256
    ):
        raise RuntimeError("Communication protocol does not bind the parent protocol")
    if efficiency_protocol.get("format_version") != "antibody_efficiency_gate_v1":
        raise RuntimeError("Unexpected efficiency protocol")
    protocol_candidates = communication_protocol.get("candidates")
    expected = [
        {
            "batched_grad_copy": EXPECTED_CANDIDATES[name][1],
            "communication_dtype": EXPECTED_CANDIDATES[name][0],
            "label": name,
        }
        for name in CANDIDATES
    ]
    if protocol_candidates != expected:
        raise RuntimeError("Communication candidates differ from the frozen grid")
    if communication_protocol_sha256 == efficiency_protocol_sha256:
        raise RuntimeError("Protocol digests unexpectedly collide")


def _validate_receipt(
    name: str,
    receipt: dict[str, object],
    trainer_state: dict[str, object],
    communication_protocol_sha256: str,
    efficiency_protocol_sha256: str,
) -> None:
    dtype, batched_copy = EXPECTED_CANDIDATES[name]
    expected = {
        "backend": "ddp",
        "batch_quantum": 128,
        "ddp_communication_protocol_sha256": communication_protocol_sha256,
        "deterministic_algorithms": True,
        "efficiency_protocol_sha256": efficiency_protocol_sha256,
        "format_version": "antibody_pretraining_optimized_pilot_v1",
        "micro_batch_per_gpu": 16,
        "model_family_sha256": EXPECTED_MODEL_FAMILY_SHA256,
        "optimization": _optimization(dtype, batched_copy),
        "preset": "abf_650m",
        "shard_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "target_residue_tokens": 16384,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "training_protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
        "world_size": EXPECTED_WORLD_SIZE,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Receipt fields differ for {name}")
    if receipt.get("authorization") != {
        "production_pretraining": False,
        "scientific_claim": False,
    }:
        raise RuntimeError(f"Receipt authorization differs for {name}")
    if len(receipt.get("trace", [])) != 20:
        raise RuntimeError(f"Receipt must contain 20 updates for {name}")
    if not _rank_parity(receipt):
        raise RuntimeError(f"Eight-rank state parity failed for {name}")
    moments = receipt.get("parameter_moments", {})
    if moments.get("count") != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Parameter count differs for {name}")
    ddp = receipt.get("ddp_logging_data", {})
    if (
        ddp.get("backend_name") != "nccl"
        or ddp.get("world_size") != EXPECTED_WORLD_SIZE
        or ddp.get("num_buckets_reduced") != 65
        or ddp.get("total_parameter_size_bytes") != EXPECTED_PARAMETER_BYTES
        or ddp.get("gradient_as_bucket_view") != 1
        or ddp.get("has_rebuilt_buckets") != 1
        or ddp.get("unused_parameter_size") != 0
        or "can_set_static_graph" in ddp
    ):
        raise RuntimeError(f"DDP receipt differs for {name}")
    expected_state = {
        "backend": "ddp",
        "batch_quantum": 128,
        "consumed_residue_tokens": receipt["trace"][-1]["consumed_residue_tokens"],
        "ddp_communication_protocol_sha256": communication_protocol_sha256,
        "efficiency_protocol_sha256": efficiency_protocol_sha256,
        "micro_batch_per_gpu": 16,
        "model_family_sha256": EXPECTED_MODEL_FAMILY_SHA256,
        "next_update": 20,
        "optimization": _optimization(dtype, batched_copy),
        "preset": "abf_650m",
        "sample_cursor": 5120,
        "shard_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "target_residue_tokens": 16384,
        "training_protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
        "world_size": EXPECTED_WORLD_SIZE,
    }
    if any(trainer_state.get(key) != value for key, value in expected_state.items()):
        raise RuntimeError(f"Trainer state differs for {name}")


def _validate_telemetry(path: Path) -> dict[str, object]:
    with path.open("r", encoding="ascii", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [
        {key.strip(): value.strip() for key, value in row.items()}
        for row in raw_rows
    ]
    if not rows:
        raise RuntimeError(f"GPU telemetry is empty: {path}")
    indices = {int(row["index"]) for row in rows}
    totals = {int(row["memory.total [MiB]"].removesuffix(" MiB")) for row in rows}
    names = {row["name"] for row in rows}
    if indices != set(range(EXPECTED_WORLD_SIZE)) or totals != {32607}:
        raise RuntimeError(f"GPU telemetry topology differs: {path}")
    if names != {"NVIDIA GeForce RTX 5090"}:
        raise RuntimeError(f"GPU telemetry device differs: {path}")
    return {
        "device": next(iter(names)),
        "gpu_indices": sorted(indices),
        "rows": len(rows),
        "total_memory_mib": next(iter(totals)),
    }


def _summary(receipt: dict[str, object], warmup: int) -> dict[str, float | int | bool]:
    measured = receipt["trace"][warmup:]
    seconds = [float(item["seconds"]) for item in measured]
    if not all(math.isfinite(value) and value > 0.0 for value in seconds):
        raise RuntimeError("Receipt contains invalid timing")
    p50 = statistics.median(seconds)
    p95 = sorted(seconds)[math.ceil(0.95 * len(seconds)) - 1]
    tokens = sum(int(item["residue_tokens"]) for item in measured)
    return {
        "measured_updates": len(measured),
        "p50_step_seconds": p50,
        "p95_over_p50": p95 / p50,
        "peak_reserved_memory_mib": float(receipt["peak_reserved_memory_mib"]),
        "residue_tokens": tokens,
        "residue_tokens_per_second": tokens / sum(seconds),
        "warmup_updates_excluded": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", default=[])
    parser.add_argument("--trainer-state", action="append", default=[])
    parser.add_argument("--gpu-telemetry", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--communication-protocol", type=Path, required=True)
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt_bindings = _bindings(args.receipt, "receipt")
    state_bindings = _bindings(args.trainer_state, "trainer state")
    telemetry_bindings = _bindings(args.gpu_telemetry, "GPU telemetry")
    job_ids = _bindings(args.job_id, "job ID")
    receipt_paths = {name: Path(value) for name, value in receipt_bindings.items()}
    state_paths = {name: Path(value) for name, value in state_bindings.items()}
    telemetry_paths = {
        name: Path(value) for name, value in telemetry_bindings.items()
    }
    receipts = {name: _load(path) for name, path in receipt_paths.items()}
    states = {name: _load(path) for name, path in state_paths.items()}
    communication_protocol = _load(args.communication_protocol)
    efficiency_protocol = _load(args.efficiency_protocol)
    communication_protocol_sha256 = _sha256(args.communication_protocol)
    efficiency_protocol_sha256 = _sha256(args.efficiency_protocol)
    _validate_protocols(
        communication_protocol,
        communication_protocol_sha256,
        efficiency_protocol,
        efficiency_protocol_sha256,
    )

    telemetry = {}
    for name in CANDIDATES:
        _validate_receipt(
            name,
            receipts[name],
            states[name],
            communication_protocol_sha256,
            efficiency_protocol_sha256,
        )
        telemetry[name] = _validate_telemetry(telemetry_paths[name])

    reference = receipts["fp32_control"]
    membership_equal = len(
        {receipt["membership_sha256"] for receipt in receipts.values()}
    ) == 1
    membership_keys = (
        "consumed_residue_tokens",
        "learning_rate",
        "micro_steps",
        "residue_tokens",
        "selected_targets",
        "sequence_count",
        "update",
    )
    trace_membership_equal = all(
        all(
            all(left[key] == right[key] for key in membership_keys)
            for left, right in zip(reference["trace"], receipt["trace"])
        )
        for receipt in receipts.values()
    )
    if not membership_equal or not trace_membership_equal:
        raise RuntimeError("Candidate membership differs")

    thresholds = efficiency_protocol["correctness_gate"]
    comparisons = {}
    correctness_passers = []
    for name in CANDIDATES:
        receipt = receipts[name]
        loss_difference = max(
            abs(float(left["loss"]) - float(right["loss"]))
            for left, right in zip(reference["trace"], receipt["trace"])
        )
        gradient_difference = max(
            _relative_difference(
                float(left["gradient_norm"]), float(right["gradient_norm"])
            )
            for left, right in zip(reference["trace"], receipt["trace"])
        )
        moment_differences = {
            key: _relative_difference(
                float(reference["parameter_moments"][key]),
                float(receipt["parameter_moments"][key]),
            )
            for key in ("absolute_maximum", "square_sum", "sum")
        }
        passed = (
            loss_difference
            <= thresholds["maximum_final_loss_absolute_difference"]
            and gradient_difference
            <= thresholds["maximum_gradient_norm_relative_difference"]
            and max(moment_differences.values())
            <= thresholds["maximum_parameter_moment_relative_difference"]
        )
        if passed:
            correctness_passers.append(name)
        comparisons[name] = {
            "maximum_gradient_norm_relative_difference": gradient_difference,
            "maximum_loss_absolute_difference": loss_difference,
            "maximum_parameter_moment_relative_difference": max(
                moment_differences.values()
            ),
            "parameter_moment_relative_differences": moment_differences,
            "passed": passed,
        }

    if tuple(correctness_passers) != CANDIDATES:
        raise RuntimeError("Not every frozen communication candidate passed correctness")
    summaries = {name: _summary(receipt, 5) for name, receipt in receipts.items()}
    reference_throughput = summaries["fp32_control"]["residue_tokens_per_second"]
    memory_threshold = efficiency_protocol["efficiency_gate"][
        "maximum_memory_fraction"
    ]
    stability_threshold = efficiency_protocol["efficiency_gate"][
        "maximum_p95_over_p50_step_time"
    ]
    for name, summary in summaries.items():
        summary["memory_fraction"] = (
            summary["peak_reserved_memory_mib"] / telemetry[name]["total_memory_mib"]
        )
        summary["memory_diagnostic_passed"] = (
            summary["memory_fraction"] <= memory_threshold
        )
        summary["stability_diagnostic_passed"] = (
            summary["p95_over_p50"] <= stability_threshold
        )
        summary["throughput_relative_to_fp32_control"] = (
            summary["residue_tokens_per_second"] / reference_throughput
        )
    selected = max(
        correctness_passers,
        key=lambda name: summaries[name]["residue_tokens_per_second"],
    )
    if selected != "bf16_compress_batched_copy":
        raise RuntimeError(f"Unexpected selected candidate: {selected}")

    payload = {
        "authorization": {
            "benchmark_500_update_selected_candidate": selected,
            "pilot_16_gpu": False,
            "pilot_32_gpu": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "auditor_sha256": _sha256(Path(__file__)),
        "claim_boundary": (
            "Twenty-update repeated-shard eight-GPU communication correctness grid; "
            "short timing is candidate-selection evidence only and no scale, "
            "production-corpus, biological, or state-of-the-art claim is authorized"
        ),
        "comparisons_to_fp32_control": comparisons,
        "communication_protocol_sha256": communication_protocol_sha256,
        "correctness_passers": correctness_passers,
        "efficiency_protocol_sha256": efficiency_protocol_sha256,
        "format_version": "antibody_ddp_communication_ablation_v1",
        "membership_equal": membership_equal,
        "runs": {
            name: {
                "gpu_telemetry_sha256": _sha256(telemetry_paths[name]),
                "job_id": job_ids[name],
                "receipt_sha256": _sha256(receipt_paths[name]),
                "slurm_state": "COMPLETED",
                "trainer_state_sha256": _sha256(state_paths[name]),
            }
            for name in CANDIDATES
        },
        "selected_candidate": selected,
        "selection_rule": (
            "highest_throughput_among_all_correctness_passing_candidates"
        ),
        "source": {
            "archive_sha256": args.source_archive_sha256,
            "runner_sha256": args.runner_sha256,
        },
        "summaries": summaries,
        "telemetry": telemetry,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "trace_membership_equal": trace_membership_equal,
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
