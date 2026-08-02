#!/usr/bin/env python3
"""Audit the counterbalanced 500-update DDP communication gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


RUNS = (
    "safe_eager_r1",
    "fp32_compiled_r1",
    "bf16_selected_r1",
    "bf16_selected_r2",
    "fp32_compiled_r2",
    "safe_eager_r2",
)
MODES = ("safe_eager", "fp32_compiled", "bf16_selected")
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
        if not separator or name not in RUNS or name in result or not item:
            raise RuntimeError(f"Invalid {label} binding: {value}")
        result[name] = item
    if tuple(result) != RUNS:
        raise RuntimeError(f"{label} bindings must use the frozen run order")
    return result


def _mode(run: str) -> str:
    return run.rsplit("_r", 1)[0]


def _relative_difference(reference: float, candidate: float) -> float:
    return abs(candidate - reference) / max(abs(reference), 1e-12)


def _relative_range_over_minimum(values: list[float]) -> float:
    if len(values) < 2 or min(values) <= 0.0:
        raise RuntimeError("Relative range requires at least two positive values")
    return (max(values) - min(values)) / min(values)


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _optimization(mode: str) -> dict[str, object]:
    common = {
        "audit_gradient_rank_parity": True,
        "ddp_bucket_cap_mb": 25,
        "ddp_static_graph": False,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": False,
        "disable_ddp_gradient_as_bucket_view": False,
        "fsdp_output_dtype": "bfloat16",
        "length_bucketing": False,
    }
    if mode == "safe_eager":
        return {
            **common,
            "bucketed_padding": False,
            "compile_blocks": False,
            "compile_prewarm_buckets": False,
            "ddp_batched_grad_copy": False,
            "ddp_communication_dtype": "float32",
            "fused_adamw": False,
            "pinned_staging": False,
        }
    if mode == "fp32_compiled":
        return {
            **common,
            "bucketed_padding": True,
            "compile_blocks": True,
            "compile_prewarm_buckets": True,
            "ddp_batched_grad_copy": False,
            "ddp_communication_dtype": "float32",
            "fused_adamw": True,
            "pinned_staging": True,
        }
    if mode == "bf16_selected":
        return {
            **common,
            "bucketed_padding": True,
            "compile_blocks": True,
            "compile_prewarm_buckets": True,
            "ddp_batched_grad_copy": True,
            "ddp_communication_dtype": "bfloat16",
            "fused_adamw": True,
            "pinned_staging": True,
        }
    raise RuntimeError(f"Unknown mode: {mode}")


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


def _validate_protocols(
    paired: dict[str, object],
    paired_sha256: str,
    communication: dict[str, object],
    communication_sha256: str,
    efficiency: dict[str, object],
    efficiency_sha256: str,
) -> None:
    if paired.get("format_version") != "antibody_ddp_communication_paired_gate_v1":
        raise RuntimeError("Unexpected paired gate protocol")
    if communication.get("format_version") != (
        "antibody_ddp_communication_optimization_v1"
    ):
        raise RuntimeError("Unexpected communication protocol")
    if efficiency.get("format_version") != "antibody_efficiency_gate_v1":
        raise RuntimeError("Unexpected efficiency protocol")
    if paired.get("parent_communication_protocol_sha256") != communication_sha256:
        raise RuntimeError("Paired protocol does not bind communication protocol")
    if paired.get("parent_efficiency_protocol_sha256") != efficiency_sha256:
        raise RuntimeError("Paired protocol does not bind efficiency protocol")
    if paired["benchmark"].get("run_order") != list(RUNS):
        raise RuntimeError("Paired protocol run order differs")
    if paired["benchmark"].get("updates_per_run") != 500:
        raise RuntimeError("Paired protocol update count differs")
    if paired["benchmark"].get("candidate_warmup_updates") != 50:
        raise RuntimeError("Paired protocol warmup differs")
    if paired_sha256 in {communication_sha256, efficiency_sha256}:
        raise RuntimeError("Protocol digests unexpectedly collide")


def _validate_allocation(
    allocation: dict[str, object],
    paired_sha256: str,
    receipts: dict[str, dict[str, object]],
    job_ids: dict[str, str],
) -> dict[str, object]:
    if allocation.get("format_version") != (
        "antibody_communication_paired_allocation_v1"
    ):
        raise RuntimeError("Unexpected allocation receipt")
    slurm = allocation.get("slurm", {})
    job_id = slurm.get("job_id")
    if (
        allocation.get("gpu_count") != EXPECTED_WORLD_SIZE
        or allocation.get("protocol_sha256") != paired_sha256
        or allocation.get("run_order") != list(RUNS)
        or slurm.get("node_count") != 1
        or set(job_ids.values()) != {job_id}
    ):
        raise RuntimeError("Allocation policy differs")
    inventory = allocation.get("gpu_inventory")
    if (
        not isinstance(inventory, list)
        or len(inventory) != EXPECTED_WORLD_SIZE
        or len({item.get("uuid") for item in inventory}) != EXPECTED_WORLD_SIZE
    ):
        raise RuntimeError("Allocation GPU inventory differs")
    expected_receipt_allocation = {
        "hostname": allocation.get("hostname"),
        "slurm_job_id": job_id,
        "slurm_job_nodelist": slurm.get("job_nodelist"),
    }
    if any(
        receipt.get("allocation") != expected_receipt_allocation
        for receipt in receipts.values()
    ):
        raise RuntimeError("Training receipts and allocation disagree")
    if allocation.get("test_access") != {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }:
        raise RuntimeError("Allocation receipt contains forbidden test access")
    return {
        "gpu_uuids": [item["uuid"] for item in inventory],
        "hostname": allocation["hostname"],
        "job_id": job_id,
        "job_nodelist": slurm["job_nodelist"],
    }


def _validate_receipt(
    run: str,
    receipt: dict[str, object],
    state: dict[str, object],
    communication_sha256: str,
    efficiency_sha256: str,
) -> None:
    mode = _mode(run)
    expected_communication_sha256 = (
        None if mode == "safe_eager" else communication_sha256
    )
    expected = {
        "backend": "ddp",
        "batch_quantum": 128,
        "ddp_communication_protocol_sha256": expected_communication_sha256,
        "deterministic_algorithms": True,
        "efficiency_protocol_sha256": efficiency_sha256,
        "format_version": "antibody_pretraining_optimized_pilot_v1",
        "length_bucket_boundaries": [96, 128, 144, 194],
        "micro_batch_per_gpu": 16,
        "model_family_sha256": EXPECTED_MODEL_FAMILY_SHA256,
        "optimization": _optimization(mode),
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
        raise RuntimeError(f"Receipt fields differ for {run}")
    if receipt.get("authorization") != {
        "production_pretraining": False,
        "scientific_claim": False,
    }:
        raise RuntimeError(f"Receipt authorization differs for {run}")
    if len(receipt.get("trace", [])) != 500:
        raise RuntimeError(f"Receipt must contain 500 updates for {run}")
    if not _rank_parity(receipt):
        raise RuntimeError(f"Eight-rank state parity failed for {run}")
    if receipt.get("parameter_moments", {}).get("count") != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Parameter count differs for {run}")
    ddp = receipt.get("ddp_logging_data", {})
    if (
        ddp.get("backend_name") != "nccl"
        or ddp.get("world_size") != EXPECTED_WORLD_SIZE
        or ddp.get("num_buckets_reduced") != 65
        or ddp.get("total_parameter_size_bytes") != EXPECTED_PARAMETER_BYTES
        or ddp.get("unused_parameter_size") != 0
        or "can_set_static_graph" in ddp
    ):
        raise RuntimeError(f"DDP logging differs for {run}")
    state_expected = {
        "backend": "ddp",
        "batch_quantum": 128,
        "consumed_residue_tokens": receipt["trace"][-1]["consumed_residue_tokens"],
        "ddp_communication_protocol_sha256": expected_communication_sha256,
        "efficiency_protocol_sha256": efficiency_sha256,
        "micro_batch_per_gpu": 16,
        "model_family_sha256": EXPECTED_MODEL_FAMILY_SHA256,
        "next_update": 500,
        "optimization": _optimization(mode),
        "preset": "abf_650m",
        "sample_cursor": 128000,
        "shard_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "target_residue_tokens": 16384,
        "training_protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
        "world_size": EXPECTED_WORLD_SIZE,
    }
    if any(state.get(key) != value for key, value in state_expected.items()):
        raise RuntimeError(f"Trainer state differs for {run}")


def _number(value: str) -> float:
    return float(value.strip().split()[0])


def _telemetry(
    path: Path, allocation_gpu_uuids: set[str]
) -> dict[str, float | int | str | list[str]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [
        {key.strip(): value.strip() for key, value in row.items()}
        for row in raw_rows
    ]
    if not rows:
        raise RuntimeError(f"GPU telemetry is empty: {path}")
    indices = {int(row["index"]) for row in rows}
    uuids = {row["uuid"] for row in rows}
    totals = {int(_number(row["memory.total [MiB]"])) for row in rows}
    names = {row["name"] for row in rows}
    if (
        indices != set(range(EXPECTED_WORLD_SIZE))
        or uuids != allocation_gpu_uuids
        or totals != {32607}
        or names != {"NVIDIA GeForce RTX 5090"}
    ):
        raise RuntimeError(f"GPU telemetry topology differs: {path}")
    active = [row for row in rows if _number(row["memory.used [MiB]"]) > 100.0]
    if not active:
        raise RuntimeError(f"GPU telemetry has no active samples: {path}")
    return {
        "active_samples": len(active),
        "gpu_indices": sorted(indices),
        "gpu_utilization_mean_percent": statistics.fmean(
            _number(row["utilization.gpu [%]"]) for row in active
        ),
        "gpu_utilization_median_percent": statistics.median(
            _number(row["utilization.gpu [%]"]) for row in active
        ),
        "maximum_memory_mib": max(
            _number(row["memory.used [MiB]"]) for row in active
        ),
        "mean_power_watts": statistics.fmean(
            _number(row["power.draw [W]"]) for row in active
        ),
        "rows": len(rows),
        "total_memory_mib": next(iter(totals)),
    }


def _summary(receipt: dict[str, object], warmup: int) -> dict[str, float | int]:
    measured = receipt["trace"][warmup:]
    seconds = [float(item["seconds"]) for item in measured]
    if not all(math.isfinite(value) and value > 0.0 for value in seconds):
        raise RuntimeError("Receipt contains invalid timing")
    tokens = sum(int(item["residue_tokens"]) for item in measured)
    p50 = _quantile(seconds, 0.5)
    p95 = _quantile(seconds, 0.95)
    return {
        "maximum_step_seconds": max(seconds),
        "measured_updates": len(measured),
        "p50_step_seconds": p50,
        "p95_over_p50": p95 / p50,
        "p95_step_seconds": p95,
        "residue_tokens": tokens,
        "residue_tokens_per_second": tokens / sum(seconds),
        "warmup_updates_excluded": warmup,
    }


def _comparison(
    reference: dict[str, object],
    candidate: dict[str, object],
    thresholds: dict[str, object],
) -> dict[str, object]:
    maximum_loss = max(
        abs(float(left["loss"]) - float(right["loss"]))
        for left, right in zip(reference["trace"], candidate["trace"])
    )
    maximum_gradient = max(
        _relative_difference(
            float(left["gradient_norm"]), float(right["gradient_norm"])
        )
        for left, right in zip(reference["trace"], candidate["trace"])
    )
    moments = {
        key: _relative_difference(
            float(reference["parameter_moments"][key]),
            float(candidate["parameter_moments"][key]),
        )
        for key in ("absolute_maximum", "square_sum", "sum")
    }
    passed = (
        maximum_loss <= thresholds["maximum_final_loss_absolute_difference"]
        and maximum_gradient
        <= thresholds["maximum_gradient_norm_relative_difference"]
        and max(moments.values())
        <= thresholds["maximum_parameter_moment_relative_difference"]
    )
    return {
        "maximum_gradient_norm_relative_difference": maximum_gradient,
        "maximum_loss_absolute_difference": maximum_loss,
        "maximum_parameter_moment_relative_difference": max(moments.values()),
        "parameter_moment_relative_differences": moments,
        "passed": passed,
    }


def _same_scientific_state(
    left: dict[str, object], right: dict[str, object]
) -> bool:
    keys = tuple(key for key in left["trace"][0] if key != "seconds")
    return (
        all(
            all(left_item[key] == right_item[key] for key in keys)
            for left_item, right_item in zip(left["trace"], right["trace"])
        )
        and left["parameter_moments"] == right["parameter_moments"]
        and left["final_model_sha256"] == right["final_model_sha256"]
        and left["final_optimizer_sha256"] == right["final_optimizer_sha256"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", default=[])
    parser.add_argument("--trainer-state", action="append", default=[])
    parser.add_argument("--telemetry", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--allocation-receipt", type=Path, required=True)
    parser.add_argument("--paired-protocol", type=Path, required=True)
    parser.add_argument("--communication-protocol", type=Path, required=True)
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt_bindings = _bindings(args.receipt, "receipt")
    state_bindings = _bindings(args.trainer_state, "trainer state")
    telemetry_bindings = _bindings(args.telemetry, "telemetry")
    job_ids = _bindings(args.job_id, "job ID")
    receipt_paths = {name: Path(value) for name, value in receipt_bindings.items()}
    state_paths = {name: Path(value) for name, value in state_bindings.items()}
    telemetry_paths = {name: Path(value) for name, value in telemetry_bindings.items()}
    receipts = {name: _load(path) for name, path in receipt_paths.items()}
    states = {name: _load(path) for name, path in state_paths.items()}
    paired = _load(args.paired_protocol)
    communication = _load(args.communication_protocol)
    efficiency = _load(args.efficiency_protocol)
    paired_sha256 = _sha256(args.paired_protocol)
    communication_sha256 = _sha256(args.communication_protocol)
    efficiency_sha256 = _sha256(args.efficiency_protocol)
    _validate_protocols(
        paired,
        paired_sha256,
        communication,
        communication_sha256,
        efficiency,
        efficiency_sha256,
    )
    allocation = _load(args.allocation_receipt)
    allocation_summary = _validate_allocation(
        allocation, paired_sha256, receipts, job_ids
    )
    allocation_gpu_uuids = set(allocation_summary["gpu_uuids"])
    telemetry = {}
    for run in RUNS:
        _validate_receipt(
            run,
            receipts[run],
            states[run],
            communication_sha256,
            efficiency_sha256,
        )
        telemetry[run] = _telemetry(
            telemetry_paths[run], allocation_gpu_uuids
        )

    membership_keys = (
        "consumed_residue_tokens",
        "learning_rate",
        "micro_steps",
        "residue_tokens",
        "selected_targets",
        "sequence_count",
        "update",
    )
    reference_trace = receipts[RUNS[0]]["trace"]
    membership_equal = (
        len({receipt["membership_sha256"] for receipt in receipts.values()}) == 1
        and all(
            all(
                all(left[key] == right[key] for key in membership_keys)
                for left, right in zip(reference_trace, receipt["trace"])
            )
            for receipt in receipts.values()
        )
    )
    replicate_exact = {
        mode: _same_scientific_state(
            receipts[f"{mode}_r1"], receipts[f"{mode}_r2"]
        )
        for mode in MODES
    }
    comparisons = {}
    correctness_thresholds = efficiency["correctness_gate"]
    for replicate in ("r1", "r2"):
        reference = receipts[f"safe_eager_{replicate}"]
        for mode in ("fp32_compiled", "bf16_selected"):
            label = f"{mode}_{replicate}_to_safe_eager_{replicate}"
            comparisons[label] = _comparison(
                reference,
                receipts[f"{mode}_{replicate}"],
                correctness_thresholds,
            )

    warmup = paired["benchmark"]["candidate_warmup_updates"]
    summaries = {run: _summary(receipts[run], warmup) for run in RUNS}
    for run, summary in summaries.items():
        summary["maximum_memory_fraction"] = (
            max(
                float(receipts[run]["peak_allocated_memory_mib"]),
                float(receipts[run]["peak_reserved_memory_mib"]),
                float(telemetry[run]["maximum_memory_mib"]),
            )
            / telemetry[run]["total_memory_mib"]
        )
    replicate_ranges = {
        mode: _relative_range_over_minimum(
            [
                summaries[f"{mode}_r1"]["residue_tokens_per_second"],
                summaries[f"{mode}_r2"]["residue_tokens_per_second"],
            ]
        )
        for mode in MODES
    }
    round_ratios = {
        replicate: (
            summaries[f"bf16_selected_{replicate}"]["residue_tokens_per_second"]
            / summaries[f"safe_eager_{replicate}"]["residue_tokens_per_second"]
        )
        for replicate in ("r1", "r2")
    }

    def pooled_throughput(mode: str) -> float:
        runs = (f"{mode}_r1", f"{mode}_r2")
        tokens = sum(summaries[run]["residue_tokens"] for run in runs)
        seconds = sum(
            summaries[run]["residue_tokens"]
            / summaries[run]["residue_tokens_per_second"]
            for run in runs
        )
        return tokens / seconds

    pooled_throughputs = {mode: pooled_throughput(mode) for mode in MODES}
    pooled_ratios = {
        mode: value / pooled_throughputs["safe_eager"]
        for mode, value in pooled_throughputs.items()
    }
    acceptance = paired["acceptance"]
    gates = {
        "allocation": True,
        "memory": all(
            summary["maximum_memory_fraction"]
            <= acceptance["maximum_memory_fraction"]
            for summary in summaries.values()
        ),
        "membership": membership_equal,
        "numerical_correctness": all(
            comparison["passed"] for comparison in comparisons.values()
        ),
        "rank_parity": True,
        "replicate_exact": all(replicate_exact.values()),
        "replicate_throughput_range": all(
            value <= acceptance["maximum_replicate_throughput_relative_range"]
            for value in replicate_ranges.values()
        ),
        "stability": all(
            summary["p95_over_p50"]
            <= acceptance["maximum_p95_over_p50_step_time_each_run"]
            for summary in summaries.values()
        ),
        "throughput_each_round": all(
            ratio
            >= acceptance[
                "minimum_selected_throughput_relative_to_safe_reference_each_round"
            ]
            for ratio in round_ratios.values()
        ),
        "throughput_pooled": (
            pooled_ratios["bf16_selected"]
            >= acceptance[
                "minimum_selected_throughput_relative_to_safe_reference_pooled"
            ]
        ),
    }
    gate_passed = all(gates.values())
    payload = {
        "allocation": {
            **allocation_summary,
            "receipt_sha256": _sha256(args.allocation_receipt),
        },
        "authorization": {
            "exact_recovery": gate_passed,
            "pilot_16_gpu": False,
            "pilot_32_gpu": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Counterbalanced 500-update repeated-shard eight-GPU systems gate; "
            "no production-corpus, biological, scaling, or state-of-the-art claim"
        ),
        "comparisons_to_safe_eager": comparisons,
        "format_version": "antibody_ddp_communication_paired_gate_result_v1",
        "gate_passed": gate_passed,
        "gates": gates,
        "membership_equal": membership_equal,
        "paired_protocol_sha256": paired_sha256,
        "pooled_throughput_relative_to_safe_eager": pooled_ratios,
        "pooled_throughputs": pooled_throughputs,
        "relative_range_formula": "(maximum-minimum)/minimum",
        "replicate_exact": replicate_exact,
        "replicate_throughput_relative_ranges": replicate_ranges,
        "round_selected_throughput_relative_to_safe_eager": round_ratios,
        "runs": {
            run: {
                "job_id": job_ids[run],
                "receipt_sha256": _sha256(receipt_paths[run]),
                "slurm_state": "COMPLETED",
                "telemetry_sha256": _sha256(telemetry_paths[run]),
                "trainer_state_sha256": _sha256(state_paths[run]),
            }
            for run in RUNS
        },
        "source": {
            "archive_sha256": args.source_archive_sha256,
            "auditor_sha256": _sha256(Path(__file__)),
            "runner_sha256": args.runner_sha256,
        },
        "summaries": summaries,
        "telemetry": telemetry,
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
