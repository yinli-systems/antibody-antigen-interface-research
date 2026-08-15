#!/usr/bin/env python3
"""Audit a real-OAS four-GPU A/B/B/A efficiency and equivalence gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping

import numpy as np

import audit_antibody_production_batch_efficiency as artifacts
import freeze_antibody_four_gpu_efficiency_gate as freezer


FORMAT_VERSION = "antibody_four_gpu_efficiency_result_v1"
RUN_FORMAT = "antibody_production_loader_recovery_run_v1"
EVALUATION_FORMAT = "antibody_production_checkpoint_evaluation_v1"
PARAMETER_COUNT = 634_513_920
PEAK_DENSE_BF16_TFLOPS = 209.5
GPU_MODEL = "NVIDIA GeForce RTX 5090"
RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("compiled_r1", "compiled"),
    ("compiled_r2", "compiled"),
    ("safe_eager_r2", "safe_eager"),
)
RUN_TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
RUN_AUTHORIZATION = {
    "loader_recovery_gate": True,
    "production_pretraining": False,
    "scale_out": False,
    "scientific_claim": False,
}
EVALUATION_AUTHORIZATION = {
    "checkpoint_evaluation": True,
    "configuration_promotion": False,
    "production_pretraining": False,
    "scientific_claim": False,
}


class EfficiencyAuditError(RuntimeError):
    """Report invalid evidence or a broken frozen contract."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise EfficiencyAuditError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise EfficiencyAuditError(f"Non-finite JSON constant: {value}")


def _load(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise EfficiencyAuditError(f"Evidence path is invalid: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EfficiencyAuditError(f"Evidence is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise EfficiencyAuditError(f"Evidence is not a canonical object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise EfficiencyAuditError(f"{name} is not finite")
    return result


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EfficiencyAuditError("Cannot summarize an empty timing trace")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _number(value: str) -> float:
    return float(value.strip().split()[0])


def _telemetry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    active: list[dict[str, str]] = []
    for row in rows:
        if any(key is None or value is None for key, value in row.items()):
            raise EfficiencyAuditError(f"Malformed GPU telemetry row: {path}")
        normalized = {
            key.strip(): value.strip()
            for key, value in row.items()
        }
        if _number(normalized["memory.used [MiB]"]) > 100.0:
            active.append(normalized)
    if not active:
        raise EfficiencyAuditError(f"No active GPU telemetry: {path}")
    uuids = sorted({row["uuid"] for row in active})
    if len(uuids) != 4 or {row["name"] for row in active} != {GPU_MODEL}:
        raise EfficiencyAuditError("Telemetry does not bind four RTX 5090 GPUs")
    totals = {_number(row["memory.total [MiB]"]) for row in active}
    if len(totals) != 1:
        raise EfficiencyAuditError("GPU total memory differs within an allocation")
    return {
        "active_samples": len(active),
        "file_sha256": _sha256(path),
        "gpu_utilization_mean_percent": statistics.fmean(
            _number(row["utilization.gpu [%]"]) for row in active
        ),
        "gpu_uuids": uuids,
        "maximum_memory_mib": max(
            _number(row["memory.used [MiB]"]) for row in active
        ),
        "mean_power_watts": statistics.fmean(
            _number(row["power.draw [W]"]) for row in active
        ),
        "total_memory_mib": totals.pop(),
    }


def _validate_trace(receipt: Mapping[str, Any], updates: int) -> None:
    trace = receipt.get("trace")
    if not isinstance(trace, list) or len(trace) != updates:
        raise EfficiencyAuditError("Training trace length differs")
    consumed = 0
    for expected, item in enumerate(trace, start=1):
        if not isinstance(item, Mapping) or item.get("update") != expected:
            raise EfficiencyAuditError("Training trace is not contiguous")
        tokens = item.get("residue_tokens")
        total = item.get("consumed_residue_tokens")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total != consumed + tokens
            or int(item.get("micro_steps", 0)) <= 0
            or int(item.get("selected_targets", 0)) <= 0
            or int(item.get("sequence_count", 0)) <= 0
            or _finite(item.get("seconds"), "step seconds") <= 0.0
            or _finite(item.get("loss"), "loss") < 0.0
            or _finite(item.get("gradient_norm"), "gradient norm") < 0.0
        ):
            raise EfficiencyAuditError("Training trace contains invalid values")
        consumed = total


def _evaluation(
    value: Mapping[str, Any],
    *,
    gate_sha256: str,
    receipt_sha256: str,
    protocol: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panel = protocol["validation_panel"]
    if (
        value.get("format_version") != EVALUATION_FORMAT
        or value.get("world_size") != 4
        or value.get("gate_protocol_sha256") != gate_sha256
        or value.get("checkpoint", {}).get("receipt_file_sha256") != receipt_sha256
        or value.get("panel")
        != {
            "indices_sha256": panel["indices_sha256"],
            "mask_seed": panel["mask_seed"],
            "mask_update": panel["mask_update"],
            "record_count": panel["record_count"],
            "seed": panel["seed"],
            "split": "validation",
        }
        or value.get("authorization") != EVALUATION_AUTHORIZATION
        or value.get("test_access") != freezer.TEST_ACCESS
    ):
        raise EfficiencyAuditError("Checkpoint evaluation identity differs")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != panel["record_count"]:
        raise EfficiencyAuditError("Checkpoint evaluation panel differs")
    indices = np.asarray([int(item["index"]) for item in records], dtype=np.int64)
    cross_entropy = np.asarray(
        [_finite(item["masked_cross_entropy"], "cross entropy") for item in records],
        dtype=np.float64,
    )
    accuracy = np.asarray(
        [_finite(item["masked_accuracy"], "accuracy") for item in records],
        dtype=np.float64,
    )
    if (
        len(np.unique(indices)) != len(indices)
        or np.any(cross_entropy < 0.0)
        or np.any(accuracy < 0.0)
        or np.any(accuracy > 1.0)
    ):
        raise EfficiencyAuditError("Evaluation records are invalid")
    aggregate = value.get("aggregate", {})
    targets = sum(int(item["selected_targets"]) for item in records)
    correct = sum(int(item["correct"]) for item in records)
    loss_sum = sum(_finite(item["loss_sum"], "loss sum") for item in records)
    if (
        aggregate.get("selected_targets") != targets
        or not math.isclose(
            _finite(aggregate.get("masked_accuracy"), "aggregate accuracy"),
            correct / targets,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            _finite(aggregate.get("masked_cross_entropy"), "aggregate CE"),
            loss_sum / targets,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise EfficiencyAuditError("Evaluation aggregate does not replay")
    order = np.argsort(indices)
    return indices[order], cross_entropy[order], accuracy[order]


def _validate_master_protocol(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("format_version") != freezer.FORMAT_VERSION
        or protocol.get("acceptance") != freezer.ACCEPTANCE
        or protocol.get("batch") != freezer.BATCH
        or protocol.get("measurement") != freezer.MEASUREMENT
        or protocol.get("hardware") != freezer.HARDWARE
        or protocol.get("validation_panel") != freezer.VALIDATION_PANEL
        or protocol.get("authorization") != freezer.AUTHORIZATION
        or protocol.get("test_access") != freezer.TEST_ACCESS
        or set(protocol.get("source", {})) != set(freezer.SOURCE_FILES)
    ):
        raise EfficiencyAuditError("Master efficiency protocol differs")


def _validate_mode_gate(gate: Mapping[str, Any], *, compiled: bool) -> None:
    expected_optimization = {
        "bucketed_padding": compiled,
        "compile_blocks": compiled,
        "compile_prewarm_buckets": compiled,
        "ddp_bucket_cap_mb": 25,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": compiled,
        "fused_adamw": compiled,
        "length_bucketing": compiled,
        "pinned_staging": compiled,
    }
    if (
        gate.get("format_version") != "antibody_production_loader_recovery_gate_v1"
        or gate.get("hardware_identity")
        != {"gpu_count": 4, "gpu_model": GPU_MODEL, "node_count": 1}
        or gate.get("authorization") != RUN_AUTHORIZATION
        or gate.get("test_access") != RUN_TEST_ACCESS
        or gate.get("execution", {}).get("world_size") != 4
        or gate.get("execution", {}).get("backend") != "ddp"
        or gate.get("execution", {}).get("optimization") != expected_optimization
        or gate.get("runs")
        != {
            "continuous": {
                "batch_quantum_sequences": 256,
                "checkpoint": True,
                "micro_batch_per_gpu": 16,
                "preset": "abf_650m",
                "resume": False,
                "run_seed": 17,
                "steps": 600,
                "target_residue_tokens": 131072,
            }
        }
    ):
        raise EfficiencyAuditError("Mode gate identity differs")


def _throughput(receipt: Mapping[str, Any], warmup: int) -> dict[str, Any]:
    trace = receipt["trace"][warmup:]
    seconds = [_finite(item["seconds"], "step seconds") for item in trace]
    tokens = sum(int(item["residue_tokens"]) for item in trace)
    rate = tokens / sum(seconds)
    return {
        "approximate_6n_mfu": (
            6.0
            * PARAMETER_COUNT
            * rate
            / (4 * PEAK_DENSE_BF16_TFLOPS * 1.0e12)
        ),
        "measured_seconds": sum(seconds),
        "measured_updates": len(trace),
        "nonpadding_residue_tokens": tokens,
        "p50_step_seconds": _quantile(seconds, 0.50),
        "p95_over_p50": _quantile(seconds, 0.95) / _quantile(seconds, 0.50),
        "p95_step_seconds": _quantile(seconds, 0.95),
        "residue_tokens_per_second": rate,
        "warmup_updates_excluded": warmup,
    }


def _bootstrap_noninferiority(
    eager: tuple[np.ndarray, np.ndarray, np.ndarray],
    compiled: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if not np.array_equal(eager[0], compiled[0]):
        raise EfficiencyAuditError("Paired evaluation membership differs")
    eager_ce, eager_acc = eager[1], eager[2]
    compiled_ce, compiled_acc = compiled[1], compiled[2]
    observed_ce = (compiled_ce.mean() - eager_ce.mean()) / eager_ce.mean()
    observed_acc = eager_acc.mean() - compiled_acc.mean()
    rng = np.random.default_rng(seed)
    ce_samples: list[np.ndarray] = []
    acc_samples: list[np.ndarray] = []
    remaining = iterations
    while remaining:
        size = min(100, remaining)
        indices = rng.integers(0, len(eager_ce), size=(size, len(eager_ce)))
        left_ce = eager_ce[indices].mean(axis=1)
        right_ce = compiled_ce[indices].mean(axis=1)
        ce_samples.append((right_ce - left_ce) / left_ce)
        acc_samples.append(
            eager_acc[indices].mean(axis=1) - compiled_acc[indices].mean(axis=1)
        )
        remaining -= size
    return {
        "bootstrap_confidence": confidence,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "cross_entropy_relative_increase_observed": float(observed_ce),
        "cross_entropy_relative_increase_upper": float(
            np.quantile(np.concatenate(ce_samples), confidence)
        ),
        "masked_accuracy_decrease_observed": float(observed_acc),
        "masked_accuracy_decrease_upper": float(
            np.quantile(np.concatenate(acc_samples), confidence)
        ),
        "paired_sample_count": len(eager_ce),
    }


def audit_gate(
    *,
    run_root: Path,
    protocol_path: Path,
    bundle_root: Path,
) -> dict[str, Any]:
    """Audit source closure, paired execution, quality, and throughput."""
    protocol, protocol_sha256 = _load(protocol_path)
    _validate_master_protocol(protocol)
    source_root = Path(__file__).resolve().parents[1]
    for relative, expected in protocol["source"].items():
        if _sha256(source_root / relative) != expected:
            raise EfficiencyAuditError(f"Source closure differs: {relative}")
    gates: dict[str, tuple[dict[str, Any], str]] = {}
    for mode in ("compiled", "safe_eager"):
        value, digest = _load(bundle_root / f"{mode}_gate.json")
        if digest != protocol["mode_gate_file_sha256"][mode]:
            raise EfficiencyAuditError(f"Mode gate hash differs: {mode}")
        _validate_mode_gate(value, compiled=mode == "compiled")
        gates[mode] = (value, digest)
    artifact_audit = artifacts._verify_artifacts(run_root)
    receipts: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    allocation = None
    updates = int(protocol["batch"]["updates_per_run"])
    for label, mode in RUNS:
        gate, gate_sha256 = gates[mode]
        receipt, receipt_sha256 = _load(run_root / label / "run" / "receipt.json")
        state, _ = _load(run_root / label / "run" / "trainer_state.json")
        expected_optimization = gate["execution"]["optimization"]
        if (
            receipt.get("format_version") != RUN_FORMAT
            or receipt.get("run_role") != "continuous"
            or receipt.get("world_size") != 4
            or receipt.get("device") != GPU_MODEL
            or receipt.get("authorization") != RUN_AUTHORIZATION
            or receipt.get("test_access") != RUN_TEST_ACCESS
            or receipt.get("gate_protocol_sha256") != gate_sha256
            or receipt.get("optimization") != expected_optimization
            or receipt.get("resume_used") is not False
            or receipt.get("checkpoint_model_sha256")
            != receipt.get("final_model_sha256")
            or receipt.get("checkpoint_optimizer_sha256") is None
            or receipt.get("checkpoint_rng_manifest_sha256") is None
            or state.get("next_update") != updates
            or state.get("world_size") != 4
            or state.get("optimization") != expected_optimization
        ):
            raise EfficiencyAuditError(f"Run identity differs: {label}")
        identity = tuple(
            receipt.get("allocation", {}).get(key)
            for key in ("hostname", "slurm_job_id", "slurm_job_nodelist")
        )
        if allocation is None:
            allocation = identity
        elif identity != allocation:
            raise EfficiencyAuditError("Runs did not share one allocation")
        _validate_trace(receipt, updates)
        evaluation, _ = _load(run_root / label / "evaluation.json")
        evaluations[label] = _evaluation(
            evaluation,
            gate_sha256=gate_sha256,
            receipt_sha256=receipt_sha256,
            protocol=protocol,
        )
        receipts[label] = receipt
        telemetry[label] = _telemetry(run_root / label / "gpu.csv")
    if len({tuple(value["gpu_uuids"]) for value in telemetry.values()}) != 1:
        raise EfficiencyAuditError("GPU identities changed during crossover")
    if len({value["initial_model_sha256"] for value in receipts.values()}) != 1:
        raise EfficiencyAuditError("Initial model identity differs")
    replicate_state_equal = all(
        receipts[left][field] == receipts[right][field]
        for left, right in (
            ("safe_eager_r1", "safe_eager_r2"),
            ("compiled_r1", "compiled_r2"),
        )
        for field in ("final_model_sha256", "final_optimizer_sha256")
    )
    membership_fields = (
        "consumed_residue_tokens",
        "learning_rate",
        "micro_steps",
        "residue_tokens",
        "selected_targets",
        "sequence_count",
        "update",
    )
    reference = receipts[RUNS[0][0]]
    membership_equal = len({item["membership_sha256"] for item in receipts.values()}) == 1
    membership_equal = membership_equal and all(
        all(
            all(left[key] == right[key] for key in membership_fields)
            for left, right in zip(reference["trace"], receipt["trace"])
        )
        for receipt in receipts.values()
    )
    acceptance = protocol["acceptance"]
    summaries = {
        label: _throughput(
            receipt,
            int(acceptance["warmup_updates_excluded"]),
        )
        for label, receipt in receipts.items()
    }
    for label, summary in summaries.items():
        summary["maximum_memory_fraction"] = max(
            _finite(receipts[label]["peak_allocated_memory_mib"], "allocated memory"),
            _finite(receipts[label]["peak_reserved_memory_mib"], "reserved memory"),
            telemetry[label]["maximum_memory_mib"],
        ) / telemetry[label]["total_memory_mib"]
    round_pairs = {
        "r1": ("safe_eager_r1", "compiled_r1"),
        "r2": ("safe_eager_r2", "compiled_r2"),
    }
    short_horizon: dict[str, Any] = {}
    noninferiority: dict[str, Any] = {}
    throughput_ratios: dict[str, float] = {}
    for index, (round_name, (eager_name, compiled_name)) in enumerate(round_pairs.items()):
        limit = int(acceptance["short_horizon_updates"])
        eager_trace = receipts[eager_name]["trace"][:limit]
        compiled_trace = receipts[compiled_name]["trace"][:limit]
        maximum_loss = max(
            abs(_finite(left["loss"], "loss") - _finite(right["loss"], "loss"))
            for left, right in zip(eager_trace, compiled_trace)
        )
        maximum_gradient = max(
            abs(
                _finite(left["gradient_norm"], "gradient norm")
                - _finite(right["gradient_norm"], "gradient norm")
            )
            / max(abs(_finite(left["gradient_norm"], "gradient norm")), 1.0e-12)
            for left, right in zip(eager_trace, compiled_trace)
        )
        short_horizon[round_name] = {
            "maximum_gradient_norm_relative_difference": maximum_gradient,
            "maximum_loss_absolute_difference": maximum_loss,
            "passed": (
                maximum_gradient
                <= acceptance["maximum_short_horizon_gradient_norm_relative_difference"]
                and maximum_loss
                <= acceptance["maximum_short_horizon_loss_absolute_difference"]
            ),
        }
        value = _bootstrap_noninferiority(
            evaluations[eager_name],
            evaluations[compiled_name],
            iterations=int(acceptance["bootstrap_iterations"]),
            confidence=float(acceptance["bootstrap_confidence"]),
            seed=int(acceptance["bootstrap_seed"]) + index,
        )
        value["passed"] = (
            value["cross_entropy_relative_increase_upper"]
            <= acceptance["maximum_masked_cross_entropy_relative_increase"]
            and value["masked_accuracy_decrease_upper"]
            <= acceptance["maximum_masked_accuracy_absolute_decrease"]
        )
        noninferiority[round_name] = value
        throughput_ratios[round_name] = (
            summaries[compiled_name]["residue_tokens_per_second"]
            / summaries[eager_name]["residue_tokens_per_second"]
        )

    def pooled(mode: str) -> dict[str, float]:
        names = [label for label, item in RUNS if item == mode]
        tokens = sum(summaries[name]["nonpadding_residue_tokens"] for name in names)
        seconds = sum(summaries[name]["measured_seconds"] for name in names)
        rate = tokens / seconds
        return {
            "approximate_6n_mfu": (
                6.0 * PARAMETER_COUNT * rate / (4 * PEAK_DENSE_BF16_TFLOPS * 1.0e12)
            ),
            "residue_tokens_per_second": rate,
        }

    pooled_modes = {mode: pooled(mode) for mode in ("safe_eager", "compiled")}
    pooled_ratio = (
        pooled_modes["compiled"]["residue_tokens_per_second"]
        / pooled_modes["safe_eager"]["residue_tokens_per_second"]
    )
    replicate_ranges = {}
    for mode in ("safe_eager", "compiled"):
        values = [
            summaries[label]["residue_tokens_per_second"]
            for label, item in RUNS
            if item == mode
        ]
        replicate_ranges[mode] = (max(values) - min(values)) / min(values)
    gates = {
        "artifact_integrity": artifact_audit["manifest_verified"]
        and not artifact_audit["writable_paths"],
        "approximate_mfu": pooled_modes["compiled"]["approximate_6n_mfu"]
        >= acceptance["minimum_approximate_6n_mfu"],
        "evaluation_noninferiority": all(item["passed"] for item in noninferiority.values()),
        "membership": membership_equal,
        "memory": all(
            item["maximum_memory_fraction"] <= acceptance["maximum_memory_fraction"]
            for item in summaries.values()
        ),
        "replicate_stability": all(
            item <= acceptance["maximum_replicate_throughput_relative_range"]
            for item in replicate_ranges.values()
        ),
        "replicate_state": replicate_state_equal,
        "short_horizon_numerical": all(item["passed"] for item in short_horizon.values()),
        "step_stability": all(
            item["p95_over_p50"] <= acceptance["maximum_p95_over_p50_step_time"]
            for item in summaries.values()
        ),
        "throughput_each_round": all(
            item >= acceptance["compiled_relative_to_eager_each_round"]
            for item in throughput_ratios.values()
        ),
        "throughput_pooled": pooled_ratio
        >= acceptance["compiled_relative_to_eager_pooled"],
    }
    passed = all(gates.values())
    return {
        "artifact_audit": artifact_audit,
        "authorization": {
            "configuration_promotion": passed,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "format_version": FORMAT_VERSION,
        "gate_passed": passed,
        "gates": gates,
        "inputs": {
            "auditor_sha256": _sha256(Path(__file__)),
            "protocol_file_sha256": protocol_sha256,
            "run_root": str(run_root.resolve()),
        },
        "membership_equal": membership_equal,
        "noninferiority": noninferiority,
        "pooled": {
            "compiled_relative_to_eager": pooled_ratio,
            "modes": pooled_modes,
        },
        "replicate_throughput_relative_ranges": replicate_ranges,
        "runs": {
            label: {
                "evaluation_sha256": _sha256(run_root / label / "evaluation.json"),
                "receipt_sha256": _sha256(run_root / label / "run" / "receipt.json"),
                "summary": summaries[label],
                "telemetry": telemetry[label],
            }
            for label, _ in RUNS
        },
        "short_horizon": short_horizon,
        "test_access": protocol["test_access"],
        "throughput_ratios": throughput_ratios,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_gate(
        run_root=args.run_root,
        protocol_path=args.protocol,
        bundle_root=args.bundle_root,
    )
    _publish(args.output, result)


if __name__ == "__main__":
    main()
