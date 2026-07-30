#!/usr/bin/env python3
"""Audit the frozen production-batch eight-GPU crossover experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np


FORMAT_VERSION = "antibody_production_batch_efficiency_result_v1"
RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("fp32_compiled_r1", "fp32_compiled"),
    ("fp32_compiled_r2", "fp32_compiled"),
    ("safe_eager_r2", "safe_eager"),
)
PARAMETER_COUNT = 634_513_920
HARDWARE = "NVIDIA GeForce RTX 5090"
PEAK_DENSE_BF16_TFLOPS = 209.5


class ProductionBatchAuditError(RuntimeError):
    """Report an input, provenance, or acceptance-gate failure."""


def _reject_constant(value: str) -> None:
    raise ProductionBatchAuditError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProductionBatchAuditError(f"Duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _canonical_json_bytes(value: Any) -> bytes:
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


def _load(path: Path, *, canonical: bool = True) -> tuple[dict[str, Any], bytes]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProductionBatchAuditError(f"Input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionBatchAuditError(f"Input is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict) or (canonical and raw != _canonical_json_bytes(value)):
        raise ProductionBatchAuditError(f"Input is not canonical JSON: {path}")
    return value, raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProductionBatchAuditError(f"{name} must be finite")
    return result


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered or not 0.0 <= probability <= 1.0:
        raise ProductionBatchAuditError("Quantile inputs are invalid")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _expected_optimization(mode: str) -> dict[str, Any]:
    compiled = mode == "fp32_compiled"
    if mode not in {"safe_eager", "fp32_compiled"}:
        raise ProductionBatchAuditError(f"Unexpected mode: {mode}")
    return {
        "audit_gradient_rank_parity": True,
        "bucketed_padding": compiled,
        "compile_blocks": compiled,
        "compile_prewarm_buckets": compiled,
        "ddp_batched_grad_copy": False,
        "ddp_bucket_cap_mb": 25,
        "ddp_communication_dtype": "float32",
        "ddp_static_graph": False,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": False,
        "disable_ddp_gradient_as_bucket_view": False,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": compiled,
        "length_bucketing": False,
        "pinned_staging": compiled,
    }


def _validate_trace(receipt: Mapping[str, Any]) -> None:
    trace = receipt.get("trace")
    if not isinstance(trace, list) or len(trace) != 600:
        raise ProductionBatchAuditError("Every run must contain exactly 600 updates")
    previous_consumed = 0
    for expected_update, item in enumerate(trace, start=1):
        if not isinstance(item, Mapping) or item.get("update") != expected_update:
            raise ProductionBatchAuditError("Training updates are not contiguous")
        tokens = item.get("residue_tokens")
        consumed = item.get("consumed_residue_tokens")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
            or isinstance(consumed, bool)
            or not isinstance(consumed, int)
            or consumed != previous_consumed + tokens
            or int(item.get("micro_steps", 0)) <= 0
            or int(item.get("selected_targets", 0)) <= 0
            or int(item.get("sequence_count", 0)) <= 0
            or _finite(item.get("seconds"), "step seconds") <= 0.0
            or _finite(item.get("loss"), "loss") < 0.0
            or _finite(item.get("gradient_norm"), "gradient norm") < 0.0
        ):
            raise ProductionBatchAuditError("Training trace contains an invalid update")
        previous_consumed = consumed


def _evaluation(receipt: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = receipt.get("posttraining_evaluation")
    if not isinstance(value, Mapping) or value.get("sample_count") != 812:
        raise ProductionBatchAuditError("Posttraining evaluation inventory differs")
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) != 812:
        raise ProductionBatchAuditError("Posttraining evaluation samples differ")
    keys = np.asarray([int(item["sample_key"]) for item in samples], dtype=np.uint64)
    cross_entropy = np.asarray(
        [_finite(item["masked_cross_entropy"], "cross entropy") for item in samples],
        dtype=np.float64,
    )
    accuracy = np.asarray(
        [_finite(item["masked_accuracy"], "accuracy") for item in samples],
        dtype=np.float64,
    )
    if len(np.unique(keys)) != 812 or not np.array_equal(keys, np.sort(keys)):
        raise ProductionBatchAuditError("Evaluation sample keys are not unique and sorted")
    if np.any(cross_entropy < 0.0) or np.any(accuracy < 0.0) or np.any(accuracy > 1.0):
        raise ProductionBatchAuditError("Evaluation metrics are outside their domains")
    selected_targets = sum(int(item["selected_targets"]) for item in samples)
    correct = sum(int(item["correct"]) for item in samples)
    loss_sum = sum(_finite(item["loss_sum"], "loss sum") for item in samples)
    if (
        selected_targets != value.get("selected_targets")
        or not math.isclose(
            correct / selected_targets,
            _finite(value.get("masked_accuracy"), "aggregate accuracy"),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            loss_sum / selected_targets,
            _finite(value.get("masked_cross_entropy"), "aggregate cross entropy"),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ProductionBatchAuditError("Evaluation aggregates do not replay")
    return keys, cross_entropy, accuracy


def _rank_state_digest(receipt: Mapping[str, Any], prefix: str) -> str:
    """Require exact replicated eight-rank model or optimizer state parity."""
    field = f"{prefix}_rank_sha256"
    scalar = f"{prefix}_sha256"
    values = receipt.get(field)
    if (
        not isinstance(values, list)
        or len(values) != 8
        or len(set(values)) != 1
        or (prefix != "initial_model" and set(values) != {receipt.get(scalar)})
    ):
        raise ProductionBatchAuditError(f"Eight-rank {prefix} parity differs")
    return str(values[0])


def _throughput(receipt: Mapping[str, Any], warmup: int) -> dict[str, Any]:
    trace = receipt["trace"][warmup:]
    seconds = [_finite(item["seconds"], "step seconds") for item in trace]
    tokens = sum(int(item["residue_tokens"]) for item in trace)
    total_seconds = sum(seconds)
    rate = tokens / total_seconds
    return {
        "approximate_6n_mfu": (
            6.0
            * PARAMETER_COUNT
            * rate
            / (8 * PEAK_DENSE_BF16_TFLOPS * 1.0e12)
        ),
        "measured_seconds": total_seconds,
        "measured_updates": len(trace),
        "nonpadding_residue_tokens": tokens,
        "p50_step_seconds": _quantile(seconds, 0.50),
        "p95_over_p50": _quantile(seconds, 0.95) / _quantile(seconds, 0.50),
        "p95_step_seconds": _quantile(seconds, 0.95),
        "residue_tokens_per_second": rate,
        "warmup_updates_excluded": warmup,
    }


def _telemetry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii", newline="") as handle:
        rows = [
            {key.strip(): item.strip() for key, item in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ProductionBatchAuditError("GPU telemetry is empty")

    def number(row: Mapping[str, str], key: str) -> float:
        return _finite(row[key].split()[0], key)

    names = {row["name"] for row in rows}
    indices = {int(row["index"]) for row in rows}
    uuids = {row["uuid"] for row in rows}
    totals = {number(row, "memory.total [MiB]") for row in rows}
    if names != {HARDWARE} or indices != set(range(8)) or len(uuids) != 8 or len(totals) != 1:
        raise ProductionBatchAuditError("GPU telemetry identity differs")
    return {
        "file_sha256": _sha256(path),
        "gpu_uuids": sorted(uuids),
        "maximum_memory_mib": max(number(row, "memory.used [MiB]") for row in rows),
        "sample_count": len(rows),
        "total_memory_mib": next(iter(totals)),
    }


def _bootstrap_noninferiority(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if not np.array_equal(reference[0], candidate[0]):
        raise ProductionBatchAuditError("Paired evaluation sample keys differ")
    size = len(reference[0])
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    ce_effect = np.empty(iterations, dtype=np.float64)
    accuracy_effect = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 256):
        count = min(256, iterations - start)
        indices = generator.integers(0, size, size=(count, size))
        ce_effect[start : start + count] = (
            candidate[1][indices].mean(axis=1) / reference[1][indices].mean(axis=1) - 1.0
        )
        accuracy_effect[start : start + count] = (
            reference[2][indices].mean(axis=1) - candidate[2][indices].mean(axis=1)
        )
    return {
        "bootstrap_confidence": confidence,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "cross_entropy_relative_increase_observed": float(
            candidate[1].mean() / reference[1].mean() - 1.0
        ),
        "cross_entropy_relative_increase_upper": float(
            np.quantile(ce_effect, confidence, method="linear")
        ),
        "masked_accuracy_decrease_observed": float(
            reference[2].mean() - candidate[2].mean()
        ),
        "masked_accuracy_decrease_upper": float(
            np.quantile(accuracy_effect, confidence, method="linear")
        ),
        "paired_sample_count": size,
    }


def _verify_artifacts(run_root: Path) -> dict[str, Any]:
    manifest = run_root / "artifact_sha256.txt"
    expected_files = {
        path.resolve()
        for path in run_root.rglob("*")
        if path.is_file() and path.name != manifest.name
    }
    if any(path.is_symlink() for path in run_root.rglob("*")):
        raise ProductionBatchAuditError("Artifact tree contains a symbolic link")
    observed: set[Path] = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, separator, raw_path = line.partition("  ")
        path = Path(raw_path).resolve()
        if not separator or path in observed or run_root.resolve() not in path.parents:
            raise ProductionBatchAuditError("Artifact manifest line is invalid")
        if _sha256(path) != digest:
            raise ProductionBatchAuditError(f"Artifact digest differs: {path}")
        observed.add(path)
    if observed != expected_files:
        raise ProductionBatchAuditError("Artifact manifest inventory differs")
    writable = [
        str(path)
        for path in expected_files | {manifest.resolve()}
        if path.stat().st_mode & 0o222
    ]
    return {
        "file_count": len(expected_files) + 1,
        "manifest_file_sha256": _sha256(manifest),
        "manifest_verified": True,
        "writable_paths": writable,
    }


def audit_production_batch(
    *,
    run_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol, protocol_raw = _load(protocol_path, canonical=False)
    if protocol.get("format_version") != "antibody_production_batch_efficiency_protocol_v1":
        raise ProductionBatchAuditError("Production-batch protocol format differs")
    if protocol["measurement"]["crossover_run_order"] != [name for name, _ in RUNS]:
        raise ProductionBatchAuditError("Crossover run order differs")
    if _sha256(run_root / "protocol.json") != hashlib.sha256(protocol_raw).hexdigest():
        raise ProductionBatchAuditError("Run-local protocol differs")
    artifact_audit = _verify_artifacts(run_root)
    receipts: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    allocation_identity = None
    for name, mode in RUNS:
        receipt, _ = _load(run_root / name / "run/receipt.json")
        state, _ = _load(run_root / name / "run/trainer_state.json")
        expected_optimization = _expected_optimization(mode)
        if (
            receipt.get("format_version") != "antibody_pretraining_optimized_pilot_v1"
            or receipt.get("backend") != "ddp"
            or receipt.get("world_size") != 8
            or receipt.get("device") != HARDWARE
            or receipt.get("run_seed") != 97
            or receipt.get("preset") != "abf_650m"
            or receipt.get("batch_quantum") != 512
            or receipt.get("micro_batch_per_gpu") != 16
            or receipt.get("target_residue_tokens") != 262_144
            or receipt.get("parameter_moments", {}).get("count") != PARAMETER_COUNT
            or receipt.get("optimization") != expected_optimization
            or receipt.get("checkpoint_format") is not None
            or receipt.get("resume_used") is not False
            or receipt.get("authorization")
            != {"production_pretraining": False, "scientific_claim": False}
            or receipt.get("test_access") != protocol["test_access"]
            or state.get("next_update") != 600
            or state.get("training_universe_count") != 7300
            or state.get("optimization") != expected_optimization
        ):
            raise ProductionBatchAuditError(f"Receipt or state differs for {name}")
        expected_inputs = {
            "ddp_communication_protocol_sha256": protocol["inputs"]["communication_protocol_sha256"],
            "efficiency_protocol_sha256": protocol["inputs"]["efficiency_protocol_sha256"],
            "model_family_sha256": protocol["inputs"]["model_family_sha256"],
            "parity_protocol_sha256": protocol["inputs"]["parity_protocol_sha256"],
            "parity_split_receipt_file_sha256": protocol["inputs"]["parity_split_receipt_file_sha256"],
            "shard_manifest_sha256": protocol["inputs"]["dense_manifest_payload_sha256"],
            "training_protocol_sha256": protocol["inputs"]["training_protocol_sha256"],
        }
        if any(receipt.get(key) != value for key, value in expected_inputs.items()):
            raise ProductionBatchAuditError(f"Input provenance differs for {name}")
        identity = tuple(
            receipt.get("allocation", {}).get(field)
            for field in ("hostname", "slurm_job_id", "slurm_job_nodelist")
        )
        if allocation_identity is None:
            allocation_identity = identity
        elif identity != allocation_identity:
            raise ProductionBatchAuditError("Runs did not share one allocation")
        _validate_trace(receipt)
        _rank_state_digest(receipt, "initial_model")
        _rank_state_digest(receipt, "final_model")
        _rank_state_digest(receipt, "final_optimizer")
        receipts[name] = receipt
        evaluations[name] = _evaluation(receipt)
        telemetry[name] = _telemetry(run_root / name / "gpu.csv")
    if len({tuple(value["gpu_uuids"]) for value in telemetry.values()}) != 1:
        raise ProductionBatchAuditError("GPU identity changed during the crossover")
    if len({_rank_state_digest(value, "initial_model") for value in receipts.values()}) != 1:
        raise ProductionBatchAuditError("Initial model identity differs across runs")
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
    membership_equal = len({value["membership_sha256"] for value in receipts.values()}) == 1
    membership_equal = membership_equal and all(
        all(
            all(left[field] == right[field] for field in membership_fields)
            for left, right in zip(reference["trace"], receipt["trace"])
        )
        for receipt in receipts.values()
    )
    acceptance = protocol["acceptance"]
    summaries = {
        name: _throughput(receipt, int(acceptance["warmup_updates_excluded"]))
        for name, receipt in receipts.items()
    }
    for name in summaries:
        summaries[name]["maximum_memory_fraction"] = max(
            _finite(receipts[name]["peak_allocated_memory_mib"], "allocated memory"),
            _finite(receipts[name]["peak_reserved_memory_mib"], "reserved memory"),
            telemetry[name]["maximum_memory_mib"],
        ) / telemetry[name]["total_memory_mib"]
    round_pairs = {
        "r1": ("safe_eager_r1", "fp32_compiled_r1"),
        "r2": ("safe_eager_r2", "fp32_compiled_r2"),
    }
    short_horizon: dict[str, Any] = {}
    noninferiority: dict[str, Any] = {}
    throughput_ratios: dict[str, float] = {}
    for round_index, (round_name, (eager_name, compiled_name)) in enumerate(
        round_pairs.items()
    ):
        eager_trace = receipts[eager_name]["trace"]
        compiled_trace = receipts[compiled_name]["trace"]
        limit = int(acceptance["short_horizon_updates"])
        maximum_loss = max(
            abs(_finite(left["loss"], "loss") - _finite(right["loss"], "loss"))
            for left, right in zip(eager_trace[:limit], compiled_trace[:limit])
        )
        maximum_gradient = max(
            abs(
                _finite(left["gradient_norm"], "gradient norm")
                - _finite(right["gradient_norm"], "gradient norm")
            )
            / max(abs(_finite(left["gradient_norm"], "gradient norm")), 1.0e-12)
            for left, right in zip(eager_trace[:limit], compiled_trace[:limit])
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
        result = _bootstrap_noninferiority(
            evaluations[eager_name],
            evaluations[compiled_name],
            iterations=int(acceptance["bootstrap_iterations"]),
            confidence=float(acceptance["bootstrap_confidence"]),
            seed=int(acceptance["bootstrap_seed"]) + round_index,
        )
        result["passed"] = (
            result["cross_entropy_relative_increase_upper"]
            <= acceptance["maximum_masked_cross_entropy_relative_increase"]
            and result["masked_accuracy_decrease_upper"]
            <= acceptance["maximum_masked_accuracy_absolute_decrease"]
        )
        noninferiority[round_name] = result
        throughput_ratios[round_name] = (
            summaries[compiled_name]["residue_tokens_per_second"]
            / summaries[eager_name]["residue_tokens_per_second"]
        )

    def pooled(mode: str) -> dict[str, float]:
        names = [name for name, run_mode in RUNS if run_mode == mode]
        tokens = sum(summaries[name]["nonpadding_residue_tokens"] for name in names)
        seconds = sum(summaries[name]["measured_seconds"] for name in names)
        throughput = tokens / seconds
        return {
            "approximate_6n_mfu": (
                6.0
                * PARAMETER_COUNT
                * throughput
                / (8 * PEAK_DENSE_BF16_TFLOPS * 1.0e12)
            ),
            "residue_tokens_per_second": throughput,
        }

    pooled_modes = {mode: pooled(mode) for mode in ("safe_eager", "fp32_compiled")}
    pooled_ratio = (
        pooled_modes["fp32_compiled"]["residue_tokens_per_second"]
        / pooled_modes["safe_eager"]["residue_tokens_per_second"]
    )
    replicate_ranges = {}
    for mode in ("safe_eager", "fp32_compiled"):
        values = [
            summaries[name]["residue_tokens_per_second"]
            for name, run_mode in RUNS
            if run_mode == mode
        ]
        replicate_ranges[mode] = (max(values) - min(values)) / min(values)
    gates = {
        "artifact_integrity": artifact_audit["manifest_verified"] and not artifact_audit["writable_paths"],
        "approximate_mfu": pooled_modes["fp32_compiled"]["approximate_6n_mfu"]
        >= acceptance["minimum_approximate_6n_mfu"],
        "evaluation_noninferiority": all(value["passed"] for value in noninferiority.values()),
        "membership": membership_equal,
        "memory": all(
            value["maximum_memory_fraction"] <= acceptance["maximum_memory_fraction"]
            for value in summaries.values()
        ),
        "replicate_stability": all(
            value <= acceptance["maximum_replicate_throughput_relative_range"]
            for value in replicate_ranges.values()
        ),
        "short_horizon_numerical": all(value["passed"] for value in short_horizon.values()),
        "step_stability": all(
            value["p95_over_p50"] <= acceptance["maximum_p95_over_p50_step_time"]
            for value in summaries.values()
        ),
        "throughput_each_round": all(
            value >= acceptance["compiled_relative_to_eager_each_round"]
            for value in throughput_ratios.values()
        ),
        "throughput_pooled": pooled_ratio
        >= acceptance["compiled_relative_to_eager_pooled"],
    }
    gate_passed = all(gates.values())
    return {
        "artifact_audit": artifact_audit,
        "authorization": {
            "configuration_promotion": gate_passed,
            "production_pretraining": False,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
            "scientific_claim": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "format_version": FORMAT_VERSION,
        "gate_passed": gate_passed,
        "gates": gates,
        "inputs": {
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_file_sha256": hashlib.sha256(protocol_raw).hexdigest(),
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
            name: {
                "receipt_sha256": _sha256(run_root / name / "run/receipt.json"),
                "summary": summaries[name],
                "telemetry": telemetry[name],
            }
            for name, _ in RUNS
        },
        "short_horizon": short_horizon,
        "test_access": protocol["test_access"],
        "throughput_ratios": throughput_ratios,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = audit_production_batch(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
    )
    _publish(arguments.output, result)
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
