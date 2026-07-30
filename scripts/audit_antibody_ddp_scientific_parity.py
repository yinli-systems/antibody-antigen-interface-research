#!/usr/bin/env python3
"""Audit the frozen nine-run DDP scientific-parity experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


RUNS = (
    ("seed17_safe_eager", 17, "safe_eager"),
    ("seed17_fp32_compiled", 17, "fp32_compiled"),
    ("seed17_bf16_selected", 17, "bf16_selected"),
    ("seed42_bf16_selected", 42, "bf16_selected"),
    ("seed42_safe_eager", 42, "safe_eager"),
    ("seed42_fp32_compiled", 42, "fp32_compiled"),
    ("seed73_fp32_compiled", 73, "fp32_compiled"),
    ("seed73_bf16_selected", 73, "bf16_selected"),
    ("seed73_safe_eager", 73, "safe_eager"),
)
MODES = ("safe_eager", "fp32_compiled", "bf16_selected")
EXPECTED_PARAMETER_COUNT = 634_513_920


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="ascii"),
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _quantile(values: list[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("Quantile inputs are invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def _verify_artifact_manifest(run_root: Path) -> dict[str, Any]:
    manifest = run_root / "artifact_sha256.txt"
    lines = manifest.read_text(encoding="ascii").splitlines()
    symlinks = [str(path) for path in run_root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError("Artifact trees cannot contain symbolic links")
    expected_files = {
        path.resolve()
        for path in run_root.rglob("*")
        if path.is_file() and path.name != manifest.name
    }
    observed_files: set[Path] = set()
    for line in lines:
        digest, separator, raw_path = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError("Artifact manifest line is malformed")
        unresolved = Path(raw_path)
        if not unresolved.is_absolute():
            raise ValueError("Artifact manifest paths must be absolute")
        path = unresolved.resolve()
        if path in observed_files or run_root.resolve() not in path.parents:
            raise ValueError("Artifact manifest path is duplicated or escapes the run")
        if _sha256(path) != digest:
            raise ValueError(f"Artifact digest differs: {path}")
        observed_files.add(path)
    if observed_files != expected_files:
        raise ValueError("Artifact manifest inventory differs from the run tree")
    writable = [
        str(path)
        for path in run_root.rglob("*")
        if path.is_file() and path.stat().st_mode & 0o222
    ]
    return {
        "file_count": len(expected_files) + 1,
        "manifest_file_sha256": _sha256(manifest),
        "manifest_verified": True,
        "writable_paths": writable,
    }


def _expected_optimization(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unknown optimization mode: {mode}")
    optimized = mode != "safe_eager"
    return {
        "audit_gradient_rank_parity": True,
        "bucketed_padding": optimized,
        "compile_blocks": optimized,
        "compile_prewarm_buckets": optimized,
        "ddp_batched_grad_copy": mode == "bf16_selected",
        "ddp_bucket_cap_mb": 25,
        "ddp_communication_dtype": (
            "bfloat16" if mode == "bf16_selected" else "float32"
        ),
        "ddp_static_graph": False,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": False,
        "disable_ddp_gradient_as_bucket_view": False,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": optimized,
        "length_bucketing": False,
        "pinned_staging": optimized,
    }


def _validate_trace(receipt: dict[str, Any], name: str) -> None:
    trace = receipt.get("trace")
    if not isinstance(trace, list) or len(trace) != 500:
        raise ValueError(f"Every run must contain exactly 500 updates: {name}")
    previous_consumed = 0
    for expected_update, item in enumerate(trace, start=1):
        if item.get("update") != expected_update:
            raise ValueError(f"Training updates are not contiguous: {name}")
        consumed = int(item["consumed_residue_tokens"])
        residue_tokens = int(item["residue_tokens"])
        loss = _finite(item["loss"], "loss")
        gradient_norm = _finite(item["gradient_norm"], "gradient norm")
        learning_rate = _finite(item["learning_rate"], "learning rate")
        if (
            consumed <= previous_consumed
            or residue_tokens <= 0
            or consumed != previous_consumed + residue_tokens
            or int(item["selected_targets"]) <= 0
            or int(item["sequence_count"]) <= 0
            or int(item["micro_steps"]) <= 0
            or _finite(item["seconds"], "step seconds") <= 0.0
            or loss < 0.0
            or gradient_norm < 0.0
            or learning_rate < 0.0
        ):
            raise ValueError(f"Training trace contains an invalid update: {name}")
        previous_consumed = consumed


def _initial_model_digest(receipt: dict[str, Any]) -> str:
    values = receipt.get("initial_model_rank_sha256")
    if not isinstance(values, list) or len(values) != 8 or len(set(values)) != 1:
        raise ValueError("Initial model does not have exact eight-rank parity")
    return str(values[0])


def _evaluation(receipt: dict[str, Any], stage: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = receipt.get(stage)
    if not isinstance(value, dict) or value.get("sample_count") != 812:
        raise ValueError(f"{stage} evaluation inventory differs")
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) != 812:
        raise ValueError(f"{stage} sample records differ")
    keys = np.asarray([int(item["sample_key"]) for item in samples], dtype=np.uint64)
    cross_entropy = np.asarray(
        [_finite(item["masked_cross_entropy"], "sample cross entropy") for item in samples],
        dtype=np.float64,
    )
    accuracy = np.asarray(
        [_finite(item["masked_accuracy"], "sample accuracy") for item in samples],
        dtype=np.float64,
    )
    if len(np.unique(keys)) != len(keys) or not np.array_equal(keys, np.sort(keys)):
        raise ValueError(f"{stage} sample keys must be unique and sorted")
    if np.any(cross_entropy < 0.0) or np.any(accuracy < 0.0) or np.any(accuracy > 1.0):
        raise ValueError(f"{stage} sample metrics are outside their domains")
    total_targets = sum(int(item["selected_targets"]) for item in samples)
    total_correct = sum(int(item["correct"]) for item in samples)
    total_loss = sum(_finite(item["loss_sum"], "sample loss sum") for item in samples)
    if (
        total_targets != value.get("selected_targets")
        or not math.isclose(total_correct / total_targets, _finite(value["masked_accuracy"], "accuracy"), rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(total_loss / total_targets, _finite(value["masked_cross_entropy"], "cross entropy"), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"{stage} aggregate metrics do not replay")
    return keys, cross_entropy, accuracy


def _throughput(receipt: dict[str, Any], warmup: int) -> dict[str, float | int]:
    trace = receipt.get("trace")
    if not isinstance(trace, list) or len(trace) != 500:
        raise ValueError("Every run must contain exactly 500 updates")
    measured = trace[warmup:]
    seconds = [_finite(item["seconds"], "step seconds") for item in measured]
    if any(value <= 0.0 for value in seconds):
        raise ValueError("Step seconds must be positive")
    tokens = sum(int(item["residue_tokens"]) for item in measured)
    p50 = _quantile(seconds, 0.50)
    p95 = _quantile(seconds, 0.95)
    return {
        "measured_updates": len(measured),
        "p50_step_seconds": p50,
        "p95_step_seconds": p95,
        "p95_over_p50": p95 / p50,
        "residue_tokens": tokens,
        "residue_tokens_per_second": tokens / sum(seconds),
        "total_seconds": sum(seconds),
        "warmup_updates_excluded": warmup,
    }


def _telemetry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii", newline="") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError("GPU telemetry is empty")

    def number(row: dict[str, str], key: str) -> float:
        return _finite(row[key].split()[0], key)

    indices = {int(row["index"]) for row in rows}
    uuids = {row["uuid"] for row in rows}
    names = {row["name"] for row in rows}
    totals = {number(row, "memory.total [MiB]") for row in rows}
    if indices != set(range(8)) or len(uuids) != 8 or names != {"NVIDIA GeForce RTX 5090"} or len(totals) != 1:
        raise ValueError("GPU telemetry identity differs")
    return {
        "file_sha256": _sha256(path),
        "gpu_uuids": sorted(uuids),
        "maximum_memory_mib": max(number(row, "memory.used [MiB]") for row in rows),
        "row_count": len(rows),
        "total_memory_mib": next(iter(totals)),
    }


def _bootstrap_upper_bounds(
    reference_cross_entropy: np.ndarray,
    candidate_cross_entropy: np.ndarray,
    reference_accuracy: np.ndarray,
    candidate_accuracy: np.ndarray,
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, float | int]:
    arrays = (
        reference_cross_entropy,
        candidate_cross_entropy,
        reference_accuracy,
        candidate_accuracy,
    )
    size = len(reference_cross_entropy)
    if size == 0 or any(array.shape != (size,) for array in arrays):
        raise ValueError("Paired bootstrap arrays must be nonempty and aligned")
    generator = np.random.Generator(np.random.PCG64(seed))
    ce_effect = np.empty(iterations, dtype=np.float64)
    accuracy_effect = np.empty(iterations, dtype=np.float64)
    for offset in range(0, iterations, 256):
        stop = min(iterations, offset + 256)
        indices = generator.integers(0, size, size=(stop - offset, size), endpoint=False)
        reference_ce_means = reference_cross_entropy[indices].mean(axis=1)
        candidate_ce_means = candidate_cross_entropy[indices].mean(axis=1)
        ce_effect[offset:stop] = candidate_ce_means / reference_ce_means - 1.0
        accuracy_effect[offset:stop] = (
            reference_accuracy[indices].mean(axis=1)
            - candidate_accuracy[indices].mean(axis=1)
        )
    return {
        "bootstrap_confidence": confidence,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "cross_entropy_relative_increase_observed": float(
            candidate_cross_entropy.mean() / reference_cross_entropy.mean() - 1.0
        ),
        "cross_entropy_relative_increase_upper": float(
            np.quantile(ce_effect, confidence, method="linear")
        ),
        "masked_accuracy_decrease_observed": float(
            reference_accuracy.mean() - candidate_accuracy.mean()
        ),
        "masked_accuracy_decrease_upper": float(
            np.quantile(accuracy_effect, confidence, method="linear")
        ),
        "paired_sample_count": size,
    }


def _publish(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    execution = _load(args.execution)
    registration = _load(args.registration)
    gate = _load(args.gate)
    if execution.get("format_version") != "antibody_ddp_scientific_parity_execution_v2_1":
        raise ValueError("Unexpected execution protocol")
    if registration.get("format_version") not in {
        "antibody_ddp_scientific_parity_registration_v1",
        "antibody_ddp_scientific_parity_registration_v2",
        "antibody_ddp_scientific_parity_registration_v3",
        "antibody_ddp_scientific_parity_registration_v4",
    }:
        raise ValueError("Unexpected registration")
    if gate.get("format_version") != "antibody_ddp_scientific_parity_gate_v2":
        raise ValueError("Unexpected gate protocol")
    if execution["inputs"]["gate_protocol_sha256"] != _sha256(args.gate):
        raise ValueError("Gate protocol digest differs")
    if registration["execution_sha256"] != _sha256(args.execution):
        raise ValueError("Execution protocol digest differs")
    if execution.get("test_access") != {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    } or registration.get("test_access") != execution["test_access"]:
        raise ValueError("Protocol test-access boundary differs")
    if [name for name, _, _ in RUNS] != execution["run_order"]:
        raise ValueError("Run order differs")
    if _sha256(run_root / "execution_protocol.json") != _sha256(args.execution):
        raise ValueError("Run-local execution protocol differs")
    if _sha256(run_root / "registration.json") != _sha256(args.registration):
        raise ValueError("Run-local registration differs")
    artifact_audit = _verify_artifact_manifest(run_root)
    receipts: dict[str, dict[str, Any]] = {}
    states: dict[str, dict[str, Any]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    common_allocation: tuple[str | None, str | None, str | None] | None = None
    for name, seed, mode in RUNS:
        root = run_root / name
        receipt = _load(root / "run/receipt.json")
        state = _load(root / "run/trainer_state.json")
        if (
            receipt.get("format_version") != "antibody_pretraining_optimized_pilot_v1"
            or receipt.get("backend") != "ddp"
            or receipt.get("run_seed") != seed
            or receipt.get("world_size") != 8
            or receipt.get("preset") != "abf_650m"
            or receipt.get("batch_quantum") != 128
            or receipt.get("micro_batch_per_gpu") != 16
            or receipt.get("target_residue_tokens") != 16384
            or receipt.get("deterministic_algorithms") is not True
            or receipt.get("device") != "NVIDIA GeForce RTX 5090"
            or receipt.get("model_family_sha256")
            != execution["inputs"]["model_family_sha256"]
            or receipt.get("training_protocol_sha256")
            != execution["inputs"]["training_protocol_sha256"]
            or receipt.get("efficiency_protocol_sha256")
            != execution["inputs"]["efficiency_protocol_sha256"]
            or receipt.get("ddp_communication_protocol_sha256")
            != execution["inputs"]["communication_protocol_sha256"]
            or receipt.get("parity_evaluation_count") != 812
            or receipt.get("parity_protocol_sha256") != _sha256(args.gate)
            or receipt.get("parity_split_receipt_file_sha256")
            != execution["inputs"]["split_receipt_file_sha256"]
            or receipt.get("shard_manifest_sha256")
            != execution["inputs"]["dense_manifest_payload_sha256"]
            or receipt.get("authorization")
            != {"production_pretraining": False, "scientific_claim": False}
            or receipt.get("test_access")
            != {"downstream_labels_used": False, "sabdab2_heldout_accessed": False}
            or receipt.get("resume_used") is not False
            or receipt.get("checkpoint_format") is not None
            or receipt.get("parameter_moments", {}).get("count")
            != EXPECTED_PARAMETER_COUNT
            or state.get("backend") != "ddp"
            or state.get("world_size") != 8
            or state.get("training_universe_count") != 7300
            or state.get("run_seed") != seed
            or state.get("next_update") != 500
            or state.get("parity_protocol_sha256") != _sha256(args.gate)
            or state.get("parity_split_receipt_file_sha256")
            != execution["inputs"]["split_receipt_file_sha256"]
        ):
            raise ValueError(f"Receipt or state provenance differs for {name}")
        expected_mode = _expected_optimization(mode)
        if receipt.get("optimization") != expected_mode:
            raise ValueError(f"Optimization mode differs for {name}")
        if state.get("optimization") != expected_mode:
            raise ValueError(f"Trainer-state optimization mode differs for {name}")
        _validate_trace(receipt, name)
        _initial_model_digest(receipt)
        for key, scalar_key in (
            ("final_model_rank_sha256", "final_model_sha256"),
            ("final_optimizer_rank_sha256", "final_optimizer_sha256"),
        ):
            values = receipt.get(key)
            if (
                not isinstance(values, list)
                or len(values) != 8
                or set(values) != {receipt.get(scalar_key)}
            ):
                raise ValueError(f"Eight-rank state parity differs for {name}")
        allocation = receipt.get("allocation", {})
        identity = (
            allocation.get("hostname"),
            allocation.get("slurm_job_id"),
            allocation.get("slurm_job_nodelist"),
        )
        if common_allocation is None:
            common_allocation = identity
        elif identity != common_allocation:
            raise ValueError("Runs did not execute in the same allocation")
        receipts[name] = receipt
        states[name] = state
        telemetry[name] = _telemetry(root / "gpu.csv")
        evaluations[name] = {
            "pre": _evaluation(receipt, "pretraining_evaluation"),
            "post": _evaluation(receipt, "posttraining_evaluation"),
        }
        if not np.array_equal(evaluations[name]["pre"][0], evaluations[name]["post"][0]):
            raise ValueError(f"Pre/post evaluation keys differ for {name}")
    if len({tuple(value["gpu_uuids"]) for value in telemetry.values()}) != 1:
        raise ValueError("GPU identity changed within the allocation")

    membership_fields = (
        "consumed_residue_tokens",
        "learning_rate",
        "micro_steps",
        "residue_tokens",
        "selected_targets",
        "sequence_count",
        "update",
    )
    membership_gates: dict[str, bool] = {}
    for seed in (17, 42, 73):
        seed_names = [name for name, run_seed, _ in RUNS if run_seed == seed]
        reference = receipts[seed_names[0]]
        membership_gates[str(seed)] = (
            len({receipts[name]["membership_sha256"] for name in seed_names}) == 1
            and len({_initial_model_digest(receipts[name]) for name in seed_names}) == 1
            and all(
                all(
                    all(left[field] == right[field] for field in membership_fields)
                    for left, right in zip(reference["trace"], receipts[name]["trace"])
                )
                for name in seed_names[1:]
            )
        )

    short = gate["acceptance"]["short_horizon_numerical"]
    short_results: dict[str, dict[str, Any]] = {}
    for seed in (17, 42, 73):
        reference = receipts[f"seed{seed}_safe_eager"]
        for mode in ("fp32_compiled", "bf16_selected"):
            candidate = receipts[f"seed{seed}_{mode}"]
            prefix = int(short["required_updates"])
            maximum_loss = max(
                abs(_finite(left["loss"], "loss") - _finite(right["loss"], "loss"))
                for left, right in zip(reference["trace"][:prefix], candidate["trace"][:prefix])
            )
            maximum_gradient = max(
                abs(_finite(left["gradient_norm"], "gradient") - _finite(right["gradient_norm"], "gradient"))
                / max(abs(_finite(left["gradient_norm"], "gradient")), 1e-12)
                for left, right in zip(reference["trace"][:prefix], candidate["trace"][:prefix])
            )
            short_results[f"seed{seed}_{mode}_vs_safe_eager"] = {
                "maximum_gradient_norm_relative_difference": maximum_gradient,
                "maximum_loss_absolute_difference": maximum_loss,
                "passed": (
                    maximum_loss <= short["maximum_loss_absolute_difference"]
                    and maximum_gradient <= short["maximum_gradient_norm_relative_difference"]
                ),
            }

    learning_policy = gate["acceptance"]["learning"]
    learning: dict[str, dict[str, Any]] = {}
    for name, _, _ in RUNS:
        pre = receipts[name]["pretraining_evaluation"]
        post = receipts[name]["posttraining_evaluation"]
        ce_ratio = _finite(post["masked_cross_entropy"], "post CE") / _finite(pre["masked_cross_entropy"], "pre CE")
        accuracy_gain = _finite(post["masked_accuracy"], "post accuracy") - _finite(pre["masked_accuracy"], "pre accuracy")
        learning[name] = {
            "post_over_pre_masked_cross_entropy": ce_ratio,
            "post_minus_pre_masked_accuracy": accuracy_gain,
            "passed": (
                ce_ratio <= learning_policy["maximum_post_over_pre_masked_cross_entropy_each_run"]
                and accuracy_gain >= learning_policy["minimum_post_minus_pre_masked_accuracy_each_run"]
            ),
        }

    noninferiority = gate["acceptance"]["evaluation_noninferiority"]
    per_seed: dict[str, dict[str, Any]] = {}
    bootstrap: dict[str, dict[str, Any]] = {}
    for comparison in noninferiority["comparisons"]:
        candidate_mode = comparison["candidate"]
        reference_mode = comparison["reference"]
        candidate_ce_by_seed = []
        reference_ce_by_seed = []
        candidate_acc_by_seed = []
        reference_acc_by_seed = []
        shared_keys: np.ndarray | None = None
        for seed in (17, 42, 73):
            candidate_eval = evaluations[f"seed{seed}_{candidate_mode}"]["post"]
            reference_eval = evaluations[f"seed{seed}_{reference_mode}"]["post"]
            if not np.array_equal(candidate_eval[0], reference_eval[0]):
                raise ValueError("Paired evaluation sample keys differ")
            shared_keys = candidate_eval[0] if shared_keys is None else shared_keys
            if not np.array_equal(shared_keys, candidate_eval[0]):
                raise ValueError("Evaluation sample keys differ across seeds")
            ce_increase = float(candidate_eval[1].mean() / reference_eval[1].mean() - 1.0)
            accuracy_decrease = float(reference_eval[2].mean() - candidate_eval[2].mean())
            passed = (
                ce_increase <= noninferiority["maximum_masked_cross_entropy_relative_increase_each_seed"]
                and accuracy_decrease <= noninferiority["maximum_masked_accuracy_absolute_decrease_each_seed"]
            )
            per_seed[f"seed{seed}_{candidate_mode}_vs_{reference_mode}"] = {
                "cross_entropy_relative_increase": ce_increase,
                "masked_accuracy_decrease": accuracy_decrease,
                "passed": passed,
            }
            candidate_ce_by_seed.append(candidate_eval[1])
            reference_ce_by_seed.append(reference_eval[1])
            candidate_acc_by_seed.append(candidate_eval[2])
            reference_acc_by_seed.append(reference_eval[2])
        result = _bootstrap_upper_bounds(
            np.stack(reference_ce_by_seed).mean(axis=0),
            np.stack(candidate_ce_by_seed).mean(axis=0),
            np.stack(reference_acc_by_seed).mean(axis=0),
            np.stack(candidate_acc_by_seed).mean(axis=0),
            iterations=int(noninferiority["bootstrap_iterations"]),
            seed=int(noninferiority["bootstrap_seed"]),
            confidence=float(noninferiority["bootstrap_confidence"]),
        )
        result["passed"] = (
            result["cross_entropy_relative_increase_upper"]
            <= noninferiority["maximum_masked_cross_entropy_relative_increase_upper_bound"]
            and result["masked_accuracy_decrease_upper"]
            <= noninferiority["maximum_masked_accuracy_absolute_decrease_upper_bound"]
        )
        bootstrap[f"{candidate_mode}_vs_{reference_mode}"] = result

    efficiency_policy = gate["acceptance"]["efficiency"]
    summaries = {
        name: _throughput(receipts[name], int(efficiency_policy["warmup_updates_excluded"]))
        for name, _, _ in RUNS
    }
    for name, _, _ in RUNS:
        summaries[name]["maximum_memory_fraction"] = max(
            _finite(receipts[name]["peak_allocated_memory_mib"], "allocated memory"),
            _finite(receipts[name]["peak_reserved_memory_mib"], "reserved memory"),
            _finite(telemetry[name]["maximum_memory_mib"], "telemetry memory"),
        ) / _finite(telemetry[name]["total_memory_mib"], "total memory")
    seed_ratios: dict[str, dict[str, float | bool]] = {}
    for seed in (17, 42, 73):
        eager = summaries[f"seed{seed}_safe_eager"]["residue_tokens_per_second"]
        compiled = summaries[f"seed{seed}_fp32_compiled"]["residue_tokens_per_second"]
        bf16 = summaries[f"seed{seed}_bf16_selected"]["residue_tokens_per_second"]
        seed_ratios[str(seed)] = {
            "bf16_over_fp32_compiled": bf16 / compiled,
            "bf16_over_safe_eager": bf16 / eager,
            "passed": (
                bf16 / compiled >= efficiency_policy["minimum_bf16_relative_to_fp32_compiled_each_seed"]
                and bf16 / eager >= efficiency_policy["minimum_bf16_relative_to_safe_eager_each_seed"]
            ),
        }

    def pooled(mode: str) -> float:
        names = [name for name, _, run_mode in RUNS if run_mode == mode]
        return sum(summaries[name]["residue_tokens"] for name in names) / sum(
            summaries[name]["total_seconds"] for name in names
        )

    pooled_throughput = {mode: pooled(mode) for mode in MODES}
    pooled_ratios = {
        "bf16_over_fp32_compiled": pooled_throughput["bf16_selected"] / pooled_throughput["fp32_compiled"],
        "bf16_over_safe_eager": pooled_throughput["bf16_selected"] / pooled_throughput["safe_eager"],
    }
    pooled_efficiency_passed = (
        pooled_ratios["bf16_over_fp32_compiled"] >= efficiency_policy["minimum_bf16_relative_to_fp32_compiled_pooled"]
        and pooled_ratios["bf16_over_safe_eager"] >= efficiency_policy["minimum_bf16_relative_to_safe_eager_pooled"]
    )
    gates = {
        "artifact_integrity": artifact_audit["manifest_verified"] and not artifact_audit["writable_paths"],
        "bootstrap_noninferiority": all(value["passed"] for value in bootstrap.values()),
        "efficiency_each_seed": all(value["passed"] for value in seed_ratios.values()),
        "efficiency_pooled": pooled_efficiency_passed,
        "learning": all(value["passed"] for value in learning.values()),
        "membership": all(membership_gates.values()),
        "memory": all(summary["maximum_memory_fraction"] <= efficiency_policy["maximum_memory_fraction"] for summary in summaries.values()),
        "per_seed_noninferiority": all(value["passed"] for value in per_seed.values()),
        "short_horizon_numerical": all(value["passed"] for value in short_results.values()),
        "stability": all(summary["p95_over_p50"] <= efficiency_policy["maximum_p95_over_p50_step_time_each_run"] for summary in summaries.values()),
    }
    gate_passed = all(gates.values())
    payload = {
        "artifact_audit": artifact_audit,
        "authorization": {
            "exact_recovery": gate_passed,
            "pilot_16_gpu": False,
            "pilot_32_gpu": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "bootstrap_noninferiority": bootstrap,
        "claim_boundary": "Training-excluded dense-pilot MLM systems parity only; no production-corpus, biological, scaling, or state-of-the-art claim",
        "efficiency": {
            "pooled_ratios": pooled_ratios,
            "pooled_throughput": pooled_throughput,
            "seed_ratios": seed_ratios,
            "summaries": summaries,
        },
        "format_version": "antibody_ddp_scientific_parity_result_v1",
        "gate_passed": gate_passed,
        "gates": gates,
        "inputs": {
            "auditor_sha256": _sha256(Path(__file__)),
            "execution_sha256": _sha256(args.execution),
            "gate_sha256": _sha256(args.gate),
            "registration_sha256": _sha256(args.registration),
            "run_root": str(run_root),
        },
        "learning": learning,
        "membership_by_seed": membership_gates,
        "per_seed_noninferiority": per_seed,
        "runs": {
            name: {
                "receipt_sha256": _sha256(run_root / name / "run/receipt.json"),
                "run_seed": seed,
                "telemetry_sha256": telemetry[name]["file_sha256"],
                "trainer_state_sha256": _sha256(run_root / name / "run/trainer_state.json"),
            }
            for name, seed, _ in RUNS
        },
        "short_horizon_numerical": short_results,
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    _publish(args.output, payload)


if __name__ == "__main__":
    main()
