#!/usr/bin/env python3
"""Audit the eight-GPU compile numerics and Python-GC diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import audit_antibody_production_batch_efficiency as parent


FORMAT_VERSION = "antibody_compile_numerics_diagnostic_result_v1"
PROTOCOL_FORMAT = "antibody_compile_numerics_diagnostic_protocol_v1"
SIDECAR_FORMAT = "antibody_compile_numerics_diagnostic_receipt_v1"
RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("compile_default", "compile_default"),
    ("compile_gc_off", "compile_gc_off"),
    ("compile_accuracy", "compile_accuracy"),
    ("compile_accuracy_gc_off", "compile_accuracy_gc_off"),
    ("safe_eager_r2", "safe_eager"),
)
REFERENCE_RUNS = ("safe_eager_r1", "safe_eager_r2")
CANDIDATE_RUNS = tuple(name for name, mode in RUNS if mode != "safe_eager")
PARAMETER_COUNT = 634_513_920
HARDWARE = "NVIDIA GeForce RTX 5090"
PEAK_DENSE_BF16_TFLOPS = 209.5


class CompileNumericsAuditError(RuntimeError):
    """Report invalid diagnostic evidence or provenance."""


def _expected_optimization(compiled: bool) -> dict[str, Any]:
    return {
        "audit_gradient_rank_parity": True,
        "bucketed_padding": False,
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
        "fused_adamw": False,
        "length_bucketing": False,
        "pinned_staging": False,
    }


def _self_hashed(path: Path) -> dict[str, Any]:
    value, _ = parent._load(path)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != hashlib.sha256(parent._canonical_json_bytes(unsigned)).hexdigest():
        raise CompileNumericsAuditError(f"Receipt self-hash differs: {path}")
    return value


def _diagnostic_sidecar(
    mode_root: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    mode: str,
) -> dict[str, Any]:
    path = mode_root / "run/compile_numerics_diagnostic_receipt.json"
    value = _self_hashed(path)
    flags = protocol["modes"][mode]
    reduction = value.get("bf16_reduced_precision_reduction", {})
    python_gc = value.get("python_gc", {})
    if (
        value.get("format_version") != SIDECAR_FORMAT
        or value.get("mode") != mode
        or value.get("mode_flags") != flags
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("wrapper_sha256")
        != protocol["inputs"]["diagnostic_wrapper_sha256"]
        or value.get("authorization") != protocol["authorization"]
        or value.get("test_access") != protocol["test_access"]
        or value.get("base_receipt_file_sha256")
        != parent._sha256(mode_root / "run/receipt.json")
        or value.get("base_trainer_state_file_sha256")
        != parent._sha256(mode_root / "run/trainer_state.json")
        or value.get("parity_extension_receipt_file_sha256")
        != parent._sha256(mode_root / "run/parity_extension_receipt.json")
        or reduction
        != {
            "after_restore": True,
            "before": True,
            "during_run": not flags["accuracy_first_bf16_reduction"],
        }
        or python_gc.get("enabled_before") is not True
        or python_gc.get("enabled_after_restore") is not True
        or python_gc.get("enabled_during_run")
        is not (not flags["disable_python_gc"])
    ):
        raise CompileNumericsAuditError(f"Diagnostic sidecar differs: {mode}")
    before = python_gc.get("stats_before")
    after = python_gc.get("stats_after")
    delta = python_gc.get("stats_delta")
    if (
        not isinstance(before, list)
        or not isinstance(after, list)
        or not isinstance(delta, list)
        or len(before) != len(after)
        or len(before) != len(delta)
        or any(
            right[key] - left[key] != difference[key]
            for left, right, difference in zip(before, after, delta)
            for key in ("collected", "collections", "uncollectable")
        )
    ):
        raise CompileNumericsAuditError(f"Python-GC statistics differ: {mode}")
    if flags["disable_python_gc"] and any(
        item["collections"] != 0 for item in delta
    ):
        raise CompileNumericsAuditError(
            f"Python-GC collected while disabled: {mode}"
        )
    return {
        "file_sha256": parent._sha256(path),
        "python_gc": python_gc,
        "receipt_sha256": value["receipt_sha256"],
    }


def _parity_sidecar(
    mode_root: Path, protocol: Mapping[str, Any]
) -> dict[str, str]:
    path = mode_root / "run/parity_extension_receipt.json"
    value = _self_hashed(path)
    if (
        value.get("format_version")
        != "antibody_pretraining_parity_extension_receipt_v1"
        or value.get("extension_protocol_sha256")
        != protocol["inputs"]["extension_protocol_sha256"]
        or value.get("base_runner_sha256")
        != protocol["inputs"]["base_runner_sha256"]
        or value.get("wrapper_sha256")
        != protocol["inputs"]["parity_wrapper_sha256"]
        or value.get("test_access") != protocol["test_access"]
        or value.get("base_receipt_file_sha256")
        != parent._sha256(mode_root / "run/receipt.json")
        or value.get("base_trainer_state_file_sha256")
        != parent._sha256(mode_root / "run/trainer_state.json")
    ):
        raise CompileNumericsAuditError("Parity extension sidecar differs")
    return {
        "file_sha256": parent._sha256(path),
        "receipt_sha256": value["receipt_sha256"],
    }


def _short_horizon(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    limit: int,
    loss_limit: float,
    gradient_limit: float,
) -> dict[str, Any]:
    left = reference["trace"][:limit]
    right = candidate["trace"][:limit]
    maximum_loss = max(
        abs(parent._finite(a["loss"], "loss") - parent._finite(b["loss"], "loss"))
        for a, b in zip(left, right)
    )
    maximum_gradient = max(
        abs(
            parent._finite(a["gradient_norm"], "gradient norm")
            - parent._finite(b["gradient_norm"], "gradient norm")
        )
        / max(abs(parent._finite(a["gradient_norm"], "gradient norm")), 1.0e-12)
        for a, b in zip(left, right)
    )
    return {
        "maximum_gradient_norm_relative_difference": maximum_gradient,
        "maximum_loss_absolute_difference": maximum_loss,
        "passed": maximum_gradient <= gradient_limit and maximum_loss <= loss_limit,
    }


def _pooled_reference(
    summaries: Mapping[str, Mapping[str, Any]]
) -> dict[str, float]:
    tokens = sum(
        int(summaries[name]["nonpadding_residue_tokens"])
        for name in REFERENCE_RUNS
    )
    seconds = sum(float(summaries[name]["measured_seconds"]) for name in REFERENCE_RUNS)
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


def _slow_step_overlap(
    left: Mapping[str, Any], right: Mapping[str, Any], warmup: int
) -> dict[str, float]:
    left_times = np.asarray(
        [parent._finite(item["seconds"], "step seconds") for item in left["trace"][warmup:]],
        dtype=np.float64,
    )
    right_times = np.asarray(
        [parent._finite(item["seconds"], "step seconds") for item in right["trace"][warmup:]],
        dtype=np.float64,
    )
    if len(left_times) != len(right_times) or len(left_times) < 10:
        raise CompileNumericsAuditError("Slow-step vectors differ")
    count = max(1, math.ceil(0.10 * len(left_times)))
    left_slow = set(np.argsort(left_times)[-count:].tolist())
    right_slow = set(np.argsort(right_times)[-count:].tolist())
    return {
        "pearson_step_time": float(np.corrcoef(left_times, right_times)[0, 1]),
        "slowest_10_percent_jaccard": len(left_slow & right_slow)
        / len(left_slow | right_slow),
    }


def audit_compile_numerics(
    *, run_root: Path, protocol_path: Path
) -> dict[str, Any]:
    protocol, protocol_raw = parent._load(protocol_path, canonical=False)
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("measurement", {}).get("run_order")
        != [name for name, _ in RUNS]
    ):
        raise CompileNumericsAuditError("Diagnostic protocol differs")
    if parent._sha256(run_root / "protocol.json") != protocol_sha256:
        raise CompileNumericsAuditError("Run-local protocol differs")
    if parent._sha256(Path(__file__)) != protocol["inputs"]["auditor_sha256"]:
        raise CompileNumericsAuditError("Diagnostic auditor source differs")
    artifact_audit = parent._verify_artifacts(run_root)
    expected_updates = int(protocol["batch"]["updates_per_run"])
    expected_seed = int(protocol["measurement"]["run_seed"])
    receipts: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    diagnostic_sidecars: dict[str, dict[str, Any]] = {}
    parity_sidecars: dict[str, dict[str, str]] = {}
    allocation_identity = None

    for name, mode in RUNS:
        mode_root = run_root / name
        receipt, _ = parent._load(mode_root / "run/receipt.json")
        state, _ = parent._load(mode_root / "run/trainer_state.json")
        compiled = bool(protocol["modes"][mode]["compile_blocks"])
        expected_optimization = _expected_optimization(compiled)
        if (
            receipt.get("format_version")
            != "antibody_pretraining_optimized_pilot_v1"
            or receipt.get("backend") != "ddp"
            or receipt.get("world_size") != 8
            or receipt.get("device") != HARDWARE
            or receipt.get("run_seed") != expected_seed
            or receipt.get("preset") != "abf_650m"
            or receipt.get("batch_quantum") != 512
            or receipt.get("micro_batch_per_gpu") != 16
            or receipt.get("target_residue_tokens") != 262_144
            or receipt.get("parameter_moments", {}).get("count")
            != PARAMETER_COUNT
            or receipt.get("optimization") != expected_optimization
            or receipt.get("checkpoint_format") is not None
            or receipt.get("resume_used") is not False
            or receipt.get("authorization")
            != {"production_pretraining": False, "scientific_claim": False}
            or receipt.get("test_access") != protocol["test_access"]
            or state.get("next_update") != expected_updates
            or state.get("training_universe_count") != 7300
            or state.get("optimization") != expected_optimization
        ):
            raise CompileNumericsAuditError(f"Receipt or state differs: {name}")
        expected_inputs = {
            "ddp_communication_protocol_sha256": protocol["inputs"][
                "communication_protocol_sha256"
            ],
            "efficiency_protocol_sha256": protocol["inputs"][
                "efficiency_protocol_sha256"
            ],
            "model_family_sha256": protocol["inputs"]["model_family_sha256"],
            "parity_protocol_sha256": protocol["inputs"]["parity_protocol_sha256"],
            "parity_split_receipt_file_sha256": protocol["inputs"][
                "parity_split_receipt_file_sha256"
            ],
            "shard_manifest_sha256": protocol["inputs"][
                "dense_manifest_payload_sha256"
            ],
            "training_protocol_sha256": protocol["inputs"][
                "training_protocol_sha256"
            ],
        }
        if any(receipt.get(key) != value for key, value in expected_inputs.items()):
            raise CompileNumericsAuditError(f"Input provenance differs: {name}")
        identity = tuple(
            receipt.get("allocation", {}).get(field)
            for field in ("hostname", "slurm_job_id", "slurm_job_nodelist")
        )
        if allocation_identity is None:
            allocation_identity = identity
        elif identity != allocation_identity:
            raise CompileNumericsAuditError("Runs did not share one allocation")
        parent._validate_trace(receipt, expected_updates)
        parent._rank_state_digest(receipt, "initial_model")
        parent._rank_state_digest(receipt, "final_model")
        parent._rank_state_digest(receipt, "final_optimizer")
        receipts[name] = receipt
        evaluations[name] = parent._evaluation(receipt)
        telemetry[name] = parent._telemetry(mode_root / "gpu.csv")
        diagnostic_sidecars[name] = _diagnostic_sidecar(
            mode_root,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            mode=mode,
        )
        parity_sidecars[name] = _parity_sidecar(mode_root, protocol)

    if len({tuple(value["gpu_uuids"]) for value in telemetry.values()}) != 1:
        raise CompileNumericsAuditError("GPU identity changed during diagnostic")
    if len(
        {
            parent._rank_state_digest(receipt, "initial_model")
            for receipt in receipts.values()
        }
    ) != 1:
        raise CompileNumericsAuditError("Initial model identity differs")

    membership_fields = (
        "consumed_residue_tokens",
        "learning_rate",
        "micro_steps",
        "residue_tokens",
        "selected_targets",
        "sequence_count",
        "update",
    )
    first = receipts[RUNS[0][0]]
    membership_equal = len(
        {value["membership_sha256"] for value in receipts.values()}
    ) == 1 and all(
        all(
            all(left[field] == right[field] for field in membership_fields)
            for left, right in zip(first["trace"], receipt["trace"])
        )
        for receipt in receipts.values()
    )

    acceptance = protocol["acceptance"]
    warmup = int(acceptance["warmup_updates_excluded"])
    summaries = {
        name: parent._throughput(receipt, warmup)
        for name, receipt in receipts.items()
    }
    for name in summaries:
        summaries[name]["maximum_memory_fraction"] = max(
            parent._finite(
                receipts[name]["peak_allocated_memory_mib"], "allocated memory"
            ),
            parent._finite(
                receipts[name]["peak_reserved_memory_mib"], "reserved memory"
            ),
            telemetry[name]["maximum_memory_mib"],
        ) / telemetry[name]["total_memory_mib"]

    reference_throughputs = [
        summaries[name]["residue_tokens_per_second"] for name in REFERENCE_RUNS
    ]
    reference_range = (
        max(reference_throughputs) - min(reference_throughputs)
    ) / min(reference_throughputs)
    reference_state_equal = (
        parent._rank_state_digest(receipts[REFERENCE_RUNS[0]], "final_model")
        == parent._rank_state_digest(receipts[REFERENCE_RUNS[1]], "final_model")
        and parent._rank_state_digest(receipts[REFERENCE_RUNS[0]], "final_optimizer")
        == parent._rank_state_digest(receipts[REFERENCE_RUNS[1]], "final_optimizer")
    )
    reference_evaluation_equal = all(
        np.array_equal(left, right)
        for left, right in zip(
            evaluations[REFERENCE_RUNS[0]], evaluations[REFERENCE_RUNS[1]]
        )
    )
    reference_stable = (
        reference_range
        <= acceptance["maximum_reference_throughput_relative_range"]
        and reference_state_equal
        and reference_evaluation_equal
    )
    pooled_reference = _pooled_reference(summaries)

    comparisons: dict[str, Any] = {}
    for candidate_index, candidate_name in enumerate(CANDIDATE_RUNS):
        short_horizon = {}
        noninferiority = {}
        for reference_index, reference_name in enumerate(REFERENCE_RUNS):
            short_horizon[reference_name] = _short_horizon(
                receipts[reference_name],
                receipts[candidate_name],
                limit=int(acceptance["short_horizon_updates"]),
                loss_limit=float(
                    acceptance["maximum_short_horizon_loss_absolute_difference"]
                ),
                gradient_limit=float(
                    acceptance[
                        "maximum_short_horizon_gradient_norm_relative_difference"
                    ]
                ),
            )
            result = parent._bootstrap_noninferiority(
                evaluations[reference_name],
                evaluations[candidate_name],
                iterations=int(acceptance["bootstrap_iterations"]),
                confidence=float(acceptance["bootstrap_confidence"]),
                seed=int(acceptance["bootstrap_seed"])
                + candidate_index * 2
                + reference_index,
            )
            result["passed"] = (
                result["cross_entropy_relative_increase_upper"]
                <= acceptance["maximum_masked_cross_entropy_relative_increase"]
                and result["masked_accuracy_decrease_upper"]
                <= acceptance["maximum_masked_accuracy_absolute_decrease"]
            )
            noninferiority[reference_name] = result
        throughput_ratio = (
            summaries[candidate_name]["residue_tokens_per_second"]
            / pooled_reference["residue_tokens_per_second"]
        )
        gates = {
            "evaluation_noninferiority": all(
                value["passed"] for value in noninferiority.values()
            ),
            "memory": summaries[candidate_name]["maximum_memory_fraction"]
            <= acceptance["maximum_memory_fraction"],
            "short_horizon_numerical": all(
                value["passed"] for value in short_horizon.values()
            ),
            "step_stability": summaries[candidate_name]["p95_over_p50"]
            <= acceptance["maximum_p95_over_p50_step_time"],
            "throughput": throughput_ratio
            >= acceptance["minimum_candidate_relative_throughput"],
        }
        comparisons[candidate_name] = {
            "eligible_for_strict_confirmation": all(gates.values()),
            "gates": gates,
            "noninferiority": noninferiority,
            "short_horizon": short_horizon,
            "summary": summaries[candidate_name],
            "throughput_relative_to_eager": throughput_ratio,
        }

    eligible = [
        name
        for name in CANDIDATE_RUNS
        if comparisons[name]["eligible_for_strict_confirmation"]
    ]
    selected_candidate = max(
        eligible,
        key=lambda name: summaries[name]["residue_tokens_per_second"],
        default=None,
    )
    diagnostic_gates = {
        "artifact_integrity": artifact_audit["manifest_verified"]
        and not artifact_audit["writable_paths"],
        "membership": membership_equal,
        "reference_stability": reference_stable,
        "sidecars": len(diagnostic_sidecars) == len(RUNS)
        and len(parity_sidecars) == len(RUNS),
    }
    return {
        "artifact_audit": artifact_audit,
        "authorization": protocol["authorization"],
        "candidate_found": selected_candidate is not None,
        "claim_boundary": protocol["claim_boundary"],
        "comparisons": comparisons,
        "diagnostic_gates": diagnostic_gates,
        "diagnostic_valid": all(diagnostic_gates.values()),
        "format_version": FORMAT_VERSION,
        "inputs": {
            "auditor_sha256": parent._sha256(Path(__file__)),
            "protocol_file_sha256": protocol_sha256,
            "run_root": str(run_root.resolve()),
        },
        "membership_equal": membership_equal,
        "mode_sidecars": diagnostic_sidecars,
        "parity_extension_sidecars": parity_sidecars,
        "pooled_reference": pooled_reference,
        "reference": {
            "evaluation_equal": reference_evaluation_equal,
            "final_state_equal": reference_state_equal,
            "throughput_relative_range": reference_range,
        },
        "runtime_stall_attribution": {
            "default_vs_gc_off": _slow_step_overlap(
                receipts["compile_default"], receipts["compile_gc_off"], warmup
            ),
            "accuracy_vs_accuracy_gc_off": _slow_step_overlap(
                receipts["compile_accuracy"],
                receipts["compile_accuracy_gc_off"],
                warmup,
            ),
        },
        "selected_candidate_for_strict_confirmation": selected_candidate,
        "summaries": summaries,
        "test_access": protocol["test_access"],
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(
        parent._canonical_json_bytes(payload)
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(parent._canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    result = audit_compile_numerics(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
    )
    _publish(arguments.output, result)
    return 0 if result["diagnostic_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
