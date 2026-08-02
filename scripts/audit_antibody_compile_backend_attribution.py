#!/usr/bin/env python3
"""Audit a five-arm eight-GPU compiler-backend attribution."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import audit_antibody_compile_numerics_diagnostic as numerics
import audit_antibody_production_batch_efficiency as parent


FORMAT_VERSION = "antibody_compile_backend_attribution_result_v1"
PROTOCOL_FORMAT = "antibody_compile_backend_attribution_protocol_v1"
SIDECAR_FORMAT = "antibody_compile_backend_attribution_receipt_v1"
RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("dynamo_eager", "dynamo_eager_gc_off"),
    ("aot_eager", "aot_eager_gc_off"),
    ("inductor_no_cudagraphs", "inductor_no_cudagraphs_gc_off"),
    ("safe_eager_r2", "safe_eager"),
)
REFERENCE_RUNS = ("safe_eager_r1", "safe_eager_r2")
ATTRIBUTION_RUNS = ("dynamo_eager", "aot_eager", "inductor_no_cudagraphs")
PROMOTABLE_RUNS = ("inductor_no_cudagraphs",)
PARAMETER_COUNT = 634_513_920
HARDWARE = "NVIDIA GeForce RTX 5090"
PEAK_DENSE_BF16_TFLOPS = 209.5


class BackendAttributionAuditError(RuntimeError):
    """Report invalid backend-attribution evidence or provenance."""


def _self_hashed(path: Path) -> dict[str, Any]:
    value, _ = parent._load(path)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(parent._canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise BackendAttributionAuditError(f"Receipt self-hash differs: {path}")
    return value


def _expected_optimization(flags: Mapping[str, Any]) -> dict[str, Any]:
    compiled = bool(flags["compile_blocks"])
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
        "disable_compiled_cudagraphs": flags[
            "disable_compiled_cudagraphs"
        ],
        "disable_ddp_gradient_as_bucket_view": False,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": False,
        "length_bucketing": False,
        "pinned_staging": False,
    }


def _sidecar(
    mode_root: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    mode: str,
) -> dict[str, Any]:
    path = mode_root / "run/compile_backend_attribution_receipt.json"
    value = _self_hashed(path)
    flags = protocol["modes"][mode]
    python_gc = value.get("python_gc", {})
    if (
        value.get("format_version") != SIDECAR_FORMAT
        or value.get("mode") != mode
        or value.get("mode_flags") != flags
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("wrapper_sha256")
        != protocol["inputs"]["wrapper_sha256"]
        or value.get("authorization") != protocol["authorization"]
        or value.get("test_access") != protocol["test_access"]
        or value.get("base_receipt_file_sha256")
        != parent._sha256(mode_root / "run/receipt.json")
        or value.get("base_trainer_state_file_sha256")
        != parent._sha256(mode_root / "run/trainer_state.json")
        or python_gc.get("enabled_before") is not True
        or python_gc.get("enabled_after_restore") is not True
        or python_gc.get("enabled_during_run")
        is not (not flags["disable_python_gc"])
    ):
        raise BackendAttributionAuditError(f"Sidecar differs: {mode_root.name}")
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
        raise BackendAttributionAuditError(
            f"Python-GC statistics differ: {mode_root.name}"
        )
    return {
        "automatic_gc_disabled_during_run": flags["disable_python_gc"],
        "collection_counters_include_explicit_calls": True,
        "compile_backend": flags["compile_backend"],
        "file_sha256": parent._sha256(path),
        "observed_collection_counter_delta": [
            item["collections"] for item in delta
        ],
        "receipt_sha256": value["receipt_sha256"],
    }


def _pooled_reference(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    tokens = sum(
        int(summaries[name]["nonpadding_residue_tokens"])
        for name in REFERENCE_RUNS
    )
    seconds = sum(
        float(summaries[name]["measured_seconds"])
        for name in REFERENCE_RUNS
    )
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


def _evaluation_aggregate(receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = receipt["posttraining_evaluation"]
    return {
        "final_model_sha256": receipt["final_model_sha256"],
        "masked_accuracy": parent._finite(
            value["masked_accuracy"], "masked accuracy"
        ),
        "masked_cross_entropy": parent._finite(
            value["masked_cross_entropy"], "masked cross entropy"
        ),
        "selected_targets": int(value["selected_targets"]),
    }


def audit_backend_attribution(
    *,
    run_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    """Validate immutable backend evidence and apply frozen gates."""
    protocol, protocol_raw = parent._load(protocol_path, canonical=False)
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("measurement", {}).get("run_order")
        != [name for name, _ in RUNS]
        or protocol.get("measurement", {}).get("promotable_runs")
        != list(PROMOTABLE_RUNS)
    ):
        raise BackendAttributionAuditError("Protocol semantics differ")
    if parent._sha256(run_root / "protocol.json") != protocol_sha256:
        raise BackendAttributionAuditError("Run-local protocol differs")
    if parent._sha256(Path(__file__)) != protocol["inputs"]["auditor_sha256"]:
        raise BackendAttributionAuditError("Auditor source binding differs")
    artifact_audit = parent._verify_artifacts(run_root)
    expected_updates = int(protocol["batch"]["updates_per_run"])
    expected_seed = int(protocol["measurement"]["run_seed"])
    receipts: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    sidecars: dict[str, dict[str, Any]] = {}
    allocation_identity = None

    for name, mode in RUNS:
        mode_root = run_root / name
        receipt, _ = parent._load(mode_root / "run/receipt.json")
        state, _ = parent._load(mode_root / "run/trainer_state.json")
        flags = protocol["modes"][mode]
        expected_optimization = _expected_optimization(flags)
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
            raise BackendAttributionAuditError(f"Receipt differs: {name}")
        expected_inputs = {
            "ddp_communication_protocol_sha256": protocol["inputs"][
                "communication_protocol_sha256"
            ],
            "efficiency_protocol_sha256": protocol["inputs"][
                "efficiency_protocol_sha256"
            ],
            "model_family_sha256": protocol["inputs"]["model_family_sha256"],
            "parity_protocol_sha256": protocol["inputs"][
                "parity_protocol_sha256"
            ],
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
            raise BackendAttributionAuditError(f"Input differs: {name}")
        identity = tuple(
            receipt.get("allocation", {}).get(field)
            for field in ("hostname", "slurm_job_id", "slurm_job_nodelist")
        )
        if allocation_identity is None:
            allocation_identity = identity
        elif identity != allocation_identity:
            raise BackendAttributionAuditError("Runs did not share one allocation")
        parent._validate_trace(receipt, expected_updates)
        parent._rank_state_digest(receipt, "initial_model")
        parent._rank_state_digest(receipt, "final_model")
        parent._rank_state_digest(receipt, "final_optimizer")
        receipts[name] = receipt
        evaluations[name] = parent._evaluation(receipt)
        telemetry[name] = parent._telemetry(mode_root / "gpu.csv")
        sidecars[name] = _sidecar(
            mode_root,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            mode=mode,
        )

    if len({tuple(value["gpu_uuids"]) for value in telemetry.values()}) != 1:
        raise BackendAttributionAuditError("GPU identity changed")
    if len(
        {
            parent._rank_state_digest(receipt, "initial_model")
            for receipt in receipts.values()
        }
    ) != 1:
        raise BackendAttributionAuditError("Initial model identity differs")
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
                receipts[name]["peak_allocated_memory_mib"],
                "allocated memory",
            ),
            parent._finite(
                receipts[name]["peak_reserved_memory_mib"],
                "reserved memory",
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
        and parent._rank_state_digest(
            receipts[REFERENCE_RUNS[0]], "final_optimizer"
        )
        == parent._rank_state_digest(
            receipts[REFERENCE_RUNS[1]], "final_optimizer"
        )
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
    for candidate_index, candidate_name in enumerate(ATTRIBUTION_RUNS):
        short_horizon = {}
        noninferiority = {}
        for reference_index, reference_name in enumerate(REFERENCE_RUNS):
            short_horizon[reference_name] = numerics._short_horizon(
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
        promotable = candidate_name in PROMOTABLE_RUNS
        comparisons[candidate_name] = {
            "backend_layer": protocol["modes"][
                dict(RUNS)[candidate_name]
            ]["compile_backend"],
            "eligible_for_strict_confirmation": promotable
            and all(gates.values()),
            "gates": gates,
            "noninferiority": noninferiority,
            "promotable_backend": promotable,
            "short_horizon": short_horizon,
            "summary": summaries[candidate_name],
            "throughput_relative_to_eager": throughput_ratio,
        }

    eligible = [
        name
        for name in PROMOTABLE_RUNS
        if comparisons[name]["eligible_for_strict_confirmation"]
    ]
    selected = max(
        eligible,
        key=lambda name: summaries[name]["residue_tokens_per_second"],
        default=None,
    )
    diagnostic_gates = {
        "artifact_integrity": artifact_audit["manifest_verified"]
        and not artifact_audit["writable_paths"],
        "membership": membership_equal,
        "reference_stability": reference_stable,
        "sidecars": len(sidecars) == len(RUNS),
    }
    return {
        "artifact_audit": artifact_audit,
        "authorization": protocol["authorization"],
        "candidate_found": selected is not None,
        "claim_boundary": protocol["claim_boundary"],
        "comparisons": comparisons,
        "diagnostic_gates": diagnostic_gates,
        "diagnostic_valid": all(diagnostic_gates.values()),
        "evaluation_aggregates": {
            name: _evaluation_aggregate(receipt)
            for name, receipt in receipts.items()
        },
        "format_version": FORMAT_VERSION,
        "inputs": {
            "auditor_sha256": parent._sha256(Path(__file__)),
            "protocol_file_sha256": protocol_sha256,
            "run_root": str(run_root.resolve()),
        },
        "membership_equal": membership_equal,
        "mode_sidecars": sidecars,
        "pooled_reference": pooled_reference,
        "reference": {
            "evaluation_equal": reference_evaluation_equal,
            "final_state_equal": reference_state_equal,
            "throughput_relative_range": reference_range,
        },
        "selected_candidate_for_strict_confirmation": selected,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_backend_attribution(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
    )
    _publish(arguments.output, result)
    return 0 if result["diagnostic_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
