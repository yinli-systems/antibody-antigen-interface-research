#!/usr/bin/env python3
"""Audit the source-bound eight-GPU efficiency factor-attribution experiment."""

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


FORMAT_VERSION = "antibody_efficiency_factor_attribution_result_v1"
RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("compile_only", "compile_only"),
    ("fused_adamw_only", "fused_adamw_only"),
    ("data_pipeline_only", "data_pipeline_only"),
    ("compile_data_pipeline", "compile_data_pipeline"),
    ("full_bundle", "full_bundle"),
    ("safe_eager_r2", "safe_eager"),
)
REFERENCE_RUNS = ("safe_eager_r1", "safe_eager_r2")
PARAMETER_COUNT = 634_513_920
HARDWARE = "NVIDIA GeForce RTX 5090"
PEAK_DENSE_BF16_TFLOPS = 209.5


class FactorAttributionAuditError(RuntimeError):
    """Report an invalid factor-attribution artifact or provenance binding."""


def _expected_optimization(flags: Mapping[str, bool]) -> dict[str, Any]:
    return {
        "audit_gradient_rank_parity": True,
        "bucketed_padding": flags["bucketed_padding"],
        "compile_blocks": flags["compile_blocks"],
        "compile_prewarm_buckets": flags["compile_prewarm_buckets"],
        "ddp_batched_grad_copy": False,
        "ddp_bucket_cap_mb": 25,
        "ddp_communication_dtype": "float32",
        "ddp_static_graph": False,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": False,
        "disable_ddp_gradient_as_bucket_view": False,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": flags["fused_adamw"],
        "length_bucketing": False,
        "pinned_staging": flags["pinned_staging"],
    }


def _extension_sidecar(
    mode_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    path = mode_root / "run/parity_extension_receipt.json"
    value, _ = parent._load(path)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    launch = value.get("launch_argv")
    if (
        claimed != hashlib.sha256(parent._canonical_json_bytes(unsigned)).hexdigest()
        or value.get("format_version")
        != "antibody_pretraining_parity_extension_receipt_v1"
        or value.get("authorization")
        != {
            "model_selection": False,
            "production_pretraining": False,
            "scientific_claim": False,
        }
        or value.get("test_access") != protocol["test_access"]
        or value.get("extension_protocol_sha256")
        != protocol["inputs"]["extension_protocol_sha256"]
        or value.get("base_runner_sha256") != protocol["inputs"]["runner_sha256"]
        or value.get("wrapper_sha256")
        != protocol["inputs"]["runner_wrapper_sha256"]
        or value.get("base_receipt_file_sha256")
        != parent._sha256(mode_root / "run/receipt.json")
        or value.get("base_trainer_state_file_sha256")
        != parent._sha256(mode_root / "run/trainer_state.json")
        or not isinstance(launch, list)
        or launch.count("--steps") != 1
        or launch[launch.index("--steps") + 1]
        != str(protocol["batch"]["updates_per_run"])
    ):
        raise FactorAttributionAuditError("Parity extension sidecar differs")
    return {
        "file_sha256": parent._sha256(path),
        "receipt_sha256": claimed,
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


def _pooled_reference(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    tokens = sum(int(summaries[name]["nonpadding_residue_tokens"]) for name in REFERENCE_RUNS)
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


def audit_factor_attribution(
    *,
    run_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol, protocol_raw = parent._load(protocol_path, canonical=False)
    if protocol.get("format_version") != "antibody_efficiency_factor_attribution_protocol_v1":
        raise FactorAttributionAuditError("Factor-attribution protocol format differs")
    if protocol["measurement"]["run_order"] != [name for name, _ in RUNS]:
        raise FactorAttributionAuditError("Factor-attribution run order differs")
    if parent._sha256(run_root / "protocol.json") != hashlib.sha256(protocol_raw).hexdigest():
        raise FactorAttributionAuditError("Run-local protocol differs")

    parent_auditor = Path(parent.__file__).resolve()
    if parent._sha256(parent_auditor) != protocol["inputs"]["parent_auditor_sha256"]:
        raise FactorAttributionAuditError("Parent audit implementation differs")
    artifact_audit = parent._verify_artifacts(run_root)
    expected_updates = int(protocol["batch"]["updates_per_run"])
    expected_seed = int(protocol["measurement"]["run_seed"])
    receipts: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    sidecars: dict[str, dict[str, Any]] = {}
    allocation_identity = None

    for name, mode in RUNS:
        receipt, _ = parent._load(run_root / name / "run/receipt.json")
        state, _ = parent._load(run_root / name / "run/trainer_state.json")
        expected_optimization = _expected_optimization(protocol["modes"][mode])
        if (
            receipt.get("format_version") != "antibody_pretraining_optimized_pilot_v1"
            or receipt.get("backend") != "ddp"
            or receipt.get("world_size") != 8
            or receipt.get("device") != HARDWARE
            or receipt.get("run_seed") != expected_seed
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
            or state.get("next_update") != expected_updates
            or state.get("training_universe_count") != 7300
            or state.get("optimization") != expected_optimization
        ):
            raise FactorAttributionAuditError(f"Receipt or state differs for {name}")
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
            raise FactorAttributionAuditError(f"Input provenance differs for {name}")
        identity = tuple(
            receipt.get("allocation", {}).get(field)
            for field in ("hostname", "slurm_job_id", "slurm_job_nodelist")
        )
        if allocation_identity is None:
            allocation_identity = identity
        elif identity != allocation_identity:
            raise FactorAttributionAuditError("Runs did not share one allocation")
        parent._validate_trace(receipt, expected_updates)
        parent._rank_state_digest(receipt, "initial_model")
        parent._rank_state_digest(receipt, "final_model")
        parent._rank_state_digest(receipt, "final_optimizer")
        receipts[name] = receipt
        evaluations[name] = parent._evaluation(receipt)
        telemetry[name] = parent._telemetry(run_root / name / "gpu.csv")
        sidecars[name] = _extension_sidecar(run_root / name, protocol)

    if len({tuple(value["gpu_uuids"]) for value in telemetry.values()}) != 1:
        raise FactorAttributionAuditError("GPU identity changed during attribution")
    initial_states = {
        parent._rank_state_digest(receipt, "initial_model") for receipt in receipts.values()
    }
    if len(initial_states) != 1:
        raise FactorAttributionAuditError("Initial model identity differs across runs")

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
    membership_equal = len({value["membership_sha256"] for value in receipts.values()}) == 1
    membership_equal = membership_equal and all(
        all(
            all(left[field] == right[field] for field in membership_fields)
            for left, right in zip(first["trace"], receipt["trace"])
        )
        for receipt in receipts.values()
    )

    acceptance = protocol["acceptance"]
    summaries = {
        name: parent._throughput(receipt, int(acceptance["warmup_updates_excluded"]))
        for name, receipt in receipts.items()
    }
    for name in summaries:
        summaries[name]["maximum_memory_fraction"] = max(
            parent._finite(receipts[name]["peak_allocated_memory_mib"], "allocated memory"),
            parent._finite(receipts[name]["peak_reserved_memory_mib"], "reserved memory"),
            telemetry[name]["maximum_memory_mib"],
        ) / telemetry[name]["total_memory_mib"]

    reference_throughputs = [summaries[name]["residue_tokens_per_second"] for name in REFERENCE_RUNS]
    reference_range = (max(reference_throughputs) - min(reference_throughputs)) / min(reference_throughputs)
    reference_state_equal = (
        parent._rank_state_digest(receipts[REFERENCE_RUNS[0]], "final_model")
        == parent._rank_state_digest(receipts[REFERENCE_RUNS[1]], "final_model")
        and parent._rank_state_digest(receipts[REFERENCE_RUNS[0]], "final_optimizer")
        == parent._rank_state_digest(receipts[REFERENCE_RUNS[1]], "final_optimizer")
    )
    reference_evaluation_equal = all(
        np.array_equal(left, right)
        for left, right in zip(evaluations[REFERENCE_RUNS[0]], evaluations[REFERENCE_RUNS[1]])
    )
    reference_stable = (
        reference_range <= acceptance["maximum_reference_throughput_relative_range"]
        and reference_state_equal
        and reference_evaluation_equal
    )
    pooled_reference = _pooled_reference(summaries)

    comparisons: dict[str, Any] = {}
    for candidate_index, candidate_name in enumerate(protocol["selection"]["eligible_modes"]):
        short_horizon = {}
        noninferiority = {}
        for reference_index, reference_name in enumerate(REFERENCE_RUNS):
            short_horizon[reference_name] = _short_horizon(
                receipts[reference_name],
                receipts[candidate_name],
                limit=int(acceptance["short_horizon_updates"]),
                loss_limit=float(acceptance["maximum_short_horizon_loss_absolute_difference"]),
                gradient_limit=float(acceptance["maximum_short_horizon_gradient_norm_relative_difference"]),
            )
            result = parent._bootstrap_noninferiority(
                evaluations[reference_name],
                evaluations[candidate_name],
                iterations=int(acceptance["bootstrap_iterations"]),
                confidence=float(acceptance["bootstrap_confidence"]),
                seed=int(acceptance["bootstrap_seed"]) + candidate_index * 2 + reference_index,
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
            "evaluation_noninferiority": all(value["passed"] for value in noninferiority.values()),
            "memory": summaries[candidate_name]["maximum_memory_fraction"] <= acceptance["maximum_memory_fraction"],
            "short_horizon_numerical": all(value["passed"] for value in short_horizon.values()),
            "step_stability": summaries[candidate_name]["p95_over_p50"] <= acceptance["maximum_p95_over_p50_step_time"],
            "throughput": throughput_ratio >= acceptance["minimum_candidate_relative_throughput"],
        }
        comparisons[candidate_name] = {
            "eligible": all(gates.values()),
            "gates": gates,
            "noninferiority": noninferiority,
            "short_horizon": short_horizon,
            "throughput_relative_to_pooled_eager": throughput_ratio,
        }

    selection_order = {name: index for index, name in enumerate(protocol["selection"]["eligible_modes"])}
    eligible = [name for name, value in comparisons.items() if value["eligible"]]
    selected_mode = None
    if eligible and membership_equal and reference_stable:
        selected_mode = min(
            eligible,
            key=lambda name: (
                -summaries[name]["residue_tokens_per_second"],
                selection_order[name],
            ),
        )
    global_gates = {
        "artifact_integrity": artifact_audit["manifest_verified"] and not artifact_audit["writable_paths"],
        "membership": membership_equal,
        "reference_stability": reference_stable,
    }
    attribution_gate_passed = all(global_gates.values()) and selected_mode is not None
    return {
        "artifact_audit": artifact_audit,
        "attribution_gate_passed": attribution_gate_passed,
        "authorization": protocol["authorization"],
        "claim_boundary": protocol["claim_boundary"],
        "comparisons": comparisons,
        "format_version": FORMAT_VERSION,
        "global_gates": global_gates,
        "inputs": {
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parent_auditor_sha256": parent._sha256(parent_auditor),
            "protocol_file_sha256": hashlib.sha256(protocol_raw).hexdigest(),
            "run_root": str(run_root.resolve()),
        },
        "membership_equal": membership_equal,
        "parity_extension_sidecars": sidecars,
        "pooled_reference": pooled_reference,
        "reference": {
            "evaluation_equal": reference_evaluation_equal,
            "final_state_equal": reference_state_equal,
            "throughput_relative_range": reference_range,
        },
        "required_successor": protocol["selection"]["required_successor"],
        "runs": {
            name: {
                "receipt_sha256": parent._sha256(run_root / name / "run/receipt.json"),
                "summary": summaries[name],
                "telemetry": telemetry[name],
            }
            for name, _ in RUNS
        },
        "selected_mode": selected_mode,
        "test_access": protocol["test_access"],
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(parent._canonical_json_bytes(payload)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(parent._canonical_json_bytes(payload))
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
    result = audit_factor_attribution(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
    )
    _publish(arguments.output, result)
    return 0 if result["attribution_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
