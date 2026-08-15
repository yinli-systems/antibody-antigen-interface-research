#!/usr/bin/env python3
"""Audit the non-authorizing four-GPU OAS shape-stability crossover."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_four_gpu_efficiency_gate as parent
import freeze_antibody_four_gpu_shape_stability_probe as freezer


FORMAT_VERSION = "antibody_four_gpu_shape_stability_result_v1"
RUNS = (
    ("bucketed_r1", "bucketed"),
    ("static_194_r1", "static_194"),
    ("static_194_r2", "static_194"),
    ("bucketed_r2", "bucketed"),
)


class ShapeProbeAuditError(RuntimeError):
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value, digest = parent._load(path)
    except parent.EfficiencyAuditError as error:
        raise ShapeProbeAuditError(str(error)) from error
    return value, digest


def _validate_master(protocol: Mapping[str, Any]) -> None:
    if (
        protocol.get("format_version") != freezer.FORMAT_VERSION
        or protocol.get("acceptance") != freezer.ACCEPTANCE
        or protocol.get("authorization") != freezer.AUTHORIZATION
        or protocol.get("batch") != freezer.BATCH
        or protocol.get("candidates") != freezer.CANDIDATES
        or protocol.get("hardware") != freezer.HARDWARE
        or protocol.get("run_order") != list(freezer.RUN_ORDER)
        or protocol.get("test_access") != freezer.TEST_ACCESS
        or set(protocol.get("source", {})) != set(freezer.SOURCE_FILES)
    ):
        raise ShapeProbeAuditError("Master shape-probe protocol differs")


def _expected_optimization() -> dict[str, Any]:
    return {
        "bucketed_padding": True,
        "compile_blocks": True,
        "compile_prewarm_buckets": True,
        "ddp_bucket_cap_mb": 25,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": True,
        "fused_adamw": True,
        "length_bucketing": True,
        "pinned_staging": True,
    }


def _validate_gate(gate: Mapping[str, Any], candidate: str) -> None:
    if (
        candidate not in freezer.CANDIDATES
        or gate.get("format_version") != "antibody_production_loader_recovery_gate_v1"
        or gate.get("hardware_identity") != freezer.HARDWARE
        or gate.get("authorization") != parent.RUN_AUTHORIZATION
        or gate.get("test_access") != parent.RUN_TEST_ACCESS
        or gate.get("execution", {}).get("world_size") != 4
        or gate.get("execution", {}).get("backend") != "ddp"
        or gate.get("execution", {}).get("optimization") != _expected_optimization()
        or gate.get("execution", {}).get("length_bucket_boundaries")
        != freezer.CANDIDATES[candidate]
        or gate.get("runs")
        != {
            "continuous": {
                "batch_quantum_sequences": 256,
                "checkpoint": False,
                "micro_batch_per_gpu": 16,
                "preset": "abf_650m",
                "resume": False,
                "run_seed": 17,
                "steps": 300,
                "target_residue_tokens": 131072,
            }
        }
    ):
        raise ShapeProbeAuditError(f"Candidate gate identity differs: {candidate}")


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ShapeProbeAuditError(f"{name} is not finite")
    return result


def _short_horizon(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    limit: int,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    left_trace = left["trace"][:limit]
    right_trace = right["trace"][:limit]
    maximum_loss = max(
        abs(_finite(a["loss"], "loss") - _finite(b["loss"], "loss"))
        for a, b in zip(left_trace, right_trace)
    )
    maximum_gradient = max(
        abs(
            _finite(a["gradient_norm"], "gradient norm")
            - _finite(b["gradient_norm"], "gradient norm")
        )
        / max(abs(_finite(a["gradient_norm"], "gradient norm")), 1.0e-12)
        for a, b in zip(left_trace, right_trace)
    )
    return {
        "maximum_gradient_norm_relative_difference": maximum_gradient,
        "maximum_loss_absolute_difference": maximum_loss,
        "passed": (
            maximum_gradient
            <= acceptance["maximum_short_horizon_gradient_norm_relative_difference"]
            and maximum_loss
            <= acceptance["maximum_short_horizon_loss_absolute_difference"]
        ),
        "updates": limit,
    }


def _configuration_summary(
    names: list[str],
    summaries: Mapping[str, Mapping[str, Any]],
    receipts: Mapping[str, Mapping[str, Any]],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    tokens = sum(int(summaries[name]["nonpadding_residue_tokens"]) for name in names)
    seconds = sum(float(summaries[name]["measured_seconds"]) for name in names)
    rate = tokens / seconds
    replicate_rates = [float(summaries[name]["residue_tokens_per_second"]) for name in names]
    replicate_range = (max(replicate_rates) - min(replicate_rates)) / min(replicate_rates)
    replicate_state_equal = all(
        receipts[names[0]][field] == receipts[name][field]
        for name in names[1:]
        for field in ("final_model_sha256", "final_optimizer_sha256")
    )
    pooled_mfu = (
        6.0
        * parent.PARAMETER_COUNT
        * rate
        / (4 * parent.PEAK_DENSE_BF16_TFLOPS * 1.0e12)
    )
    gates = {
        "approximate_mfu": pooled_mfu >= acceptance["minimum_approximate_6n_mfu"],
        "memory": all(
            summaries[name]["maximum_memory_fraction"]
            <= acceptance["maximum_memory_fraction"]
            for name in names
        ),
        "replicate_stability": replicate_range
        <= acceptance["maximum_replicate_throughput_relative_range"],
        "replicate_state": replicate_state_equal,
        "step_stability": all(
            summaries[name]["p95_over_p50"]
            <= acceptance["maximum_p95_over_p50_step_time"]
            for name in names
        ),
    }
    return {
        "eligible": all(gates.values()),
        "gates": gates,
        "pooled": {
            "approximate_6n_mfu": pooled_mfu,
            "nonpadding_residue_tokens": tokens,
            "residue_tokens_per_second": rate,
            "seconds": seconds,
        },
        "replicate_state_equal": replicate_state_equal,
        "replicate_throughput_relative_range": replicate_range,
        "runs": names,
    }


def audit_probe(
    *,
    run_root: Path,
    protocol_path: Path,
    bundle_root: Path,
) -> dict[str, Any]:
    protocol, protocol_sha256 = _load(protocol_path)
    _validate_master(protocol)
    source_root = Path(__file__).resolve().parents[1]
    for relative, expected in protocol["source"].items():
        if _sha256(source_root / relative) != expected:
            raise ShapeProbeAuditError(f"Source closure differs: {relative}")
    candidate_gates: dict[str, tuple[dict[str, Any], str]] = {}
    for candidate in freezer.CANDIDATES:
        gate, digest = _load(bundle_root / f"{candidate}_gate.json")
        if digest != protocol["mode_gate_file_sha256"][candidate]:
            raise ShapeProbeAuditError(f"Candidate gate hash differs: {candidate}")
        _validate_gate(gate, candidate)
        candidate_gates[candidate] = (gate, digest)

    artifact_audit = parent.artifacts._verify_artifacts(run_root)
    receipts: dict[str, dict[str, Any]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    allocation = None
    updates = int(protocol["batch"]["updates_per_run"])
    for label, candidate in RUNS:
        gate, gate_sha256 = candidate_gates[candidate]
        receipt, _ = _load(run_root / label / "run" / "receipt.json")
        state, _ = _load(run_root / label / "run" / "trainer_state.json")
        optimization = _expected_optimization()
        if (
            receipt.get("format_version") != parent.RUN_FORMAT
            or receipt.get("run_role") != "continuous"
            or receipt.get("world_size") != 4
            or receipt.get("device") != parent.GPU_MODEL
            or receipt.get("authorization") != parent.RUN_AUTHORIZATION
            or receipt.get("test_access") != parent.RUN_TEST_ACCESS
            or receipt.get("gate_protocol_sha256") != gate_sha256
            or receipt.get("implementation") != gate.get("implementation")
            or receipt.get("optimization") != optimization
            or receipt.get("resume_used") is not False
            or receipt.get("checkpoint_model_sha256") is not None
            or receipt.get("checkpoint_optimizer_sha256") is not None
            or receipt.get("checkpoint_rng_manifest_sha256") is not None
            or state.get("next_update") != updates
            or state.get("world_size") != 4
            or state.get("optimization") != optimization
            or state.get("length_bucket_boundaries")
            != [int(item) for item in freezer.CANDIDATES[candidate].split(",")]
        ):
            raise ShapeProbeAuditError(f"Run identity differs: {label}")
        identity = tuple(
            receipt.get("allocation", {}).get(key)
            for key in ("hostname", "slurm_job_id", "slurm_job_nodelist")
        )
        if allocation is None:
            allocation = identity
        elif identity != allocation:
            raise ShapeProbeAuditError("Runs did not share one allocation")
        try:
            parent._validate_trace(receipt, updates)
            telemetry[label] = parent._telemetry(run_root / label / "gpu.csv")
        except parent.EfficiencyAuditError as error:
            raise ShapeProbeAuditError(str(error)) from error
        receipts[label] = receipt

    gpu_identity_equal = len(
        {tuple(value["gpu_uuids"]) for value in telemetry.values()}
    ) == 1
    initial_model_equal = len(
        {value["initial_model_sha256"] for value in receipts.values()}
    ) == 1
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
            all(a[key] == b[key] for key in membership_fields)
            for a, b in zip(reference["trace"], receipt["trace"])
        )
        for receipt in receipts.values()
    )
    acceptance = protocol["acceptance"]
    summaries = {
        label: parent._throughput(
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
    numerical = {
        "r1": _short_horizon(
            receipts["bucketed_r1"],
            receipts["static_194_r1"],
            limit=int(acceptance["short_horizon_updates"]),
            acceptance=acceptance,
        ),
        "r2": _short_horizon(
            receipts["bucketed_r2"],
            receipts["static_194_r2"],
            limit=int(acceptance["short_horizon_updates"]),
            acceptance=acceptance,
        ),
    }
    configurations = {
        candidate: _configuration_summary(
            [label for label, item in RUNS if item == candidate],
            summaries,
            receipts,
            acceptance,
        )
        for candidate in freezer.CANDIDATES
    }
    global_gates = {
        "artifact_integrity": artifact_audit["manifest_verified"]
        and not artifact_audit["writable_paths"],
        "gpu_identity": gpu_identity_equal,
        "initial_model": initial_model_equal,
        "membership": membership_equal,
        "short_horizon_numerical": all(item["passed"] for item in numerical.values()),
    }
    passed = all(global_gates.values()) and configurations["static_194"]["eligible"]
    return {
        "artifact_audit": artifact_audit,
        "authorization": {
            "configuration_promotion": False,
            "full_efficiency_gate_candidate": passed,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "configurations": configurations,
        "format_version": FORMAT_VERSION,
        "global_gates": global_gates,
        "inputs": {
            "auditor_sha256": _sha256(Path(__file__)),
            "protocol_file_sha256": protocol_sha256,
            "run_root": str(run_root.resolve()),
        },
        "numerical": numerical,
        "probe_passed": passed,
        "runs": {
            label: {
                "receipt_sha256": _sha256(run_root / label / "run" / "receipt.json"),
                "summary": summaries[label],
                "telemetry": telemetry[label],
            }
            for label, _ in RUNS
        },
        "selected_candidate_for_full_efficiency_gate": "static_194" if passed else None,
        "test_access": protocol["test_access"],
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
    if args.output.exists():
        raise ShapeProbeAuditError("Audit output must be create-only")
    result = audit_probe(
        run_root=args.run_root,
        protocol_path=args.protocol,
        bundle_root=args.bundle_root,
    )
    _publish(args.output, result)


if __name__ == "__main__":
    main()
