#!/usr/bin/env python3
"""Audit the non-authorizing four-GPU hybrid-shape crossover probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_four_gpu_shape_stability_probe as base
import freeze_antibody_four_gpu_shape_stability_probe_v2 as freezer


FORMAT_VERSION = "antibody_four_gpu_shape_stability_result_v2"
RUNS = (
    ("static_194_r1", "static_194"),
    ("hybrid_160_194_r1", "hybrid_160_194"),
    ("hybrid_128_194_r1", "hybrid_128_194"),
    ("hybrid_128_194_r2", "hybrid_128_194"),
    ("hybrid_160_194_r2", "hybrid_160_194"),
    ("static_194_r2", "static_194"),
)
HYBRID_CANDIDATES = ("hybrid_160_194", "hybrid_128_194")


class ShapeProbeV2AuditError(RuntimeError):
    """Report invalid evidence or a broken hybrid-shape contract."""


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
        return base._load(path)
    except base.ShapeProbeAuditError as error:
        raise ShapeProbeV2AuditError(str(error)) from error


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
        raise ShapeProbeV2AuditError("Master hybrid-shape protocol differs")


def _validate_gate(gate: Mapping[str, Any], candidate: str) -> None:
    if (
        candidate not in freezer.CANDIDATES
        or gate.get("format_version") != "antibody_production_loader_recovery_gate_v1"
        or gate.get("hardware_identity") != freezer.HARDWARE
        or gate.get("authorization") != base.parent.RUN_AUTHORIZATION
        or gate.get("test_access") != base.parent.RUN_TEST_ACCESS
        or gate.get("execution", {}).get("world_size") != 4
        or gate.get("execution", {}).get("backend") != "ddp"
        or gate.get("execution", {}).get("optimization")
        != base._expected_optimization()
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
        raise ShapeProbeV2AuditError(f"Candidate gate identity differs: {candidate}")


def _select_candidate(
    configurations: Mapping[str, Mapping[str, Any]],
    *,
    minimum_relative_throughput: float,
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    reference_rate = float(
        configurations["static_194"]["pooled"]["residue_tokens_per_second"]
    )
    decisions: dict[str, dict[str, Any]] = {}
    for candidate in HYBRID_CANDIDATES:
        rate = float(configurations[candidate]["pooled"]["residue_tokens_per_second"])
        ratio = rate / reference_rate
        decisions[candidate] = {
            "eligible": bool(configurations[candidate]["eligible"]),
            "passed_relative_throughput": ratio >= minimum_relative_throughput,
            "throughput_relative_to_static_194": ratio,
        }
    qualified = [
        candidate
        for candidate, decision in decisions.items()
        if decision["eligible"] and decision["passed_relative_throughput"]
    ]
    selected = max(
        qualified,
        key=lambda name: decisions[name]["throughput_relative_to_static_194"],
        default=None,
    )
    return selected, decisions


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
            raise ShapeProbeV2AuditError(f"Source closure differs: {relative}")

    candidate_gates: dict[str, tuple[dict[str, Any], str]] = {}
    for candidate in freezer.CANDIDATES:
        gate, digest = _load(bundle_root / f"{candidate}_gate.json")
        if digest != protocol["mode_gate_file_sha256"][candidate]:
            raise ShapeProbeV2AuditError(f"Candidate gate hash differs: {candidate}")
        _validate_gate(gate, candidate)
        candidate_gates[candidate] = (gate, digest)

    artifact_audit = base.parent.artifacts._verify_artifacts(run_root)
    receipts: dict[str, dict[str, Any]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    allocation = None
    updates = int(protocol["batch"]["updates_per_run"])
    for label, candidate in RUNS:
        gate, gate_sha256 = candidate_gates[candidate]
        receipt, _ = _load(run_root / label / "run" / "receipt.json")
        state, _ = _load(run_root / label / "run" / "trainer_state.json")
        optimization = base._expected_optimization()
        if (
            receipt.get("format_version") != base.parent.RUN_FORMAT
            or receipt.get("run_role") != "continuous"
            or receipt.get("world_size") != 4
            or receipt.get("device") != base.parent.GPU_MODEL
            or receipt.get("authorization") != base.parent.RUN_AUTHORIZATION
            or receipt.get("test_access") != base.parent.RUN_TEST_ACCESS
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
            raise ShapeProbeV2AuditError(f"Run identity differs: {label}")
        identity = tuple(
            receipt.get("allocation", {}).get(key)
            for key in ("hostname", "slurm_job_id", "slurm_job_nodelist")
        )
        if allocation is None:
            allocation = identity
        elif identity != allocation:
            raise ShapeProbeV2AuditError("Runs did not share one allocation")
        try:
            base.parent._validate_trace(receipt, updates)
            telemetry[label] = base.parent._telemetry(run_root / label / "gpu.csv")
        except base.parent.EfficiencyAuditError as error:
            raise ShapeProbeV2AuditError(str(error)) from error
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
    membership_equal = len(
        {item["membership_sha256"] for item in receipts.values()}
    ) == 1
    membership_equal = membership_equal and all(
        all(
            all(a[key] == b[key] for key in membership_fields)
            for a, b in zip(reference["trace"], receipt["trace"])
        )
        for receipt in receipts.values()
    )

    acceptance = protocol["acceptance"]
    summaries = {
        label: base.parent._throughput(
            receipt,
            int(acceptance["warmup_updates_excluded"]),
        )
        for label, receipt in receipts.items()
    }
    for label, summary in summaries.items():
        summary["maximum_memory_fraction"] = max(
            base._finite(receipts[label]["peak_allocated_memory_mib"], "allocated memory"),
            base._finite(receipts[label]["peak_reserved_memory_mib"], "reserved memory"),
            telemetry[label]["maximum_memory_mib"],
        ) / telemetry[label]["total_memory_mib"]

    numerical: dict[str, dict[str, Any]] = {}
    for candidate in HYBRID_CANDIDATES:
        for replicate in ("r1", "r2"):
            numerical[f"{candidate}_{replicate}"] = base._short_horizon(
                receipts[f"static_194_{replicate}"],
                receipts[f"{candidate}_{replicate}"],
                limit=int(acceptance["short_horizon_updates"]),
                acceptance=acceptance,
            )
    configurations = {
        candidate: base._configuration_summary(
            [label for label, item in RUNS if item == candidate],
            summaries,
            receipts,
            acceptance,
        )
        for candidate in freezer.CANDIDATES
    }
    selected, candidate_decisions = _select_candidate(
        configurations,
        minimum_relative_throughput=float(
            acceptance["minimum_candidate_throughput_relative_to_static_194"]
        ),
    )
    global_gates = {
        "artifact_integrity": artifact_audit["manifest_verified"]
        and not artifact_audit["writable_paths"],
        "gpu_identity": gpu_identity_equal,
        "initial_model": initial_model_equal,
        "membership": membership_equal,
        "short_horizon_numerical": all(item["passed"] for item in numerical.values()),
    }
    passed = all(global_gates.values()) and selected is not None
    if not passed:
        selected = None
    return {
        "artifact_audit": artifact_audit,
        "authorization": {
            "configuration_promotion": False,
            "full_efficiency_gate_candidate": passed,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "candidate_decisions": candidate_decisions,
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
        "selected_candidate_for_full_efficiency_gate": selected,
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
        raise ShapeProbeV2AuditError("Audit output must be create-only")
    result = audit_probe(
        run_root=args.run_root,
        protocol_path=args.protocol,
        bundle_root=args.bundle_root,
    )
    _publish(args.output, result)


if __name__ == "__main__":
    main()
