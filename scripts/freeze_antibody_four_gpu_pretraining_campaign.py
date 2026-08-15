#!/usr/bin/env python3
"""Freeze a resumable four-GPU OAS pretraining campaign after all gates pass."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import freeze_antibody_production_recovery_gate as recovery_freezer


FORMAT_VERSION = "antibody_four_gpu_pretraining_segment_gate_v1"
EFFICIENCY_FORMAT = "antibody_four_gpu_efficiency_result_v1"
EFFICIENCY_PROTOCOL_FORMAT = "antibody_four_gpu_efficiency_gate_v1"
RECOVERY_AUDIT_FORMAT = "antibody_production_loader_recovery_audit_4gpu_v1"
RECOVERY_GATE_FORMAT = "antibody_production_loader_recovery_gate_v1"
CAMPAIGN_TOKEN_LIMIT = 30_000_000_000
UNIQUE_TRAIN_RESIDUE_TOKENS = 25_259_993_406
UNIQUE_TRAIN_RECORDS = 224_887_833
VALIDATION_RECORDS = 1_743_380
SEGMENT_WALLTIME_SECONDS = 14_400
SEGMENT_RESERVE_SECONDS = 3_600
MINIMUM_SEGMENT_STEPS = 600
MAXIMUM_SEGMENT_STEPS = 6_000
TARGET_RESIDUE_TOKENS = 131_072
BATCH_QUANTUM_SEQUENCES = 256
MICRO_BATCH_PER_GPU = 16
RUN_SEED = 17
AUTHORIZATION = {
    "configuration_promotion": True,
    "downstream_transfer": False,
    "production_pretraining": True,
    "scientific_claim": False,
}
RECOVERY_AUTHORIZATION = {
    "loader_recovery_gate": True,
    "production_pretraining": False,
    "scale_out": False,
    "scientific_claim": False,
}
RECOVERY_AUDIT_AUTHORIZATION = {
    "production_pretraining": False,
    "scale_16_gpu": False,
    "scale_32_gpu": False,
    "scientific_claim": False,
}
EFFICIENCY_AUTHORIZATION = {
    "configuration_promotion": True,
    "production_pretraining": False,
    "scientific_claim": False,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
EFFICIENCY_TEST_ACCESS = {
    "downstream_labels_used": False,
    "external_test_accessed": False,
    "frozen_test_accessed": False,
    "official_test_accessed": False,
    "sabdab2_heldout_accessed": False,
}
RECOVERY_EXACT = {
    "final_model_equal": True,
    "final_optimizer_equal": True,
    "scientific_trace_equal": True,
    "terminal_coordinates_equal": True,
}
HARDWARE = {
    "gpu_count": 4,
    "gpu_model": "NVIDIA GeForce RTX 5090",
    "node_count": 1,
}
OPTIMIZATION = {
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
VALIDATION_PANEL = {
    "indices_sha256": (
        "cc9219f83bfdaef51e97ee9845f31ba9dd2cbde9863715d6fdfa7a50478197ec"
    ),
    "mask_seed": 2026081503,
    "mask_update": 0,
    "record_count": 8192,
    "seed": 2026081501,
    "split": "validation",
}
CAMPAIGN_SOURCE_FILES = (
    "scripts/audit_antibody_four_gpu_pretraining_segment.py",
    "scripts/evaluate_antibody_production_checkpoint.py",
    "scripts/freeze_antibody_four_gpu_pretraining_campaign.py",
    "scripts/run_antibody_pretraining_production_gate.py",
    "slurm/antibody_four_gpu_pretraining_segment.sbatch",
)


class CampaignFreezeError(RuntimeError):
    """Report a missing or incompatible scientific/systems prerequisite."""


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
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CampaignFreezeError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise CampaignFreezeError(f"Non-finite JSON constant: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_canonical(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CampaignFreezeError(f"Evidence path is invalid: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise CampaignFreezeError(f"Evidence file hash differs: {path}")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignFreezeError(f"Evidence is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise CampaignFreezeError(f"Evidence is not canonical JSON: {path}")
    return value


def _load_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    value = _load_canonical(path, expected_file_sha256)
    payload = dict(value)
    receipt_sha256 = payload.pop("receipt_sha256", None)
    if (
        receipt_sha256 != expected_payload_sha256
        or hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        != expected_payload_sha256
    ):
        raise CampaignFreezeError(f"Receipt payload hash differs: {path}")
    return value


def _current_implementation(source_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in recovery_freezer.IMPLEMENTATION_FILES:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise CampaignFreezeError(f"Implementation source is invalid: {relative}")
        result[relative] = _sha256(path)
    return result


def _campaign_source(source_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in CAMPAIGN_SOURCE_FILES:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise CampaignFreezeError(f"Campaign source is invalid: {relative}")
        result[relative] = _sha256(path)
    return result


def _validate_recovery(
    *,
    source_root: Path,
    gate_path: Path,
    gate_file_sha256: str,
    audit_path: Path,
    audit_file_sha256: str,
    audit_payload_sha256: str,
) -> dict[str, Any]:
    gate = _load_canonical(gate_path, gate_file_sha256)
    audit = _load_receipt(
        audit_path,
        expected_file_sha256=audit_file_sha256,
        expected_payload_sha256=audit_payload_sha256,
    )
    if (
        gate.get("format_version") != RECOVERY_GATE_FORMAT
        or gate.get("authorization") != RECOVERY_AUTHORIZATION
        or gate.get("test_access") != TEST_ACCESS
        or gate.get("hardware_identity") != HARDWARE
        or gate.get("execution", {}).get("world_size") != 4
        or gate.get("execution", {}).get("optimization") != OPTIMIZATION
        or gate.get("implementation") != _current_implementation(source_root)
        or audit.get("format_version") != RECOVERY_AUDIT_FORMAT
        or audit.get("authorization") != RECOVERY_AUDIT_AUTHORIZATION
        or audit.get("test_access") != TEST_ACCESS
        or audit.get("world_size") != 4
        or audit.get("gate_protocol_file_sha256") != gate_file_sha256
        or audit.get("exact_recovery") != RECOVERY_EXACT
        or audit.get("artifact_audit", {}).get("manifest_verified") is not True
        or audit.get("artifact_audit", {}).get("writable_paths") != []
    ):
        raise CampaignFreezeError("Exact-recovery prerequisite does not match source")
    return gate


def _validate_efficiency(
    *,
    protocol_path: Path,
    protocol_file_sha256: str,
    audit_path: Path,
    audit_file_sha256: str,
    audit_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_canonical(protocol_path, protocol_file_sha256)
    audit = _load_receipt(
        audit_path,
        expected_file_sha256=audit_file_sha256,
        expected_payload_sha256=audit_payload_sha256,
    )
    gates = audit.get("gates")
    compiled_runs = [
        audit.get("runs", {}).get(label, {}).get("summary", {})
        for label in ("compiled_r1", "compiled_r2")
    ]
    if (
        protocol.get("format_version") != EFFICIENCY_PROTOCOL_FORMAT
        or protocol.get("authorization")
        != {
            "configuration_promotion": False,
            "production_pretraining": False,
            "scientific_claim": False,
        }
        or protocol.get("test_access") != EFFICIENCY_TEST_ACCESS
        or audit.get("format_version") != EFFICIENCY_FORMAT
        or audit.get("authorization") != EFFICIENCY_AUTHORIZATION
        or audit.get("gate_passed") is not True
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
        or audit.get("inputs", {}).get("protocol_file_sha256")
        != protocol_file_sha256
        or audit.get("test_access") != protocol.get("test_access")
        or any(not isinstance(item, dict) for item in compiled_runs)
        or any(
            float(item.get("approximate_6n_mfu", -1.0))
            < float(protocol["acceptance"]["minimum_approximate_6n_mfu"])
            for item in compiled_runs
        )
    ):
        raise CampaignFreezeError("Efficiency prerequisite does not authorize promotion")
    return protocol, audit


def _segment_steps(efficiency_audit: dict[str, Any]) -> tuple[int, float]:
    p95_seconds = max(
        float(efficiency_audit["runs"][label]["summary"]["p95_step_seconds"])
        for label in ("compiled_r1", "compiled_r2")
    )
    if not math.isfinite(p95_seconds) or p95_seconds <= 0.0:
        raise CampaignFreezeError("Compiled p95 step time is invalid")
    steps = min(
        MAXIMUM_SEGMENT_STEPS,
        math.floor((SEGMENT_WALLTIME_SECONDS - SEGMENT_RESERVE_SECONDS) / p95_seconds),
    )
    if steps < MINIMUM_SEGMENT_STEPS:
        raise CampaignFreezeError("Measured throughput cannot fit a safe segment")
    return steps, p95_seconds


def freeze_campaign(
    *,
    source_root: Path,
    recovery_gate_path: Path,
    recovery_gate_file_sha256: str,
    recovery_audit_path: Path,
    recovery_audit_file_sha256: str,
    recovery_audit_payload_sha256: str,
    efficiency_protocol_path: Path,
    efficiency_protocol_file_sha256: str,
    efficiency_audit_path: Path,
    efficiency_audit_file_sha256: str,
    efficiency_audit_payload_sha256: str,
) -> dict[str, Any]:
    """Return one exact runner-compatible production campaign gate."""
    recovery_gate = _validate_recovery(
        source_root=source_root,
        gate_path=recovery_gate_path,
        gate_file_sha256=recovery_gate_file_sha256,
        audit_path=recovery_audit_path,
        audit_file_sha256=recovery_audit_file_sha256,
        audit_payload_sha256=recovery_audit_payload_sha256,
    )
    _, efficiency_audit = _validate_efficiency(
        protocol_path=efficiency_protocol_path,
        protocol_file_sha256=efficiency_protocol_file_sha256,
        audit_path=efficiency_audit_path,
        audit_file_sha256=efficiency_audit_file_sha256,
        audit_payload_sha256=efficiency_audit_payload_sha256,
    )
    segment_steps, compiled_p95_seconds = _segment_steps(efficiency_audit)
    gate = copy.deepcopy(recovery_gate)
    gate["format_version"] = FORMAT_VERSION
    gate["authorization"] = AUTHORIZATION
    gate["claim_boundary"] = (
        "Four-GPU OAS MLM production-pretraining campaign only; no downstream "
        "transfer, partner-specificity, official/frozen/external-test, biological, "
        "model-promotion, comparative-superiority, or SOTA claim"
    )
    gate["auditor_sha256"] = _sha256(
        source_root / "scripts/audit_antibody_four_gpu_pretraining_segment.py"
    )
    gate["campaign_source"] = _campaign_source(source_root)
    gate["campaign"] = {
        "authoritative_progress_unit": "admitted_residue_tokens_consumed",
        "campaign_token_limit": CAMPAIGN_TOKEN_LIMIT,
        "maximum_final_overrun_residue_tokens": (
            TARGET_RESIDUE_TOKENS + BATCH_QUANTUM_SEQUENCES * 192
        ),
        "planned_repeat_exposure_residue_tokens": (
            CAMPAIGN_TOKEN_LIMIT - UNIQUE_TRAIN_RESIDUE_TOKENS
        ),
        "segment_maximum_steps": segment_steps,
        "segment_reserve_seconds": SEGMENT_RESERVE_SECONDS,
        "segment_walltime_seconds": SEGMENT_WALLTIME_SECONDS,
        "unique_train_records": UNIQUE_TRAIN_RECORDS,
        "unique_train_residue_tokens": UNIQUE_TRAIN_RESIDUE_TOKENS,
        "validation_records": VALIDATION_RECORDS,
    }
    gate["prerequisites"] = {
        "efficiency_audit_file_sha256": efficiency_audit_file_sha256,
        "efficiency_audit_payload_sha256": efficiency_audit_payload_sha256,
        "efficiency_protocol_file_sha256": efficiency_protocol_file_sha256,
        "exact_recovery_audit_file_sha256": recovery_audit_file_sha256,
        "exact_recovery_audit_payload_sha256": recovery_audit_payload_sha256,
        "exact_recovery_gate_file_sha256": recovery_gate_file_sha256,
    }
    gate["systems_evidence"] = {
        "compiled_p95_step_seconds_upper_reference": compiled_p95_seconds,
        "compiled_pooled_approximate_6n_mfu": efficiency_audit["pooled"]["modes"][
            "compiled"
        ]["approximate_6n_mfu"],
        "compiled_pooled_residue_tokens_per_second": efficiency_audit["pooled"][
            "modes"
        ]["compiled"]["residue_tokens_per_second"],
        "profiler_measured_mfu": False,
    }

    def run(*, resume: bool) -> dict[str, object]:
        return {
            "batch_quantum_sequences": BATCH_QUANTUM_SEQUENCES,
            "campaign_token_limit": CAMPAIGN_TOKEN_LIMIT,
            "checkpoint": True,
            "micro_batch_per_gpu": MICRO_BATCH_PER_GPU,
            "preset": "abf_650m",
            "resume": resume,
            "run_seed": RUN_SEED,
            "steps": segment_steps,
            "target_residue_tokens": TARGET_RESIDUE_TOKENS,
        }

    gate["runs"] = {
        "resumed": run(resume=True),
        "stage1": run(resume=False),
    }
    gate["validation_panel"] = VALIDATION_PANEL
    if gate["execution"]["optimization"] != OPTIMIZATION:
        raise CampaignFreezeError("Promoted optimization differs from the passed gate")
    return gate


def _publish(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recovery-gate", type=Path, required=True)
    parser.add_argument("--recovery-gate-file-sha256", required=True)
    parser.add_argument("--recovery-audit", type=Path, required=True)
    parser.add_argument("--recovery-audit-file-sha256", required=True)
    parser.add_argument("--recovery-audit-payload-sha256", required=True)
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--efficiency-protocol-file-sha256", required=True)
    parser.add_argument("--efficiency-audit", type=Path, required=True)
    parser.add_argument("--efficiency-audit-file-sha256", required=True)
    parser.add_argument("--efficiency-audit-payload-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = freeze_campaign(
        source_root=args.source_root,
        recovery_gate_path=args.recovery_gate,
        recovery_gate_file_sha256=args.recovery_gate_file_sha256,
        recovery_audit_path=args.recovery_audit,
        recovery_audit_file_sha256=args.recovery_audit_file_sha256,
        recovery_audit_payload_sha256=args.recovery_audit_payload_sha256,
        efficiency_protocol_path=args.efficiency_protocol,
        efficiency_protocol_file_sha256=args.efficiency_protocol_file_sha256,
        efficiency_audit_path=args.efficiency_audit,
        efficiency_audit_file_sha256=args.efficiency_audit_file_sha256,
        efficiency_audit_payload_sha256=args.efficiency_audit_payload_sha256,
    )
    _publish(args.output, value)


if __name__ == "__main__":
    main()
