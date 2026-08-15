#!/usr/bin/env python3
"""Freeze a real-OAS four-GPU paired efficiency and equivalence gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import freeze_antibody_four_gpu_recovery_gate as recovery_freezer


FORMAT_VERSION = "antibody_four_gpu_efficiency_gate_v1"
RECOVERY_FORMAT = "antibody_production_loader_recovery_audit_4gpu_v1"
RUN_ORDER = (
    "safe_eager_r1",
    "compiled_r1",
    "compiled_r2",
    "safe_eager_r2",
)
PANEL_COUNT = 8192
PANEL_SEED = 2026081501
PANEL_INDICES_SHA256 = (
    "cc9219f83bfdaef51e97ee9845f31ba9dd2cbde9863715d6fdfa7a50478197ec"
)
TEST_ACCESS = {
    "downstream_labels_used": False,
    "external_test_accessed": False,
    "frozen_test_accessed": False,
    "official_test_accessed": False,
    "sabdab2_heldout_accessed": False,
}
SOURCE_FILES = (
    "scripts/audit_antibody_four_gpu_efficiency_gate.py",
    "scripts/evaluate_antibody_production_checkpoint.py",
    "scripts/freeze_antibody_four_gpu_efficiency_gate.py",
    "scripts/run_antibody_pretraining_production_gate.py",
    "slurm/antibody_four_gpu_efficiency_pipeline.sbatch",
)


class EfficiencyFreezeError(RuntimeError):
    """Report an invalid recovery prerequisite or protocol bundle."""


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


def _load_recovery(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise EfficiencyFreezeError("Recovery audit path is invalid")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise EfficiencyFreezeError("Recovery audit file hash differs")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EfficiencyFreezeError("Recovery audit is not strict ASCII JSON") from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise EfficiencyFreezeError("Recovery audit is not canonical JSON")
    payload = dict(value)
    receipt_sha256 = payload.pop("receipt_sha256", None)
    if (
        receipt_sha256 != expected_payload_sha256
        or hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        != expected_payload_sha256
        or value.get("format_version") != RECOVERY_FORMAT
        or value.get("world_size") != 4
        or value.get("authorization", {}).get("production_pretraining") is not False
        or value.get("artifact_audit", {}).get("manifest_verified") is not True
        or value.get("artifact_audit", {}).get("writable_paths") != []
        or value.get("exact_recovery")
        != {
            "final_model_equal": True,
            "final_optimizer_equal": True,
            "scientific_trace_equal": True,
            "terminal_coordinates_equal": True,
        }
        or any(bool(item) for item in value.get("test_access", {}).values())
    ):
        raise EfficiencyFreezeError("Recovery audit does not authorize efficiency work")
    return value


def _source_hashes(source_root: Path) -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise EfficiencyFreezeError(f"Source closure is invalid: {relative}")
        result[relative] = _sha256(path)
    return result


def _mode_gate(base: dict[str, Any], *, compiled: bool) -> dict[str, Any]:
    gate = copy.deepcopy(base)
    optimization = gate["execution"]["optimization"]
    if compiled:
        expected = {
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
    else:
        expected = {
            "bucketed_padding": False,
            "compile_blocks": False,
            "compile_prewarm_buckets": False,
            "ddp_bucket_cap_mb": 25,
            "deterministic_algorithms": True,
            "disable_compiled_cudagraphs": False,
            "fused_adamw": False,
            "length_bucketing": False,
            "pinned_staging": False,
        }
    if set(optimization) != set(expected):
        raise EfficiencyFreezeError("Base optimization contract changed")
    gate["execution"]["optimization"] = expected
    gate["runs"] = {
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
    gate["claim_boundary"] = (
        "Four-GPU real-OAS paired efficiency candidate input only; no "
        "configuration promotion, production pretraining, downstream, "
        "official-test, biological, or SOTA claim"
    )
    return gate


def freeze_bundle(
    *,
    source_root: Path,
    corpus_root: Path,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    corpus_protocol_file_sha256: str,
    independent_audit_path: Path,
    independent_audit_file_sha256: str,
    independent_audit_payload_sha256: str,
    recovery_audit_path: Path,
    recovery_audit_file_sha256: str,
    recovery_audit_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return the master protocol and two runner-compatible mode gates."""
    recovery = _load_recovery(
        recovery_audit_path,
        expected_file_sha256=recovery_audit_file_sha256,
        expected_payload_sha256=recovery_audit_payload_sha256,
    )
    base = recovery_freezer.freeze_gate(
        source_root=source_root,
        corpus_root=corpus_root,
        manifest_file_sha256=manifest_file_sha256,
        manifest_payload_sha256=manifest_payload_sha256,
        corpus_protocol_file_sha256=corpus_protocol_file_sha256,
        independent_audit_path=independent_audit_path,
        independent_audit_file_sha256=independent_audit_file_sha256,
        independent_audit_payload_sha256=independent_audit_payload_sha256,
    )
    gates = {
        "compiled": _mode_gate(base, compiled=True),
        "safe_eager": _mode_gate(base, compiled=False),
    }
    gate_hashes = {
        name: hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
        for name, value in gates.items()
    }
    master = {
        "acceptance": {
            "bootstrap_confidence": 0.95,
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 2026081502,
            "compiled_relative_to_eager_each_round": 1.05,
            "compiled_relative_to_eager_pooled": 1.05,
            "maximum_masked_accuracy_absolute_decrease": 0.005,
            "maximum_masked_cross_entropy_relative_increase": 0.01,
            "maximum_memory_fraction": 0.80,
            "maximum_p95_over_p50_step_time": 1.10,
            "maximum_replicate_throughput_relative_range": 0.05,
            "maximum_short_horizon_gradient_norm_relative_difference": 0.01,
            "maximum_short_horizon_loss_absolute_difference": 0.005,
            "minimum_approximate_6n_mfu": 0.35,
            "short_horizon_updates": 20,
            "warmup_updates_excluded": 100,
        },
        "authorization": {
            "configuration_promotion": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "batch": {
            "batch_quantum_sequences": 256,
            "global_residue_token_target_per_update": 131072,
            "micro_batch_per_gpu": 16,
            "updates_per_run": 600,
        },
        "claim_boundary": (
            "Four-GPU real-OAS A/B/B/A efficiency and checkpoint-equivalence "
            "gate; no production pretraining, downstream, official-test, "
            "biological, or SOTA claim"
        ),
        "corpus": base["corpus"],
        "format_version": FORMAT_VERSION,
        "hardware": {
            "gpu_count": 4,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "node_count": 1,
        },
        "measurement": {
            "approximate_6n_mfu": {
                "denominator_dense_bf16_tflops_per_gpu": 209.5,
                "formula": "6*N*nonpadding_residue_tokens_per_second/(4*209.5e12)",
                "parameter_count": 634513920,
                "profiler_measured": False,
            },
            "crossover_run_order": list(RUN_ORDER),
            "same_allocation_required": True,
            "same_membership_required": True,
        },
        "mode_gate_file_sha256": gate_hashes,
        "recovery_prerequisite": {
            "audit_file_sha256": recovery_audit_file_sha256,
            "audit_payload_sha256": recovery_audit_payload_sha256,
            "gate_protocol_file_sha256": recovery["gate_protocol_file_sha256"],
        },
        "source": _source_hashes(source_root),
        "test_access": TEST_ACCESS,
        "validation_panel": {
            "indices_sha256": PANEL_INDICES_SHA256,
            "mask_seed": 2026081503,
            "mask_update": 0,
            "record_count": PANEL_COUNT,
            "seed": PANEL_SEED,
            "split": "validation",
        },
    }
    return master, gates


def _publish(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--manifest-payload-sha256", required=True)
    parser.add_argument("--corpus-protocol-file-sha256", required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--independent-audit-file-sha256", required=True)
    parser.add_argument("--independent-audit-payload-sha256", required=True)
    parser.add_argument("--recovery-audit", type=Path, required=True)
    parser.add_argument("--recovery-audit-file-sha256", required=True)
    parser.add_argument("--recovery-audit-payload-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise EfficiencyFreezeError("Protocol bundle output must be create-only")
    master, gates = freeze_bundle(
        source_root=args.source_root,
        corpus_root=args.corpus_root,
        manifest_file_sha256=args.manifest_file_sha256,
        manifest_payload_sha256=args.manifest_payload_sha256,
        corpus_protocol_file_sha256=args.corpus_protocol_file_sha256,
        independent_audit_path=args.independent_audit,
        independent_audit_file_sha256=args.independent_audit_file_sha256,
        independent_audit_payload_sha256=args.independent_audit_payload_sha256,
        recovery_audit_path=args.recovery_audit,
        recovery_audit_file_sha256=args.recovery_audit_file_sha256,
        recovery_audit_payload_sha256=args.recovery_audit_payload_sha256,
    )
    args.output_dir.mkdir(mode=0o700, parents=True)
    _publish(args.output_dir / "compiled_gate.json", gates["compiled"])
    _publish(args.output_dir / "safe_eager_gate.json", gates["safe_eager"])
    _publish(args.output_dir / "protocol.json", master)
    args.output_dir.chmod(0o555)


if __name__ == "__main__":
    main()
