#!/usr/bin/env python3
"""Freeze a non-authorizing four-GPU OAS shape-stability crossover."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import freeze_antibody_four_gpu_efficiency_gate as efficiency_freezer
import freeze_antibody_four_gpu_recovery_gate as recovery_freezer


FORMAT_VERSION = "antibody_four_gpu_shape_stability_probe_v1"
RUN_ORDER = (
    "bucketed_r1",
    "static_194_r1",
    "static_194_r2",
    "bucketed_r2",
)
CANDIDATES = {
    "bucketed": "96,128,160,194",
    "static_194": "194",
}
TEST_ACCESS = dict(efficiency_freezer.TEST_ACCESS)
AUTHORIZATION = {
    "configuration_promotion": False,
    "full_efficiency_gate_candidate": False,
    "production_pretraining": False,
    "scientific_claim": False,
}
ACCEPTANCE = {
    "maximum_memory_fraction": 0.80,
    "maximum_p95_over_p50_step_time": 1.10,
    "maximum_replicate_throughput_relative_range": 0.05,
    "maximum_short_horizon_gradient_norm_relative_difference": 0.01,
    "maximum_short_horizon_loss_absolute_difference": 0.005,
    "minimum_approximate_6n_mfu": 0.35,
    "short_horizon_updates": 20,
    "warmup_updates_excluded": 100,
}
BATCH = {
    "batch_quantum_sequences": 256,
    "global_residue_token_target_per_update": 131072,
    "micro_batch_per_gpu": 16,
    "updates_per_run": 300,
}
HARDWARE = dict(efficiency_freezer.HARDWARE)
SOURCE_FILES = (
    "scripts/audit_antibody_four_gpu_efficiency_gate.py",
    "scripts/audit_antibody_four_gpu_shape_stability_probe.py",
    "scripts/audit_antibody_production_batch_efficiency.py",
    "scripts/freeze_antibody_four_gpu_efficiency_gate.py",
    "scripts/freeze_antibody_four_gpu_recovery_gate.py",
    "scripts/freeze_antibody_four_gpu_shape_stability_probe.py",
    "scripts/run_antibody_pretraining_production_gate.py",
    "slurm/antibody_four_gpu_shape_stability_probe.sbatch",
)


class ShapeProbeFreezeError(RuntimeError):
    """Report an invalid prerequisite or source closure."""


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


def _source_hashes(source_root: Path) -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise ShapeProbeFreezeError(f"Source closure is invalid: {relative}")
        result[relative] = _sha256(path)
    return result


def _candidate_gate(base: dict[str, Any], candidate: str) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ShapeProbeFreezeError(f"Unknown candidate: {candidate}")
    gate = copy.deepcopy(base)
    expected_keys = set(efficiency_freezer._mode_gate(base, compiled=True)["execution"]["optimization"])
    optimization = gate["execution"]["optimization"]
    if set(optimization) != expected_keys:
        raise ShapeProbeFreezeError("Base optimization contract changed")
    gate["execution"]["optimization"] = {
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
    gate["execution"]["length_bucket_boundaries"] = CANDIDATES[candidate]
    gate["runs"] = {
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
    gate["claim_boundary"] = (
        "Four-GPU real-OAS static-shape systems probe only; no checkpoint, "
        "configuration promotion, production pretraining, downstream, official-test, "
        "biological, or SOTA claim"
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
    recovery = efficiency_freezer._load_recovery(
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
    gates = {name: _candidate_gate(base, name) for name in CANDIDATES}
    gate_hashes = {
        name: hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
        for name, value in gates.items()
    }
    master = {
        "acceptance": ACCEPTANCE,
        "authorization": AUTHORIZATION,
        "batch": BATCH,
        "candidates": CANDIDATES,
        "claim_boundary": (
            "Non-authorizing four-GPU real-OAS A/B/B/A systems probe. The sole "
            "decision is whether static width 194 may enter a fresh full efficiency "
            "and checkpoint-equivalence gate."
        ),
        "format_version": FORMAT_VERSION,
        "hardware": HARDWARE,
        "mode_gate_file_sha256": gate_hashes,
        "recovery_prerequisite": {
            "audit_file_sha256": recovery_audit_file_sha256,
            "audit_payload_sha256": recovery_audit_payload_sha256,
            "gate_protocol_file_sha256": recovery["gate_protocol_file_sha256"],
        },
        "run_order": list(RUN_ORDER),
        "selection_rule": (
            "Select static_194 for a fresh full efficiency gate only if artifact, "
            "membership, allocation, GPU identity, short-horizon numerical, MFU, "
            "memory, replicate-state, replicate-throughput, and raw step-stability "
            "gates all pass without changing thresholds."
        ),
        "source": _source_hashes(source_root),
        "test_access": TEST_ACCESS,
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
        raise ShapeProbeFreezeError("Protocol bundle output must be create-only")
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
    for name, gate in gates.items():
        _publish(args.output_dir / f"{name}_gate.json", gate)
    _publish(args.output_dir / "protocol.json", master)
    args.output_dir.chmod(0o555)


if __name__ == "__main__":
    main()
