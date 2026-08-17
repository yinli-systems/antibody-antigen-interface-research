#!/usr/bin/env python3
"""Freeze a non-authorizing four-GPU hybrid-shape crossover probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import freeze_antibody_four_gpu_shape_stability_probe as parent


FORMAT_VERSION = "antibody_four_gpu_shape_stability_probe_v2"
RUN_ORDER = (
    "static_194_r1",
    "hybrid_160_194_r1",
    "hybrid_128_194_r1",
    "hybrid_128_194_r2",
    "hybrid_160_194_r2",
    "static_194_r2",
)
CANDIDATES = {
    "static_194": "194",
    "hybrid_160_194": "160,194",
    "hybrid_128_194": "128,194",
}
TEST_ACCESS = dict(parent.TEST_ACCESS)
AUTHORIZATION = dict(parent.AUTHORIZATION)
ACCEPTANCE = {
    **parent.ACCEPTANCE,
    "minimum_candidate_throughput_relative_to_static_194": 1.05,
}
BATCH = dict(parent.BATCH)
HARDWARE = dict(parent.HARDWARE)
SOURCE_FILES = (
    "scripts/audit_antibody_four_gpu_efficiency_gate.py",
    "scripts/audit_antibody_four_gpu_shape_stability_probe.py",
    "scripts/audit_antibody_four_gpu_shape_stability_probe_v2.py",
    "scripts/audit_antibody_production_batch_efficiency.py",
    "scripts/freeze_antibody_four_gpu_efficiency_gate.py",
    "scripts/freeze_antibody_four_gpu_recovery_gate.py",
    "scripts/freeze_antibody_four_gpu_shape_stability_probe.py",
    "scripts/freeze_antibody_four_gpu_shape_stability_probe_v2.py",
    "scripts/run_antibody_pretraining_production_gate.py",
    "slurm/antibody_four_gpu_shape_stability_probe_v2.sbatch",
)


class ShapeProbeV2FreezeError(RuntimeError):
    """Report an invalid prerequisite or hybrid-shape source closure."""


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
            raise ShapeProbeV2FreezeError(f"Source closure is invalid: {relative}")
        result[relative] = _sha256(path)
    return result


def _candidate_gate(base: dict[str, Any], candidate: str) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ShapeProbeV2FreezeError(f"Unknown candidate: {candidate}")
    gate = copy.deepcopy(base)
    expected_keys = set(
        parent.efficiency_freezer._mode_gate(base, compiled=True)["execution"][
            "optimization"
        ]
    )
    optimization = gate["execution"]["optimization"]
    if set(optimization) != expected_keys:
        raise ShapeProbeV2FreezeError("Base optimization contract changed")
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
        "Four-GPU real-OAS hybrid-shape systems probe only; no checkpoint, "
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
    recovery = parent.efficiency_freezer._load_recovery(
        recovery_audit_path,
        expected_file_sha256=recovery_audit_file_sha256,
        expected_payload_sha256=recovery_audit_payload_sha256,
    )
    base = parent.recovery_freezer.freeze_gate(
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
            "Non-authorizing four-GPU real-OAS A/B/C/C/B/A hybrid-shape probe. "
            "The sole decision is whether one two-boundary compiled candidate may "
            "enter a fresh full efficiency and checkpoint-equivalence gate."
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
            "Among eligible hybrid candidates, select the highest pooled throughput "
            "only if it is at least 1.05x static_194. All original numerical, MFU, "
            "memory, replicate, and raw step-stability thresholds remain unchanged."
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
        raise ShapeProbeV2FreezeError("Protocol bundle output must be create-only")
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
