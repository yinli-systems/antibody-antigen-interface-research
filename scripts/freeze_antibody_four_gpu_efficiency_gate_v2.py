#!/usr/bin/env python3
"""Freeze the static-shape four-GPU OAS efficiency and equivalence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import freeze_antibody_four_gpu_efficiency_gate as parent
import freeze_antibody_four_gpu_shape_stability_probe as shape_freezer


FORMAT_VERSION = "antibody_four_gpu_efficiency_gate_v2"
SHAPE_RESULT_FORMAT = "antibody_four_gpu_shape_stability_result_v1"
ACCEPTANCE = parent.ACCEPTANCE
AUTHORIZATION = parent.AUTHORIZATION
BATCH = parent.BATCH
HARDWARE = parent.HARDWARE
MEASUREMENT = parent.MEASUREMENT
TEST_ACCESS = parent.TEST_ACCESS
VALIDATION_PANEL = parent.VALIDATION_PANEL
STATIC_BOUNDARY = "194"
BUCKETED_BOUNDARIES = "96,128,160,194"
SOURCE_FILES = (
    "scripts/audit_antibody_four_gpu_efficiency_gate.py",
    "scripts/audit_antibody_four_gpu_efficiency_gate_v2.py",
    "scripts/audit_antibody_four_gpu_shape_stability_probe.py",
    "scripts/audit_antibody_production_batch_efficiency.py",
    "scripts/evaluate_antibody_production_checkpoint.py",
    "scripts/freeze_antibody_four_gpu_efficiency_gate.py",
    "scripts/freeze_antibody_four_gpu_efficiency_gate_v2.py",
    "scripts/freeze_antibody_four_gpu_recovery_gate.py",
    "scripts/freeze_antibody_four_gpu_shape_stability_probe.py",
    "scripts/run_antibody_pretraining_production_gate.py",
    "slurm/antibody_four_gpu_efficiency_pipeline.sbatch",
    "slurm/antibody_four_gpu_efficiency_pipeline_v2.sbatch",
)


class EfficiencyV2FreezeError(RuntimeError):
    """Report an invalid static-shape prerequisite or protocol bundle."""


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
            raise EfficiencyV2FreezeError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise EfficiencyV2FreezeError(f"Non-finite JSON constant: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_shape_probe(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
    expected_protocol_file_sha256: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise EfficiencyV2FreezeError("Shape-probe audit path is invalid")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise EfficiencyV2FreezeError("Shape-probe audit file hash differs")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EfficiencyV2FreezeError(
            "Shape-probe audit is not strict ASCII JSON"
        ) from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise EfficiencyV2FreezeError("Shape-probe audit is not canonical JSON")
    payload = dict(value)
    receipt_sha256 = payload.pop("receipt_sha256", None)
    static = value.get("configurations", {}).get("static_194", {})
    if (
        receipt_sha256 != expected_payload_sha256
        or hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        != expected_payload_sha256
        or value.get("format_version") != SHAPE_RESULT_FORMAT
        or value.get("probe_passed") is not True
        or value.get("selected_candidate_for_full_efficiency_gate") != "static_194"
        or value.get("authorization")
        != {
            "configuration_promotion": False,
            "full_efficiency_gate_candidate": True,
            "production_pretraining": False,
            "scientific_claim": False,
        }
        or value.get("test_access") != TEST_ACCESS
        or value.get("inputs", {}).get("protocol_file_sha256")
        != expected_protocol_file_sha256
        or not isinstance(value.get("global_gates"), dict)
        or not value["global_gates"]
        or not all(item is True for item in value["global_gates"].values())
        or static.get("eligible") is not True
        or not isinstance(static.get("gates"), dict)
        or not static["gates"]
        or not all(item is True for item in static["gates"].values())
    ):
        raise EfficiencyV2FreezeError(
            "Shape probe does not authorize a full static-shape efficiency gate"
        )
    return value


def _source_hashes(source_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise EfficiencyV2FreezeError(f"Source closure is invalid: {relative}")
        result[relative] = _sha256(path)
    return result


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
    shape_probe_audit_path: Path,
    shape_probe_audit_file_sha256: str,
    shape_probe_audit_payload_sha256: str,
    shape_probe_protocol_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return a master protocol and runner-compatible eager/static gates."""
    recovery = parent._load_recovery(
        recovery_audit_path,
        expected_file_sha256=recovery_audit_file_sha256,
        expected_payload_sha256=recovery_audit_payload_sha256,
    )
    shape = _load_shape_probe(
        shape_probe_audit_path,
        expected_file_sha256=shape_probe_audit_file_sha256,
        expected_payload_sha256=shape_probe_audit_payload_sha256,
        expected_protocol_file_sha256=shape_probe_protocol_file_sha256,
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
    compiled = parent._mode_gate(base, compiled=True)
    safe_eager = parent._mode_gate(base, compiled=False)
    compiled["execution"]["length_bucket_boundaries"] = STATIC_BOUNDARY
    safe_eager["execution"]["length_bucket_boundaries"] = BUCKETED_BOUNDARIES
    gates = {"compiled": compiled, "safe_eager": safe_eager}
    gate_hashes = {
        name: hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
        for name, value in gates.items()
    }
    master = {
        "acceptance": ACCEPTANCE,
        "authorization": AUTHORIZATION,
        "batch": BATCH,
        "claim_boundary": (
            "Four-GPU real-OAS static-shape A/B/B/A efficiency and "
            "checkpoint-equivalence gate; no production pretraining, "
            "downstream, official-test, biological, or SOTA claim"
        ),
        "corpus": base["corpus"],
        "format_version": FORMAT_VERSION,
        "hardware": HARDWARE,
        "measurement": MEASUREMENT,
        "mode_gate_file_sha256": gate_hashes,
        "recovery_prerequisite": {
            "audit_file_sha256": recovery_audit_file_sha256,
            "audit_payload_sha256": recovery_audit_payload_sha256,
            "gate_protocol_file_sha256": recovery["gate_protocol_file_sha256"],
        },
        "selected_shape_candidate": {
            "compiled_length_bucket_boundaries": STATIC_BOUNDARY,
            "name": "static_194",
            "probe_audit_file_sha256": shape_probe_audit_file_sha256,
            "probe_audit_payload_sha256": shape_probe_audit_payload_sha256,
            "probe_protocol_file_sha256": shape_probe_protocol_file_sha256,
            "probe_static_configuration": shape["configurations"]["static_194"],
        },
        "source": _source_hashes(source_root),
        "test_access": TEST_ACCESS,
        "validation_panel": VALIDATION_PANEL,
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
    parser.add_argument("--shape-probe-audit", type=Path, required=True)
    parser.add_argument("--shape-probe-audit-file-sha256", required=True)
    parser.add_argument("--shape-probe-audit-payload-sha256", required=True)
    parser.add_argument("--shape-probe-protocol-file-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise EfficiencyV2FreezeError("Protocol bundle output must be create-only")
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
        shape_probe_audit_path=args.shape_probe_audit,
        shape_probe_audit_file_sha256=args.shape_probe_audit_file_sha256,
        shape_probe_audit_payload_sha256=args.shape_probe_audit_payload_sha256,
        shape_probe_protocol_file_sha256=args.shape_probe_protocol_file_sha256,
    )
    args.output_dir.mkdir(mode=0o700, parents=True)
    _publish(args.output_dir / "compiled_gate.json", gates["compiled"])
    _publish(args.output_dir / "safe_eager_gate.json", gates["safe_eager"])
    _publish(args.output_dir / "protocol.json", master)
    args.output_dir.chmod(0o555)


if __name__ == "__main__":
    main()
