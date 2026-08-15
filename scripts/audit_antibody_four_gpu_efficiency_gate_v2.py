#!/usr/bin/env python3
"""Audit the static-shape four-GPU OAS efficiency and equivalence gate."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_four_gpu_efficiency_gate as parent
import freeze_antibody_four_gpu_efficiency_gate_v2 as freezer


FORMAT_VERSION = "antibody_four_gpu_efficiency_result_v2"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EfficiencyV2AuditError(RuntimeError):
    """Report a broken static-shape efficiency contract."""


def _validate_v2_protocol(
    protocol: Mapping[str, Any],
    *,
    bundle_root: Path,
) -> None:
    selected = protocol.get("selected_shape_candidate")
    if not isinstance(selected, Mapping):
        raise EfficiencyV2AuditError("Selected shape candidate is absent")
    static = selected.get("probe_static_configuration")
    if (
        selected.get("name") != "static_194"
        or selected.get("compiled_length_bucket_boundaries")
        != freezer.STATIC_BOUNDARY
        or any(
            SHA256.fullmatch(str(selected.get(key, ""))) is None
            for key in (
                "probe_audit_file_sha256",
                "probe_audit_payload_sha256",
                "probe_protocol_file_sha256",
            )
        )
        or not isinstance(static, Mapping)
        or static.get("eligible") is not True
        or not isinstance(static.get("gates"), Mapping)
        or not static["gates"]
        or not all(item is True for item in static["gates"].values())
    ):
        raise EfficiencyV2AuditError("Static-shape prerequisite identity differs")
    compiled, compiled_sha256 = parent._load(bundle_root / "compiled_gate.json")
    eager, eager_sha256 = parent._load(bundle_root / "safe_eager_gate.json")
    if (
        compiled_sha256 != protocol["mode_gate_file_sha256"]["compiled"]
        or eager_sha256 != protocol["mode_gate_file_sha256"]["safe_eager"]
        or compiled.get("execution", {}).get("length_bucket_boundaries")
        != freezer.STATIC_BOUNDARY
        or eager.get("execution", {}).get("length_bucket_boundaries")
        != freezer.BUCKETED_BOUNDARIES
    ):
        raise EfficiencyV2AuditError("Runner mode shape contract differs")


def audit_gate(
    *,
    run_root: Path,
    protocol_path: Path,
    bundle_root: Path,
) -> dict[str, Any]:
    """Audit one full static-shape A/B/B/A execution."""
    protocol, _ = parent._load(protocol_path)
    parent._validate_master_protocol(protocol, protocol_freezer=freezer)
    _validate_v2_protocol(protocol, bundle_root=bundle_root)
    result = parent.audit_gate(
        run_root=run_root,
        protocol_path=protocol_path,
        bundle_root=bundle_root,
        protocol_freezer=freezer,
        result_format_version=FORMAT_VERSION,
        auditor_path=Path(__file__),
    )
    result["selected_shape_candidate"] = protocol["selected_shape_candidate"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise EfficiencyV2AuditError("Audit output must be create-only")
    result = audit_gate(
        run_root=args.run_root,
        protocol_path=args.protocol,
        bundle_root=args.bundle_root,
    )
    parent._publish(args.output, result)


if __name__ == "__main__":
    main()
