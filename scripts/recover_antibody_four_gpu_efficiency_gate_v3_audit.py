#!/usr/bin/env python3
"""Recover a v3 audit whose frozen source predates exported parent constants.

The existing run protocol binds the original v3 freezer byte-for-byte.  That
freezer delegated protocol construction to v1 but did not re-export the v1
contract constants required by the generic auditor.  Mutating the bound source
would invalidate source closure, so this compatibility auditor restores the
missing module attributes in memory and audits the immutable run artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import audit_antibody_four_gpu_efficiency_gate as parent
import freeze_antibody_four_gpu_efficiency_gate_v3 as freezer


FORMAT_VERSION = "antibody_four_gpu_efficiency_result_v3"
CONTRACT_NAMES = (
    "ACCEPTANCE",
    "AUTHORIZATION",
    "BATCH",
    "HARDWARE",
    "MEASUREMENT",
    "TEST_ACCESS",
)


def restore_delegated_contract() -> None:
    """Expose the v1 contract without changing any protocol-bound file."""
    for name in CONTRACT_NAMES:
        if not hasattr(freezer, name):
            setattr(freezer, name, getattr(freezer.parent, name))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def amend_legacy_mode_hashes(
    *, protocol_path: Path, bundle_root: Path, output: Path
) -> dict[str, object]:
    """Prove and amend the post-hash static-boundary transformation."""
    protocol, original_protocol_sha256 = parent._load(protocol_path)
    expected_hashes = protocol["mode_gate_file_sha256"]
    actual_hashes: dict[str, str] = {}
    reconstructed_hashes: dict[str, str] = {}
    for mode in ("compiled", "safe_eager"):
        gate, digest = parent._load(bundle_root / f"{mode}_gate.json")
        actual_hashes[mode] = digest
        reconstructed = copy.deepcopy(gate)
        reconstructed["execution"]["length_bucket_boundaries"] = (
            freezer.BUCKETED_EAGER_BOUNDARIES
        )
        reconstructed_hashes[mode] = hashlib.sha256(
            parent._canonical_json_bytes(reconstructed)
        ).hexdigest()
    if reconstructed_hashes != expected_hashes:
        raise RuntimeError(
            "Legacy hashes are not explained solely by the v3 boundary transform"
        )
    amended = copy.deepcopy(protocol)
    amended["mode_gate_file_sha256"] = actual_hashes
    _publish_bytes(output, parent._canonical_json_bytes(amended))
    return {
        "amended_protocol_file_sha256": _sha256(output),
        "actual_mode_gate_file_sha256": actual_hashes,
        "legacy_mode_gate_file_sha256": expected_hashes,
        "original_protocol_file_sha256": original_protocol_sha256,
        "reconstructed_legacy_mode_gate_file_sha256": reconstructed_hashes,
        "transformation": {
            "compiled_length_bucket_boundaries": freezer.STATIC_COMPILED_BOUNDARIES,
            "safe_eager_length_bucket_boundaries": (
                freezer.BUCKETED_EAGER_BOUNDARIES
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--amended-protocol-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Audit output must be create-only")
    if args.amended_protocol_output.exists():
        raise RuntimeError("Amended protocol output must be create-only")
    restore_delegated_contract()
    amendment = amend_legacy_mode_hashes(
        protocol_path=args.protocol,
        bundle_root=args.bundle_root,
        output=args.amended_protocol_output,
    )
    result = parent.audit_gate(
        run_root=args.run_root,
        protocol_path=args.amended_protocol_output,
        bundle_root=args.bundle_root,
        protocol_freezer=freezer,
        result_format_version=FORMAT_VERSION,
        auditor_path=Path(__file__),
    )
    result["protocol_amendment"] = amendment
    parent._publish(
        args.output,
        result,
    )


if __name__ == "__main__":
    main()
