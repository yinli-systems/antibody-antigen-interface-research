#!/usr/bin/env python3
"""Freeze a static-194 four-GPU efficiency gate with a live corpus panel anchor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import freeze_antibody_four_gpu_efficiency_gate as parent


FORMAT_VERSION = "antibody_four_gpu_efficiency_gate_v3"
PANEL_COUNT = 8192
PANEL_SEED = 2026081501
PANEL_INDICES_SHA256 = (
    "ba24268e3897320aafb49c8b557d559a73386242e47b666d5fc862fdf453cd38"
)
STATIC_COMPILED_BOUNDARIES = "194"
BUCKETED_EAGER_BOUNDARIES = "96,128,160,194"
VALIDATION_PANEL = {
    "indices_sha256": PANEL_INDICES_SHA256,
    "mask_seed": 2026081503,
    "mask_update": 0,
    "record_count": PANEL_COUNT,
    "seed": PANEL_SEED,
    "split": "validation",
}
SOURCE_FILES = tuple(
    sorted(
        set(
            parent.SOURCE_FILES
            + (
                "scripts/audit_antibody_four_gpu_efficiency_gate_v3.py",
                "scripts/freeze_antibody_four_gpu_efficiency_gate_v3.py",
            )
        )
    )
)


def freeze_bundle(**kwargs: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Reuse the audited v1 bundle with the static-194 boundary contract.

    The shape probe selected the single-boundary ``194`` compiled candidate.
    The v1 parent freezer leaves the compiled gate at the default four
    boundaries, which makes the runner reject the otherwise correct v3 launch
    before any compiled work starts.  Keep eager execution bucketed and make
    the compiled gate agree with the selected candidate explicitly.
    """
    old_format = parent.FORMAT_VERSION
    old_panel = parent.VALIDATION_PANEL
    old_sources = parent.SOURCE_FILES
    parent.FORMAT_VERSION = FORMAT_VERSION
    parent.VALIDATION_PANEL = VALIDATION_PANEL
    parent.SOURCE_FILES = SOURCE_FILES
    try:
        master, gates = parent.freeze_bundle(**kwargs)
    finally:
        parent.FORMAT_VERSION = old_format
        parent.VALIDATION_PANEL = old_panel
        parent.SOURCE_FILES = old_sources
    gates["compiled"]["execution"]["length_bucket_boundaries"] = (
        STATIC_COMPILED_BOUNDARIES
    )
    gates["safe_eager"]["execution"]["length_bucket_boundaries"] = (
        BUCKETED_EAGER_BOUNDARIES
    )
    master["selected_candidate"] = {
        "compiled_length_bucket_boundaries": STATIC_COMPILED_BOUNDARIES,
        "name": "static_194",
    }
    master["claim_boundary"] = (
        "Four-GPU real-OAS static-194 A/B/B/A efficiency and "
        "checkpoint-equivalence gate; corrected validation panel anchor only; "
        "no production pretraining, downstream, official-test, biological, "
        "or SOTA claim"
    )
    return master, gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "corpus-root",
        "independent-audit",
        "recovery-audit",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "manifest-file-sha256",
        "manifest-payload-sha256",
        "corpus-protocol-file-sha256",
        "independent-audit-file-sha256",
        "independent-audit-payload-sha256",
        "recovery-audit-file-sha256",
        "recovery-audit-payload-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError("Protocol bundle output must be create-only")
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
    parent._publish(args.output_dir / "compiled_gate.json", gates["compiled"])
    parent._publish(args.output_dir / "safe_eager_gate.json", gates["safe_eager"])
    parent._publish(args.output_dir / "protocol.json", master)
    args.output_dir.chmod(0o555)


if __name__ == "__main__":
    main()
