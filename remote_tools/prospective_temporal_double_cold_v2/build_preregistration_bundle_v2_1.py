#!/usr/bin/env python3
"""Build a deterministic compact preregistration bundle for public deposit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


BUNDLE_FILES = (
    "results/prospective_temporal_double_cold_v2/PROSPECTIVE_TEMPORAL_DOUBLE_COLD_PROTOCOL_V2.json",
    "results/prospective_temporal_double_cold_v2/PROSPECTIVE_TEMPORAL_DOUBLE_COLD_PROTOCOL_V2_RECEIPT.json",
    "results/prospective_temporal_double_cold_v2/PROSPECTIVE_TEMPORAL_METADATA_SCREEN_INITIAL_V2.json",
    "results/prospective_temporal_double_cold_v2/PROSPECTIVE_TEMPORAL_METADATA_SCREEN_INITIAL_V2_RECEIPT.json",
    "results/prospective_temporal_double_cold_v2/authority_v2_1/PROSPECTIVE_TEMPORAL_RELEASE_DATE_AMENDMENT_V2_1.json",
    "results/prospective_temporal_double_cold_v2/authority_v2_1/PROSPECTIVE_TEMPORAL_RELEASE_DATE_AMENDMENT_V2_1_RECEIPT.json",
    "results/prospective_temporal_double_cold_v2/authority_v2_1/RCSB_INITIAL_RELEASE_METADATA_V2_1.json.gz",
    "results/prospective_temporal_double_cold_v2/authority_v2_1/RCSB_INITIAL_RELEASE_METADATA_V2_1_RECEIPT.json",
    "results/prospective_temporal_double_cold_v2/authority_v2_1/PROSPECTIVE_TEMPORAL_METADATA_SCREEN_V2_1.json",
    "results/prospective_temporal_double_cold_v2/authority_v2_1/PROSPECTIVE_TEMPORAL_METADATA_SCREEN_V2_1_RECEIPT.json",
    "results/prospective_temporal_double_cold_v2/design_audit/TEMPORAL_GATE_PRECISION_AUDIT_V2.json",
    "results/prospective_temporal_double_cold_v2/design_audit/TEMPORAL_GATE_PRECISION_AUDIT_V2_RECEIPT.json",
    "remote_tools/prospective_temporal_double_cold_v2/README.md",
    "remote_tools/prospective_temporal_double_cold_v2/THIRD_PARTY_SEALED_EXECUTION_SPEC_V2.md",
    "remote_tools/prospective_temporal_double_cold_v2/PREREGISTRATION_PUBLICATION_GUIDE_V2_1.md",
    "remote_tools/prospective_temporal_double_cold_v2/freeze_prospective_temporal_double_cold_v2.py",
    "remote_tools/prospective_temporal_double_cold_v2/screen_sabdab2_release_metadata_v2.py",
    "remote_tools/prospective_temporal_double_cold_v2/fetch_rcsb_release_metadata_v2_1.py",
    "remote_tools/prospective_temporal_double_cold_v2/freeze_release_date_amendment_v2_1.py",
    "remote_tools/prospective_temporal_double_cold_v2/screen_sabdab2_rcsb_release_metadata_v2_1.py",
    "remote_tools/prospective_temporal_double_cold_v2/audit_temporal_gate_precision_v2.py",
    "tests/test_prospective_temporal_double_cold_v2.py",
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publish(path: Path, payload: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return sha256_bytes(payload)


def zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(2026, 8, 30, 20, 37, 46))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100444 << 16
    info.create_system = 3
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    files: list[tuple[str, bytes]] = []
    manifest_entries = []
    for relative in BUNDLE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bundle input is missing or unsafe: {relative}")
        payload = path.read_bytes()
        files.append((relative, payload))
        manifest_entries.append(
            {"path": relative, "size_bytes": len(payload), "sha256": sha256_bytes(payload)}
        )

    manifest = {
        "format_version": "framecontact_temporal_preregistration_bundle_manifest_v2_1",
        "status": "READY_FOR_IMMUTABLE_EXTERNAL_REGISTRATION",
        "parent_protocol_sha256": (
            "41cad95072a3189826877dfbbb6c0ff493c9fd37c66a0132e5ae3fea9579ee7e"
        ),
        "release_date_amendment_sha256": (
            "e1d8dad751c213a93e7ebabb5da38d3177deea92331ab6011edb8ac090e82398"
        ),
        "future_temporal_labels_read": False,
        "opening_count": 0,
        "files": manifest_entries,
    }
    manifest_payload = canonical_bytes(manifest)
    files.append(("PREREGISTRATION_BUNDLE_MANIFEST_V2_1.json", manifest_payload))

    args.output_root.mkdir(parents=True, exist_ok=False)
    zip_path = args.output_root / "framecontact_temporal_preregistration_v2_1.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(files):
            archive.writestr(zip_info(name), payload)
    zip_payload = zip_path.read_bytes()
    zip_sha = sha256_bytes(zip_payload)
    manifest_sha = publish(
        args.output_root / "PREREGISTRATION_BUNDLE_MANIFEST_V2_1.json",
        manifest_payload,
    )
    receipt = {
        "format_version": "framecontact_temporal_preregistration_bundle_receipt_v2_1",
        "status": manifest["status"],
        "archive_sha256": zip_sha,
        "archive_size_bytes": len(zip_payload),
        "manifest_sha256": manifest_sha,
        "future_temporal_labels_read": False,
        "opening_count": 0,
        "external_registration_completed": False,
    }
    receipt_sha = publish(
        args.output_root / "PREREGISTRATION_BUNDLE_RECEIPT_V2_1.json",
        canonical_bytes(receipt),
    )
    print(
        json.dumps(
            {
                "archive_sha256": zip_sha,
                "manifest_sha256": manifest_sha,
                "receipt_sha256": receipt_sha,
            }
        )
    )


if __name__ == "__main__":
    main()
