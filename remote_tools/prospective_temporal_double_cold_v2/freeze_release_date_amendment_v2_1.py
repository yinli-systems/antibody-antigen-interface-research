#!/usr/bin/env python3
"""Freeze a non-scientific v2.1 authority correction for release dates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PARENT_PROTOCOL_SHA256 = (
    "41cad95072a3189826877dfbbb6c0ff493c9fd37c66a0132e5ae3fea9579ee7e"
)
INITIAL_SCREEN_SHA256 = (
    "c04e85b1a894876d84ccf62d904b1e975e91e4c5c24d8b04c883db6b2ec662a5"
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--initial-screen", type=Path, required=True)
    parser.add_argument("--fetcher", type=Path, required=True)
    parser.add_argument("--revised-screen", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if sha256_file(args.parent_protocol) != PARENT_PROTOCOL_SHA256:
        raise ValueError("parent protocol SHA-256 differs")
    if sha256_file(args.initial_screen) != INITIAL_SCREEN_SHA256:
        raise ValueError("initial screen SHA-256 differs")
    for path in (args.fetcher, args.revised_screen):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"implementation authority missing or unsafe: {path}")

    initial = json.loads(args.initial_screen.read_text())
    if (
        initial.get("counts", {}).get("post_cutoff_instances_before_metadata_prefilter")
        != 0
        or initial.get("audit", {}).get("task_labels_read") is not False
        or initial.get("audit", {}).get("opening_count") != 0
    ):
        raise ValueError("initial screen evidence boundary differs")

    amendment = {
        "format_version": "framecontact_temporal_release_date_amendment_v2_1",
        "status": "FROZEN_AUTHORITY_SEMANTIC_CORRECTION_BEFORE_FIRST_CANDIDATE",
        "amended_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "superseded_initial_screen_sha256": INITIAL_SCREEN_SHA256,
        "reason": (
            "SAbDab2 documents its summary 'date' field as the date of initial "
            "PDB deposition. The frozen protocol requires PDB initial release date."
        ),
        "correction": {
            "authoritative_release_field": (
                "RCSB rcsb_accession_info.initial_release_date"
            ),
            "sabdab_date_field_used_as_release_authority": False,
            "sabdab_summary_role": (
                "antibody-instance annotation, antigen type, method, and resolution"
            ),
            "rcsb_data_api_role": "exact initial release date",
            "old_v2_screen_retained_for_audit": True,
        },
        "scientific_protocol_changes": {
            "cutoff_changed": False,
            "candidate_definition_changed": False,
            "double_cold_thresholds_changed": False,
            "opening_minima_changed": False,
            "model_or_checkpoints_changed": False,
            "task_routing_changed": False,
            "labels_or_metrics_changed": False,
        },
        "evidence_boundary_at_amendment": {
            "post_cutoff_candidates_seen": 0,
            "future_temporal_labels_read": False,
            "bound_coordinates_read": False,
            "opening_count": 0,
            "selection_or_tuning": False,
        },
        "implementation_authority": {
            "fetcher_sha256": sha256_file(args.fetcher),
            "revised_screen_sha256": sha256_file(args.revised_screen),
            "freezer_sha256": sha256_file(Path(__file__)),
        },
    }
    amendment_path = (
        args.output_root / "PROSPECTIVE_TEMPORAL_RELEASE_DATE_AMENDMENT_V2_1.json"
    )
    amendment_sha = publish(amendment_path, amendment)
    receipt = {
        "format_version": "framecontact_temporal_release_date_amendment_receipt_v2_1",
        "status": amendment["status"],
        "amendment_sha256": amendment_sha,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "post_cutoff_candidates_seen": 0,
        "future_temporal_labels_read": False,
        "opening_count": 0,
    }
    receipt_sha = publish(
        args.output_root
        / "PROSPECTIVE_TEMPORAL_RELEASE_DATE_AMENDMENT_V2_1_RECEIPT.json",
        receipt,
    )
    print(
        json.dumps(
            {"amendment_sha256": amendment_sha, "receipt_sha256": receipt_sha}
        )
    )


if __name__ == "__main__":
    main()
