#!/usr/bin/env python3
"""Screen SAbDab2 instances using exact RCSB initial-release dates."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PARENT_PROTOCOL_SHA256 = (
    "41cad95072a3189826877dfbbb6c0ff493c9fd37c66a0132e5ae3fea9579ee7e"
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


def parse_resolution(value: str) -> float | None:
    text = value.strip()
    if not text or text.upper() in {"NA", "NONE", "NAN"}:
        return None
    result = float(text)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"invalid resolution: {value!r}")
    return result


def has_amino_acid_antigen(value: str) -> bool:
    kinds = {part.strip().upper() for part in value.split("|") if part.strip()}
    return bool(kinds.intersection({"PROTEIN", "PEPTIDE"}))


def load_release_dates(path: Path, expected_sha256: str) -> tuple[dict[str, str], dict]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("RCSB release metadata authority differs")
    snapshot = json.loads(path.read_text())
    if (
        snapshot.get("status") != "PASS_EXACT_RCSB_INITIAL_RELEASE_METADATA"
        or snapshot.get("audit", {}).get("bound_coordinates_read") is not False
        or snapshot.get("audit", {}).get("task_labels_read") is not False
    ):
        raise ValueError("RCSB release metadata evidence boundary differs")
    mapping: dict[str, str] = {}
    for record in snapshot["records"]:
        key = record["sabdab_pdb_id"].strip().lower()
        if key in mapping:
            raise ValueError(f"duplicate release authority: {key}")
        mapping[key] = record["initial_release_date"]
    if len(mapping) != snapshot["counts"]["unique_sabdab_pdb_ids"]:
        raise ValueError("RCSB release metadata count differs")
    return mapping, snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--rcsb-release-metadata", type=Path, required=True)
    parser.add_argument("--rcsb-release-metadata-sha256", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-etag", default="")
    parser.add_argument("--source-last-modified", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if sha256_file(args.protocol) != PARENT_PROTOCOL_SHA256:
        raise ValueError("parent protocol SHA-256 differs")
    if args.amendment.is_symlink() or sha256_file(args.amendment) != args.amendment_sha256:
        raise ValueError("release-date amendment SHA-256 differs")
    amendment = json.loads(args.amendment.read_text())
    if (
        amendment.get("status")
        != "FROZEN_AUTHORITY_SEMANTIC_CORRECTION_BEFORE_FIRST_CANDIDATE"
        or amendment.get("parent_protocol_sha256") != PARENT_PROTOCOL_SHA256
        or amendment.get("evidence_boundary_at_amendment", {}).get("opening_count")
        != 0
    ):
        raise ValueError("release-date amendment authority differs")
    if args.metadata_csv.is_symlink() or not args.metadata_csv.is_file():
        raise ValueError("metadata CSV is missing or unsafe")
    if sha256_file(args.metadata_csv) != args.metadata_sha256:
        raise ValueError("metadata CSV SHA-256 differs")

    protocol = json.loads(args.protocol.read_text())
    release_dates, release_snapshot = load_release_dates(
        args.rcsb_release_metadata, args.rcsb_release_metadata_sha256
    )
    cutoff = date.fromisoformat(protocol["release_boundary"]["cutoff_date_utc"])
    maximum_resolution = float(
        protocol["candidate_definition"]["resolution_angstrom_maximum"]
    )
    total = 0
    seen_pdb_ids: set[str] = set()
    post_cutoff: list[dict[str, Any]] = []
    prefiltered: list[dict[str, Any]] = []
    with args.metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "INSTANCE",
            "PDB",
            "Hchain",
            "Lchain",
            "antigen_chain",
            "antigen_type",
            "date",
            "method",
            "resolution",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("SAbDab2 summary schema differs")
        for row in reader:
            total += 1
            pdb_id = row["PDB"].strip().lower()
            seen_pdb_ids.add(pdb_id)
            release_text = release_dates.get(pdb_id)
            if release_text is None:
                raise ValueError(f"RCSB release authority missing: {pdb_id}")
            released = date.fromisoformat(release_text)
            if released <= cutoff:
                continue
            resolution = parse_resolution(row["resolution"])
            record = {
                "instance_id": row["INSTANCE"],
                "pdb_id": pdb_id,
                "initial_release_date": released.isoformat(),
                "sabdab_initial_deposition_date": row["date"].strip(),
                "method": row["method"],
                "resolution_angstrom": resolution,
                "heavy_chain": row["Hchain"],
                "light_chain": row["Lchain"],
                "antigen_chains": row["antigen_chain"],
                "antigen_types": row["antigen_type"],
            }
            post_cutoff.append(record)
            if (
                resolution is not None
                and resolution <= maximum_resolution
                and has_amino_acid_antigen(row["antigen_type"])
            ):
                prefiltered.append(record)
    if seen_pdb_ids != set(release_dates):
        raise ValueError("SAbDab2 and RCSB release-authority PDB sets differ")

    order = lambda row: (
        row["initial_release_date"],
        row["pdb_id"],
        row["instance_id"],
    )
    post_cutoff.sort(key=order)
    prefiltered.sort(key=order)
    status = (
        "WAITING_FOR_POST_CUTOFF_RELEASES"
        if not prefiltered
        else "PENDING_SEALED_SEQUENCE_DOUBLE_COLD_QUALIFICATION"
    )
    snapshot = {
        "format_version": "framecontact_prospective_temporal_metadata_screen_v2_1",
        "status": status,
        "screened_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": PARENT_PROTOCOL_SHA256,
        "release_date_amendment_sha256": args.amendment_sha256,
        "source": {
            "sabdab_url": args.source_url,
            "sabdab_etag": args.source_etag,
            "sabdab_last_modified": args.source_last_modified,
            "sabdab_metadata_sha256": args.metadata_sha256,
            "rcsb_release_metadata_sha256": args.rcsb_release_metadata_sha256,
            "rcsb_release_field": "rcsb_accession_info.initial_release_date",
        },
        "cutoff_date_utc": cutoff.isoformat(),
        "counts": {
            "metadata_rows": total,
            "unique_pdb_ids_with_exact_rcsb_release_date": len(release_dates),
            "post_cutoff_instances_before_metadata_prefilter": len(post_cutoff),
            "post_cutoff_unique_pdb_ids_before_metadata_prefilter": len(
                {row["pdb_id"] for row in post_cutoff}
            ),
            "metadata_prefiltered_instances": len(prefiltered),
            "metadata_prefiltered_unique_pdb_ids": len(
                {row["pdb_id"] for row in prefiltered}
            ),
        },
        "metadata_prefiltered_candidates": prefiltered,
        "gate": {
            "minimum_eligible_instances": protocol["label_free_gate"][
                "minimum_eligible_instances"
            ],
            "minimum_unique_pdb_ids": protocol["label_free_gate"][
                "minimum_unique_pdb_ids"
            ],
            "sequence_double_cold_qualification_complete": False,
            "opening_authorized": False,
        },
        "audit": {
            "sabdab_date_interpreted_as_initial_deposition_only": True,
            "rcsb_initial_release_date_is_authoritative": True,
            "metadata_only": True,
            "bound_coordinates_read": False,
            "cross_chain_distances_computed": False,
            "task_labels_read": False,
            "opening_count": 0,
            "selection_or_tuning": False,
            "old_sabdab2_test_accessed": False,
            "sable_labels_accessed": False,
            "rcsb_fetch_audit": release_snapshot["audit"],
        },
    }
    snapshot_path = args.output_root / "PROSPECTIVE_TEMPORAL_METADATA_SCREEN_V2_1.json"
    snapshot_sha = publish(snapshot_path, snapshot)
    receipt = {
        "format_version": "framecontact_prospective_temporal_metadata_screen_receipt_v2_1",
        "status": status,
        "snapshot_sha256": snapshot_sha,
        "protocol_sha256": PARENT_PROTOCOL_SHA256,
        "release_date_amendment_sha256": args.amendment_sha256,
        "sabdab_metadata_sha256": args.metadata_sha256,
        "rcsb_release_metadata_sha256": args.rcsb_release_metadata_sha256,
        "opening_authorized": False,
        "task_labels_read": False,
        "bound_coordinates_read": False,
    }
    receipt_sha = publish(
        args.output_root / "PROSPECTIVE_TEMPORAL_METADATA_SCREEN_V2_1_RECEIPT.json",
        receipt,
    )
    print(
        json.dumps(
            {"status": status, "snapshot_sha256": snapshot_sha, "receipt_sha256": receipt_sha}
        )
    )


if __name__ == "__main__":
    main()
