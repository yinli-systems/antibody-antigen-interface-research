#!/usr/bin/env python3
"""Run a metadata-only weekly screen for prospective post-cutoff releases."""

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


PROTOCOL_FORMAT = "framecontact_prospective_temporal_double_cold_protocol_v2"


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


def parse_release_date(value: str) -> date | None:
    text = value.strip()
    if not text or text.upper() in {"NA", "NONE", "NAN"}:
        return None
    for pattern in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    raise ValueError(f"unsupported release date: {value!r}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-etag", default="")
    parser.add_argument("--source-last-modified", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if args.metadata_csv.is_symlink() or not args.metadata_csv.is_file():
        raise ValueError("metadata CSV is missing or unsafe")
    if sha256_file(args.metadata_csv) != args.metadata_sha256:
        raise ValueError("metadata CSV SHA-256 differs")
    receipt = json.loads(args.protocol_receipt.read_text())
    if sha256_file(args.protocol) != receipt.get("protocol_sha256"):
        raise ValueError("protocol receipt SHA-256 differs")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("status")
        != "PREREGISTERED_BEFORE_FIRST_POST_CUTOFF_CANDIDATE_SCREEN"
        or protocol.get("label_free_gate", {}).get("opening_before_gate") is not False
        or protocol.get("closed_evidence", {}).get(
            "sable_v0_2_0_may_be_reused_as_blind_test"
        )
        is not False
    ):
        raise ValueError("prospective protocol authority differs")

    cutoff = date.fromisoformat(protocol["release_boundary"]["cutoff_date_utc"])
    maximum_resolution = float(
        protocol["candidate_definition"]["resolution_angstrom_maximum"]
    )
    total = 0
    missing_release_date = 0
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
            released = parse_release_date(row["date"])
            if released is None:
                missing_release_date += 1
                continue
            if released <= cutoff:
                continue
            resolution = parse_resolution(row["resolution"])
            record = {
                "instance_id": row["INSTANCE"],
                "pdb_id": row["PDB"],
                "release_date": released.isoformat(),
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

    post_cutoff.sort(key=lambda row: (row["release_date"], row["pdb_id"], row["instance_id"]))
    prefiltered.sort(key=lambda row: (row["release_date"], row["pdb_id"], row["instance_id"]))
    status = (
        "WAITING_FOR_POST_CUTOFF_RELEASES"
        if not prefiltered
        else "PENDING_SEALED_SEQUENCE_DOUBLE_COLD_QUALIFICATION"
    )
    snapshot = {
        "format_version": "framecontact_prospective_temporal_metadata_screen_v2",
        "status": status,
        "screened_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": receipt["protocol_sha256"],
        "source": {
            "url": args.source_url,
            "etag": args.source_etag,
            "last_modified": args.source_last_modified,
            "metadata_sha256": args.metadata_sha256,
        },
        "cutoff_date_utc": cutoff.isoformat(),
        "counts": {
            "metadata_rows": total,
            "rows_missing_release_date": missing_release_date,
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
            "metadata_and_sequences_only": True,
            "bound_coordinates_read": False,
            "cross_chain_distances_computed": False,
            "task_labels_read": False,
            "opening_count": 0,
            "selection_or_tuning": False,
            "old_sabdab2_test_accessed": False,
            "sable_labels_accessed": False,
        },
    }
    snapshot_path = args.output_root / "PROSPECTIVE_TEMPORAL_METADATA_SCREEN_V2.json"
    snapshot_sha = publish(snapshot_path, snapshot)
    snapshot_receipt = {
        "format_version": "framecontact_prospective_temporal_metadata_screen_receipt_v2",
        "status": status,
        "snapshot_sha256": snapshot_sha,
        "protocol_sha256": receipt["protocol_sha256"],
        "metadata_sha256": args.metadata_sha256,
        "opening_authorized": False,
        "task_labels_read": False,
        "bound_coordinates_read": False,
    }
    receipt_sha = publish(
        args.output_root / "PROSPECTIVE_TEMPORAL_METADATA_SCREEN_V2_RECEIPT.json",
        snapshot_receipt,
    )
    print(json.dumps({"status": status, "snapshot_sha256": snapshot_sha, "receipt_sha256": receipt_sha}))


if __name__ == "__main__":
    main()
