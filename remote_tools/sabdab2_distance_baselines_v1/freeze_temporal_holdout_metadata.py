#!/usr/bin/env python3
"""Freeze identity-only SAbDab2 entries strictly newer than a training snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


API_URL = "https://sabdab.opig.stats.ox.ac.uk/api/download/search-summary"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def fetch_summary(cutoff: str) -> bytes:
    payload = json.dumps(
        {"deposited_after": cutoff, "antigen_type": "PROTEIN"}
    ).encode()
    request = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", required=True, help="Strict YYYY-MM-DD cutoff")
    parser.add_argument("--existing-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    split_raw = arguments.existing_split.read_bytes()
    split_rows = list(csv.DictReader(io.StringIO(split_raw.decode())))
    existing_pdbs = {row["pdb_assembly_id"] for row in split_rows}

    response = fetch_summary(arguments.cutoff)
    rows = list(csv.DictReader(io.StringIO(response.decode())))
    strict_rows = [
        row for row in rows if row["date"].replace("/", "-") > arguments.cutoff
    ]
    if not strict_rows:
        raise ValueError("official query returned no strictly newer protein-antigen rows")

    duplicate_instances = sorted(
        instance
        for instance in {row["INSTANCE"] for row in strict_rows}
        if sum(row["INSTANCE"] == instance for row in strict_rows) != 1
    )
    if duplicate_instances:
        raise ValueError(f"duplicate antibody instances: {duplicate_instances}")

    overlap = sorted({row["PDB"] for row in strict_rows} & existing_pdbs)
    if overlap:
        raise ValueError(f"temporal PDB overlap with existing split: {overlap}")

    fields = (
        "INSTANCE",
        "PDB",
        "Hchain",
        "Lchain",
        "antigen_chain",
        "antigen_type",
        "date",
        "SABDABdepo_date",
        "SABDABupdate_date",
        "method",
        "resolution",
        "type",
    )
    records = [
        {field.lower(): row[field] for field in fields}
        for row in sorted(strict_rows, key=lambda item: item["INSTANCE"])
    ]
    manifest = {
        "format_version": "sabdab2_strict_temporal_holdout_identity_manifest_v1",
        "authority": {
            "api_url": API_URL,
            "query": {
                "deposited_after": arguments.cutoff,
                "antigen_type": "PROTEIN",
            },
            "api_response_sha256": sha256_bytes(response),
            "existing_split_sha256": sha256_bytes(split_raw),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "boundary": {
            "strict_deposition_date_after": arguments.cutoff,
            "identity_metadata_only": True,
            "structure_loaded": False,
            "contact_labels_loaded": False,
            "selection_performed": False,
            "one_shot_evaluation_allowed_only_after_full_development_freeze": True,
        },
        "audit": {
            "existing_split_rows": len(split_rows),
            "existing_unique_pdbs": len(existing_pdbs),
            "temporal_instances": len(records),
            "temporal_unique_pdbs": len({row["pdb"] for row in records}),
            "pdb_overlap_count": 0,
            "minimum_deposition_date": min(row["date"] for row in records),
            "maximum_deposition_date": max(row["date"] for row in records),
        },
        "records": records,
    }
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical(manifest))
    print(
        json.dumps(
            {
                "manifest_sha256": sha256_bytes(canonical(manifest)),
                **manifest["audit"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
