#!/usr/bin/env python3
"""Freeze exact RCSB PDB initial-release dates for a SAbDab2 summary.

The SAbDab2 ``date`` column is an initial-deposition date, not an initial
release date.  This metadata-only fetcher therefore binds every distinct PDB
identifier in the supplied SAbDab2 summary to
``rcsb_accession_info.initial_release_date`` from the official RCSB Data API.
It never requests coordinates, sequences, pair distances, or labels.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"


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


def rcsb_query_id(sabdab_pdb_id: str) -> str:
    """Convert SAbDab2's canonical PDB spelling to an RCSB query ID."""

    value = sabdab_pdb_id.strip().upper()
    if not value:
        raise ValueError("empty SAbDab2 PDB identifier")
    if value.startswith("PDB_"):
        body = value[4:]
        if len(body) != 8 or not body.isalnum():
            raise ValueError(f"unsupported canonical PDB identifier: {sabdab_pdb_id!r}")
        if body.startswith("0000"):
            return body[-4:]
        return value
    if len(value) == 4 and value.isalnum():
        return value
    raise ValueError(f"unsupported PDB identifier: {sabdab_pdb_id!r}")


def parse_initial_release_date(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"RCSB release timestamp lacks timezone: {value!r}")
    return parsed.date().isoformat()


def fetch_batch(entry_ids: list[str], attempts: int = 4) -> list[dict[str, Any]]:
    query = (
        "query($ids:[String!]!){entries(entry_ids:$ids){"
        "rcsb_id rcsb_accession_info{initial_release_date}}}"
    )
    payload = json.dumps(
        {"query": query, "variables": {"ids": entry_ids}}, separators=(",", ":")
    ).encode("ascii")
    request = Request(
        RCSB_GRAPHQL_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "FrameContact/1.0"},
        method="POST",
    )
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
            if result.get("errors"):
                raise ValueError(f"RCSB GraphQL errors: {result['errors']}")
            entries = result.get("data", {}).get("entries")
            if not isinstance(entries, list):
                raise ValueError("RCSB GraphQL response schema differs")
            return entries
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise RuntimeError("RCSB metadata request failed") from exc
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if args.metadata_csv.is_symlink() or not args.metadata_csv.is_file():
        raise ValueError("metadata CSV is missing or unsafe")
    if sha256_file(args.metadata_csv) != args.metadata_sha256:
        raise ValueError("metadata CSV SHA-256 differs")
    if args.batch_size < 1 or args.batch_size > 250:
        raise ValueError("batch size must be between 1 and 250")

    sabdab_ids: set[str] = set()
    with args.metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "PDB" not in reader.fieldnames:
            raise ValueError("SAbDab2 summary schema differs")
        for row in reader:
            sabdab_ids.add(row["PDB"].strip().lower())
    if not sabdab_ids:
        raise ValueError("SAbDab2 summary is empty")

    query_to_sabdab: dict[str, str] = {}
    for sabdab_id in sorted(sabdab_ids):
        query_id = rcsb_query_id(sabdab_id)
        prior = query_to_sabdab.setdefault(query_id, sabdab_id)
        if prior != sabdab_id:
            raise ValueError(f"non-unique RCSB identifier mapping: {query_id}")

    observed: dict[str, str] = {}
    query_ids = sorted(query_to_sabdab)
    for start in range(0, len(query_ids), args.batch_size):
        batch = query_ids[start : start + args.batch_size]
        for entry in fetch_batch(batch):
            query_id = str(entry["rcsb_id"]).upper()
            release_value = entry.get("rcsb_accession_info", {}).get(
                "initial_release_date"
            )
            if not release_value:
                raise ValueError(f"RCSB initial release date missing: {query_id}")
            if query_id in observed:
                raise ValueError(f"duplicate RCSB entry: {query_id}")
            observed[query_id] = parse_initial_release_date(str(release_value))

    missing = sorted(set(query_ids) - set(observed))
    unexpected = sorted(set(observed) - set(query_ids))
    if missing or unexpected:
        raise ValueError(
            f"RCSB entry coverage differs: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    snapshot = {
        "format_version": "framecontact_rcsb_initial_release_metadata_v2_1",
        "status": "PASS_EXACT_RCSB_INITIAL_RELEASE_METADATA",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "authority": "RCSB PDB Data API",
            "graphql_url": RCSB_GRAPHQL_URL,
            "field": "rcsb_accession_info.initial_release_date",
            "coordinates_requested": False,
            "sequences_requested": False,
        },
        "sabdab_metadata_sha256": args.metadata_sha256,
        "counts": {
            "unique_sabdab_pdb_ids": len(sabdab_ids),
            "unique_rcsb_query_ids": len(query_ids),
            "resolved_rcsb_entries": len(observed),
        },
        "records": [
            {
                "sabdab_pdb_id": query_to_sabdab[query_id],
                "rcsb_query_id": query_id,
                "initial_release_date": observed[query_id],
            }
            for query_id in query_ids
        ],
        "audit": {
            "metadata_only": True,
            "bound_coordinates_read": False,
            "task_labels_read": False,
            "cross_chain_distances_computed": False,
            "opening_count": 0,
        },
    }
    snapshot_path = args.output_root / "RCSB_INITIAL_RELEASE_METADATA_V2_1.json"
    snapshot_sha = publish(snapshot_path, snapshot)
    receipt = {
        "format_version": "framecontact_rcsb_initial_release_metadata_receipt_v2_1",
        "status": snapshot["status"],
        "snapshot_sha256": snapshot_sha,
        "sabdab_metadata_sha256": args.metadata_sha256,
        "resolved_rcsb_entries": len(observed),
        "bound_coordinates_read": False,
        "task_labels_read": False,
        "opening_count": 0,
    }
    receipt_sha = publish(
        args.output_root / "RCSB_INITIAL_RELEASE_METADATA_V2_1_RECEIPT.json", receipt
    )
    print(
        json.dumps(
            {"snapshot_sha256": snapshot_sha, "receipt_sha256": receipt_sha}
        )
    )


if __name__ == "__main__":
    main()
