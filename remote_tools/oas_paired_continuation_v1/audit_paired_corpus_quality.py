#!/usr/bin/env python3
"""Audit paired OAS CDR mappings and sequence boundaries without training."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterable

from paired_pretraining import PairedPretrainingError, encode_pair


HEADER = (
    "heavy_sequence",
    "light_sequence",
    "cdr1_aa_heavy",
    "cdr2_aa_heavy",
    "cdr3_aa_heavy",
    "cdr1_aa_light",
    "cdr2_aa_light",
    "cdr3_aa_light",
    "study",
    "run",
    "species",
    "donor_hash",
    "row_hash",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(histogram: Counter[int], fraction: float) -> int:
    total = sum(histogram.values())
    target = max(1, int(total * fraction + 0.999999999))
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= target:
            return value
    raise RuntimeError("Empty histogram")


def audit_rows(
    rows: Iterable[list[str]],
    *,
    connection: sqlite3.Connection,
    expected_rows: int,
) -> dict[str, object]:
    connection.execute("CREATE TABLE row_hashes(value BLOB PRIMARY KEY) WITHOUT ROWID")
    token_lengths: Counter[int] = Counter()
    heavy_lengths: Counter[int] = Counter()
    light_lengths: Counter[int] = Counter()
    heavy_cdr3_lengths: Counter[int] = Counter()
    light_cdr3_lengths: Counter[int] = Counter()
    studies: Counter[str] = Counter()
    species: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    unknown_residue_pairs = 0
    min_background_probability = 1.0
    max_background_probability = 0.0
    row_count = 0
    hash_batch: list[tuple[bytes]] = []
    for row_count, row in enumerate(rows, start=1):
        if len(row) != len(HEADER):
            failures["column_count"] += 1
            continue
        values = dict(zip(HEADER, row, strict=True))
        try:
            row_hash = bytes.fromhex(values["row_hash"])
            if len(row_hash) != 32 or row_hash.hex() != values["row_hash"]:
                raise ValueError
        except ValueError:
            failures["row_hash"] += 1
            continue
        hash_batch.append((row_hash,))
        if len(hash_batch) >= 100_000:
            connection.executemany("INSERT OR IGNORE INTO row_hashes VALUES (?)", hash_batch)
            connection.commit()
            hash_batch.clear()
        try:
            encoded = encode_pair(
                values["heavy_sequence"],
                values["light_sequence"],
                (
                    values["cdr1_aa_heavy"],
                    values["cdr2_aa_heavy"],
                    values["cdr3_aa_heavy"],
                ),
                (
                    values["cdr1_aa_light"],
                    values["cdr2_aa_light"],
                    values["cdr3_aa_light"],
                ),
                maximum_tokens=320,
            )
        except PairedPretrainingError as error:
            failures[str(error)] += 1
            continue
        token_lengths[len(encoded.input_ids)] += 1
        heavy_lengths[encoded.heavy_length] += 1
        light_lengths[encoded.light_length] += 1
        heavy_cdr3_lengths[len(values["cdr3_aa_heavy"])] += 1
        light_cdr3_lengths[len(values["cdr3_aa_light"])] += 1
        studies[values["study"]] += 1
        species[values["species"]] += 1
        if "X" in values["heavy_sequence"] or "X" in values["light_sequence"]:
            unknown_residue_pairs += 1
        residue_count = encoded.heavy_length + encoded.light_length
        cdr3_count = len(values["cdr3_aa_heavy"]) + len(values["cdr3_aa_light"])
        background_probability = (
            0.15 * residue_count - 0.25 * cdr3_count
        ) / (residue_count - cdr3_count)
        if not 0.0 < background_probability < 1.0:
            failures["balanced_mask_probability"] += 1
        min_background_probability = min(min_background_probability, background_probability)
        max_background_probability = max(max_background_probability, background_probability)
    if hash_batch:
        connection.executemany("INSERT OR IGNORE INTO row_hashes VALUES (?)", hash_batch)
        connection.commit()
    unique_hashes = int(connection.execute("SELECT COUNT(*) FROM row_hashes").fetchone()[0])
    valid_rows = sum(token_lengths.values())
    duplicate_row_hashes = row_count - unique_hashes
    status = (
        "PASS_PAIRED_CORPUS_QUALITY_AUDIT"
        if row_count == expected_rows
        and valid_rows == expected_rows
        and unique_hashes == expected_rows
        and not failures
        else "FAIL_PAIRED_CORPUS_QUALITY_AUDIT"
    )
    return {
        "balanced_mask_background_probability": {
            "maximum": max_background_probability,
            "minimum": min_background_probability,
        },
        "counts": {
            "duplicate_row_hashes": duplicate_row_hashes,
            "expected_rows": expected_rows,
            "observed_rows": row_count,
            "unique_row_hashes": unique_hashes,
            "unknown_residue_pairs": unknown_residue_pairs,
            "valid_rows": valid_rows,
        },
        "failures": dict(sorted(failures.items())),
        "lengths": {
            label: {
                "maximum": max(histogram),
                "median": _quantile(histogram, 0.5),
                "p95": _quantile(histogram, 0.95),
                "p99": _quantile(histogram, 0.99),
            }
            for label, histogram in (
                ("heavy_residues", heavy_lengths),
                ("light_residues", light_lengths),
                ("paired_tokens", token_lengths),
                ("heavy_cdr3_residues", heavy_cdr3_lengths),
                ("light_cdr3_residues", light_cdr3_lengths),
            )
        },
        "species_counts": dict(sorted(species.items())),
        "status": status,
        "study_count": len(studies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pairs-sha256", required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--work-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.work_database.exists():
        raise FileExistsError("Refusing to overwrite audit state")
    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    expected_rows = receipt.get("counts", {}).get("unique_exact_pairs")
    if (
        receipt.get("format_version") != "abpairfm_oas_stage_b_candidates_v2"
        or receipt.get("test_access") is not False
        or receipt.get("output", {}).get("sha256") != args.pairs_sha256
        or not isinstance(expected_rows, int)
    ):
        raise RuntimeError("Source receipt differs")
    observed_sha256 = _sha256(args.pairs)
    if observed_sha256 != args.pairs_sha256:
        raise RuntimeError("Paired corpus hash differs")
    connection = sqlite3.connect(args.work_database)
    try:
        connection.executescript(
            "PRAGMA synchronous=OFF; PRAGMA journal_mode=OFF; PRAGMA temp_store=FILE; PRAGMA cache_size=-262144;"
        )
        with gzip.open(args.pairs, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            if tuple(next(reader)) != HEADER:
                raise RuntimeError("Paired corpus header differs")
            result = audit_rows(reader, connection=connection, expected_rows=expected_rows)
    finally:
        connection.close()
    payload = {
        "format_version": "oas_paired_corpus_quality_audit_v1",
        "inputs": {
            "pairs_sha256": observed_sha256,
            "source_receipt_sha256": _sha256(args.source_receipt),
        },
        **result,
        "test_access": False,
        "training_performed": False,
    }
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    args.work_database.unlink()
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    if result["status"] != "PASS_PAIRED_CORPUS_QUALITY_AUDIT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
