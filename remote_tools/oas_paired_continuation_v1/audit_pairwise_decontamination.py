#!/usr/bin/env python3
"""Audit pair-level OAS/development overlap from frozen MMseqs hits.

The legacy filter excluded a VH/VL pair when either chain matched any
development chain.  Antibody frameworks make that rule degenerate.  This
auditor instead asks whether both chains of one OAS pair match the heavy and
light chains of the same development complex, across a frozen identity grid.
It never reads task labels or test data and never writes a filtered corpus.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Iterable


THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 1.00)
CHAINS = frozenset(("heavy", "light"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_hash(endpoint: str) -> tuple[bytes, str]:
    try:
        digest, chain = endpoint.rsplit("|", 1)
        raw = bytes.fromhex(digest)
    except (ValueError, TypeError) as error:
        raise ValueError("Malformed OAS query endpoint") from error
    if len(raw) != 32 or raw.hex() != digest or chain not in CHAINS:
        raise ValueError("Malformed OAS query endpoint")
    return raw, chain


def _target(endpoint: str) -> tuple[str, str]:
    fields = endpoint.split("|")
    if len(fields) != 4 or fields[2] not in CHAINS or not fields[3].startswith(
        "sha256="
    ):
        raise ValueError("Malformed development target endpoint")
    digest = fields[3][len("sha256=") :]
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("Malformed development target digest")
    return fields[1], fields[2]


def audit_hits(
    lines: Iterable[str],
    *,
    source_pair_count: int,
    connection: sqlite3.Connection | None = None,
    minimum_reciprocal_coverage: float = 0.80,
) -> dict[str, object]:
    """Aggregate non-contiguous heavy/light query blocks with bounded memory."""
    if source_pair_count <= 0:
        raise ValueError("source_pair_count must be positive")
    if not 0.0 < minimum_reciprocal_coverage <= 1.0:
        raise ValueError("minimum_reciprocal_coverage must be in (0, 1]")
    owns_connection = connection is None
    database = connection if connection is not None else sqlite3.connect(":memory:")
    database.executescript(
        """
        PRAGMA synchronous=OFF;
        PRAGMA journal_mode=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE pair_instance (
            row_hash BLOB NOT NULL,
            target_instance TEXT NOT NULL,
            heavy_identity REAL NOT NULL,
            light_identity REAL NOT NULL,
            PRIMARY KEY (row_hash, target_instance)
        ) WITHOUT ROWID;
        """
    )
    insert_sql = """
        INSERT INTO pair_instance(
            row_hash, target_instance, heavy_identity, light_identity
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT(row_hash, target_instance) DO UPDATE SET
            heavy_identity=MAX(heavy_identity, excluded.heavy_identity),
            light_identity=MAX(light_identity, excluded.light_identity)
    """
    batch: list[tuple[bytes, str, float, float]] = []
    hit_line_count = 0
    accepted_hit_count = 0
    target_chain_mismatch_count = 0
    below_coverage_count = 0

    for line_number, line in enumerate(lines, start=1):
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 8:
            raise ValueError(f"Malformed MMseqs hit at line {line_number}")
        row_hash, query_chain = _row_hash(fields[0])
        target_instance, target_chain = _target(fields[1])
        try:
            identity = float(fields[2])
            query_coverage = float(fields[4])
            target_coverage = float(fields[5])
        except ValueError as error:
            raise ValueError(f"Non-numeric MMseqs hit at line {line_number}") from error
        if not all(math.isfinite(value) for value in (identity, query_coverage, target_coverage)):
            raise ValueError(f"Non-finite MMseqs hit at line {line_number}")
        hit_line_count += 1
        if query_chain != target_chain:
            target_chain_mismatch_count += 1
            continue
        if (
            query_coverage + 1e-12 < minimum_reciprocal_coverage
            or target_coverage + 1e-12 < minimum_reciprocal_coverage
        ):
            below_coverage_count += 1
            continue
        accepted_hit_count += 1
        batch.append(
            (
                row_hash,
                target_instance,
                identity if query_chain == "heavy" else 0.0,
                identity if query_chain == "light" else 0.0,
            )
        )
        if len(batch) >= 50_000:
            database.executemany(insert_sql, batch)
            database.commit()
            batch.clear()
    if batch:
        database.executemany(insert_sql, batch)
        database.commit()

    database.execute(
        """
        CREATE TABLE pair_summary AS
        SELECT row_hash,
               MAX(heavy_identity) AS heavy_identity,
               MAX(light_identity) AS light_identity
        FROM pair_instance
        GROUP BY row_hash
        """
    )
    database.execute("CREATE UNIQUE INDEX pair_summary_row ON pair_summary(row_hash)")
    database.commit()
    rows_with_hits = int(database.execute("SELECT COUNT(*) FROM pair_summary").fetchone()[0])
    if rows_with_hits > source_pair_count:
        raise ValueError("Observed more OAS rows than the authenticated source")
    identity_grid: dict[str, object] = {}
    for threshold in THRESHOLDS:
        both_chain_hits = int(
            database.execute(
                "SELECT COUNT(*) FROM pair_summary WHERE heavy_identity >= ? AND light_identity >= ?",
                (threshold, threshold),
            ).fetchone()[0]
        )
        heavy_only = int(
            database.execute(
                "SELECT COUNT(*) FROM pair_summary WHERE heavy_identity >= ? AND light_identity < ?",
                (threshold, threshold),
            ).fetchone()[0]
        )
        light_only = int(
            database.execute(
                "SELECT COUNT(*) FROM pair_summary WHERE heavy_identity < ? AND light_identity >= ?",
                (threshold, threshold),
            ).fetchone()[0]
        )
        excluded = int(
            database.execute(
                "SELECT COUNT(DISTINCT row_hash) FROM pair_instance WHERE heavy_identity >= ? AND light_identity >= ?",
                (threshold, threshold),
            ).fetchone()[0]
        )
        identity_grid[f"{threshold:.2f}"] = {
            "both_chains_hit_any_targets": both_chain_hits,
            "excluded_same_development_instance": excluded,
            "excluded_fraction": excluded / source_pair_count,
            "heavy_only": heavy_only,
            "kept_pairs": source_pair_count - excluded,
            "light_only": light_only,
        }
    result = {
        "counts": {
            "accepted_chain_matched_reciprocal_coverage_hits": accepted_hit_count,
            "below_reciprocal_coverage_hits": below_coverage_count,
            "hit_lines": hit_line_count,
            "rows_with_any_hit": rows_with_hits,
            "source_pairs": source_pair_count,
            "target_chain_mismatch_hits": target_chain_mismatch_count,
        },
        "identity_grid": identity_grid,
    }
    composite_excluded = int(
        database.execute(
            """
            SELECT COUNT(*)
            FROM pair_summary
            WHERE heavy_identity >= 0.90
               OR light_identity >= 0.90
               OR row_hash IN (
                   SELECT row_hash
                   FROM pair_instance
                   WHERE heavy_identity >= 0.80 AND light_identity >= 0.80
               )
            """
        ).fetchone()[0]
    )
    result["frozen_composite_rule"] = {
        "excluded_pairs": composite_excluded,
        "excluded_fraction": composite_excluded / source_pair_count,
        "kept_pairs": source_pair_count - composite_excluded,
        "paired_same_instance_identity": 0.80,
        "single_chain_identity": 0.90,
    }
    if owns_connection:
        database.close()
    return result


def export_composite_exclusions(
    connection: sqlite3.Connection,
    output: Path,
) -> tuple[int, str]:
    """Write sorted row hashes selected by the frozen composite rule."""
    if output.exists():
        raise FileExistsError("Refusing to overwrite exclusion artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    count = 0
    digest = hashlib.sha256()
    with temporary.open("xb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle:
            cursor = connection.execute(
                """
                SELECT row_hash
                FROM pair_summary
                WHERE heavy_identity >= 0.90
                   OR light_identity >= 0.90
                   OR row_hash IN (
                       SELECT row_hash
                       FROM pair_instance
                       WHERE heavy_identity >= 0.80 AND light_identity >= 0.80
                   )
                ORDER BY row_hash
                """
            )
            for (row_hash,) in cursor:
                line = bytes(row_hash).hex().encode("ascii") + b"\n"
                handle.write(line)
                count += 1
    with temporary.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    os.replace(temporary, output)
    return count, digest.hexdigest()


def _publish(path: Path, payload: object) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "ascii"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--hits-sha256", required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--work-database", type=Path, required=True)
    parser.add_argument("--exclusions-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    source_pair_count = receipt.get("counts", {}).get("unique_exact_pairs")
    if (
        receipt.get("format_version") != "abpairfm_oas_stage_b_candidates_v2"
        or receipt.get("test_access") is not False
        or not isinstance(source_pair_count, int)
        or source_pair_count <= 0
    ):
        raise ValueError("Source receipt differs from the authenticated paired OAS corpus")
    observed_hits_sha256 = sha256_file(args.hits)
    if observed_hits_sha256 != args.hits_sha256:
        raise ValueError("Frozen MMseqs hit SHA-256 differs")
    if args.work_database.exists():
        raise FileExistsError("Refusing to reuse a work database")
    database = sqlite3.connect(args.work_database)
    try:
        database.execute("PRAGMA cache_size=-524288")
        with gzip.open(args.hits, "rt", encoding="utf-8", newline="") as handle:
            result = audit_hits(
                handle,
                source_pair_count=source_pair_count,
                connection=database,
            )
        exclusion_count, exclusion_sha256 = export_composite_exclusions(
            database,
            args.exclusions_output,
        )
    finally:
        database.close()
    if exclusion_count != result["frozen_composite_rule"]["excluded_pairs"]:
        raise RuntimeError("Exclusion artifact count differs from audit")
    payload = {
        "authorization": {
            "corpus_materialization": False,
            "development_only": True,
            "training": False,
        },
        "format_version": "oas_pairwise_decontamination_audit_v2",
        "inputs": {
            "hits_sha256": observed_hits_sha256,
            "source_receipt_sha256": sha256_file(args.source_receipt),
        },
        "exclusion_artifact": {
            "count": exclusion_count,
            "name": args.exclusions_output.name,
            "sha256": exclusion_sha256,
        },
        **result,
        "method": {
            "aggregation": "sqlite_pair_instance_max_identity_v1",
            "minimum_reciprocal_coverage": 0.80,
        },
        "status": "PASS_PAIRWISE_DECONTAMINATION_AUDIT_V2",
        "test_access": False,
        "task_labels_read": False,
    }
    _publish(args.output, payload)
    args.work_database.unlink()
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
