#!/usr/bin/env python3
"""Materialize one deterministic shard of B32K PINDER-Train contact labels."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile

import materialize_pinder_contact_labels_v1 as base


STAGE = "B32K"
SYSTEMS = 32_768
SHARDS = 32


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if len(rows) != SYSTEMS or len({row["id"] for row in rows}) != SYSTEMS:
        raise RuntimeError("B32K curriculum count or uniqueness differs")
    if any(row["split"] != "train" for row in rows):
        raise RuntimeError("B32K curriculum contains a non-train row")
    return fields, rows


def select_shard(rows: list[dict[str, str]], index: int, count: int) -> list[dict[str, str]]:
    if count != SHARDS or not 0 <= index < count:
        raise ValueError("B32K shard geometry differs")
    selected = rows[index::count]
    if len(selected) != SYSTEMS // SHARDS:
        raise RuntimeError("B32K shard size differs")
    return selected


def _ids_sha256(rows: list[dict[str, str]]) -> str:
    payload = "".join(f"{row['id']}\n" for row in rows).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _authenticate(arguments: argparse.Namespace) -> dict[str, object]:
    authenticated = (
        (arguments.curriculum, arguments.expected_curriculum_sha256),
        (arguments.curriculum_receipt, arguments.expected_curriculum_receipt_sha256),
        (
            arguments.pdb_materialization_receipt,
            arguments.expected_pdb_materialization_receipt_sha256,
        ),
        (arguments.index_parquet, arguments.expected_index_sha256),
        (arguments.duckdb, arguments.expected_duckdb_sha256),
    )
    for path, expected in authenticated:
        if path.is_symlink() or not path.is_file() or base.sha256_file(path) != expected:
            raise RuntimeError(f"Authenticated input differs: {path}")
    if arguments.pdb_archive.is_symlink() or not arguments.pdb_archive.is_file():
        raise RuntimeError("PINDER structure archive is absent")
    materialization = json.loads(
        arguments.pdb_materialization_receipt.read_text(encoding="utf-8")
    )
    if (
        materialization.get("status") != "MATERIALIZED"
        or materialization.get("archive", {}).get("sha256")
        != "125bc49d515803576c19123afdc72f4f59e84d64522866f99d089c53b7d27912"
        or materialization.get("archive", {}).get("size_bytes")
        != arguments.pdb_archive.stat().st_size
        or materialization.get("test_access", {}).get("pinder_test_rows_used") is not False
        or materialization.get("test_access", {}).get("pinder_test_structure_bytes_accessed")
        is not False
    ):
        raise RuntimeError("PINDER structure materialization authority differs")
    curriculum_receipt = json.loads(
        arguments.curriculum_receipt.read_text(encoding="utf-8")
    )
    if (
        curriculum_receipt.get("status")
        != "PASS_TRAIN_ONLY_NESTED_CLUSTER_BALANCED_CURRICULUM"
        or curriculum_receipt.get("checks", {}).get("all_rows_train") is not True
        or curriculum_receipt.get("checks", {}).get("test_rows_read") is not False
        or curriculum_receipt.get("stages", {}).get(STAGE, {}).get("sha256")
        != arguments.expected_curriculum_sha256
    ):
        raise RuntimeError("B32K curriculum authority differs")
    return materialization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--expected-curriculum-sha256", required=True)
    parser.add_argument("--curriculum-receipt", type=Path, required=True)
    parser.add_argument("--expected-curriculum-receipt-sha256", required=True)
    parser.add_argument("--pdb-archive", type=Path, required=True)
    parser.add_argument("--pdb-materialization-receipt", type=Path, required=True)
    parser.add_argument("--expected-pdb-materialization-receipt-sha256", required=True)
    parser.add_argument("--index-parquet", type=Path, required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--expected-duckdb-sha256", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=SHARDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--maximum-residues", type=int, default=1020)
    arguments = parser.parse_args()
    if arguments.maximum_residues != 1020:
        raise ValueError("B32K token budget differs")
    for output in (arguments.output, arguments.receipt):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite output: {output}")
    materialization = _authenticate(arguments)
    fields, rows = _rows(arguments.curriculum)
    selected = select_shard(rows, arguments.shard_index, arguments.shard_count)

    with tempfile.TemporaryDirectory(prefix="pinder-b32k-shard-", dir="/dev/shm") as root:
        temporary = Path(root)
        shard_curriculum = temporary / "shard.tsv.gz"
        with shard_curriculum.open("wb") as compressed:
            with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=fields, delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(selected)
        clusters = base._load_endpoint_clusters(
            arguments.duckdb,
            shard_curriculum,
            arguments.index_parquet,
            temporary,
        )
        payload, counts = base.materialize(
            shard_curriculum,
            arguments.pdb_archive,
            clusters,
            arguments.maximum_residues,
        )
    base._publish_create_only(arguments.output, payload)
    receipt = {
        "authorization": {"B32K_merge": False, "B32K_training": False},
        "checks": {
            "all_rows_train": True,
            "bound_coordinates_published_as_model_input": False,
            "contact_indices_match_published_resolved_sequence_axes": True,
            "pinder_test_rows_read": False,
            "pinder_test_structures_read": False,
        },
        "counts": counts,
        "format_version": "abpairfm_pinder_contact_label_shard_v1",
        "ids_sha256": _ids_sha256(selected),
        "inputs": {
            "curriculum_receipt_sha256": arguments.expected_curriculum_receipt_sha256,
            "curriculum_sha256": arguments.expected_curriculum_sha256,
            "duckdb_sha256": arguments.expected_duckdb_sha256,
            "index_sha256": arguments.expected_index_sha256,
            "pdb_archive_sha256": materialization["archive"]["sha256"],
            "pdb_materialization_receipt_sha256": arguments.expected_pdb_materialization_receipt_sha256,
        },
        "output": {
            "bytes": arguments.output.stat().st_size,
            "path": arguments.output.name,
            "sha256": base.sha256_file(arguments.output),
        },
        "shard_count": arguments.shard_count,
        "shard_index": arguments.shard_index,
        "source_sha256": {
            "base_materializer": base.sha256_file(Path(base.__file__)),
            "shard_materializer": base.sha256_file(Path(__file__)),
        },
        "status": "PASS_B32K_PAIR_LABEL_SHARD",
        "test_access": False,
    }
    serialized = (
        json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")
    base._publish_create_only(arguments.receipt, serialized)
    print(serialized.decode("ascii"), end="")


if __name__ == "__main__":
    main()
