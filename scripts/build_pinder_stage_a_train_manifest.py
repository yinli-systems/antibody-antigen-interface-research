#!/usr/bin/env python3
"""Build an authenticated train-only PINDER Stage A system manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable


MANIFEST_COLUMNS = (
    "split",
    "id",
    "pdb_id",
    "cluster_id",
    "uniprot_R",
    "uniprot_L",
    "chain_R",
    "chain_L",
    "contains_antibody",
    "contains_antigen",
    "holo_R_pdb",
    "holo_L_pdb",
    "predicted_R_pdb",
    "predicted_L_pdb",
    "apo_R_pdb",
    "apo_L_pdb",
    "resolution",
    "length1",
    "length2",
    "length_resolved_1",
    "length_resolved_2",
    "num_atom_types",
    "intermolecular_contacts",
)
STRUCTURE_COLUMNS = (
    "holo_R_pdb",
    "holo_L_pdb",
    "predicted_R_pdb",
    "predicted_L_pdb",
    "apo_R_pdb",
    "apo_L_pdb",
)
QUALITY_PREDICATE = """
    i.split='train'
    AND m.resolution<=4.0
    AND m.intermolecular_contacts>=5
    AND m.num_atom_types>=4
    AND m.length_resolved_1>=20
    AND m.length_resolved_2>=20
    AND m.missing_interface_residues_4A=0
    AND m.interface_atom_gaps_4A=0
    AND m.length1+m.length2<=1020
""".strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _authenticate(path: Path, expected_sha256: str) -> Path:
    selected = path.resolve()
    if selected.is_symlink() or not selected.is_file():
        raise RuntimeError(f"Expected a regular file: {selected}")
    if _sha256_file(selected) != expected_sha256:
        raise RuntimeError(f"SHA-256 differs from frozen anchor: {selected}")
    return selected


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _manifest_sql(index: Path, metadata: Path, output: Path) -> str:
    projected = ",".join(
        f"i.{column}" if column in MANIFEST_COLUMNS[:16] else f"m.{column}"
        for column in MANIFEST_COLUMNS
    )
    return f"""
    COPY (
      SELECT {projected}
      FROM read_parquet('{_sql_path(index)}') i
      JOIN read_parquet('{_sql_path(metadata)}') m USING(id)
      WHERE {QUALITY_PREDICATE}
      ORDER BY i.id
    ) TO '{_sql_path(output)}'
    (FORMAT CSV, HEADER TRUE, DELIMITER '\t');
    """


def _write_uncompressed_manifest(
    duckdb: Path, index: Path, metadata: Path, output: Path
) -> None:
    result = subprocess.run(
        [str(duckdb), "-batch", "-c", _manifest_sql(index, metadata, output)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"DuckDB manifest export failed: {result.stderr.strip()}")


def _scan_manifest(
    path: Path,
) -> tuple[int, int, dict[str, int], set[str]]:
    row_count = 0
    clusters: set[str] = set()
    required_members: set[str] = set()
    view_counts = {
        "canonical_apo_any": 0,
        "canonical_apo_both": 0,
        "canonical_holo_both": 0,
        "predicted_any": 0,
        "predicted_both": 0,
    }
    previous_id: str | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise RuntimeError("Unexpected PINDER manifest header")
        for row in reader:
            row_count += 1
            if row["split"] != "train":
                raise RuntimeError("Non-train row reached the PINDER manifest")
            if previous_id is not None and row["id"] <= previous_id:
                raise RuntimeError("PINDER manifest IDs are not unique and sorted")
            previous_id = row["id"]
            clusters.add(row["cluster_id"])
            for column in STRUCTURE_COLUMNS:
                if row[column]:
                    required_members.add(f"pdbs/{row[column]}")
            holo = (bool(row["holo_R_pdb"]), bool(row["holo_L_pdb"]))
            apo = (bool(row["apo_R_pdb"]), bool(row["apo_L_pdb"]))
            predicted = (
                bool(row["predicted_R_pdb"]),
                bool(row["predicted_L_pdb"]),
            )
            view_counts["canonical_holo_both"] += int(all(holo))
            view_counts["canonical_apo_any"] += int(any(apo))
            view_counts["canonical_apo_both"] += int(all(apo))
            view_counts["predicted_any"] += int(any(predicted))
            view_counts["predicted_both"] += int(all(predicted))
    return row_count, len(clusters), view_counts, required_members


def _validate_archive_members(
    archive: Path, required_members: set[str], lines: Iterable[str] | None = None
) -> dict[str, int | bool]:
    missing = set(required_members)
    member_count = 0
    test_named_member = False
    process: subprocess.Popen[str] | None = None
    if lines is None:
        process = subprocess.Popen(
            ["unzip", "-Z1", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stdout is None:
            raise RuntimeError("Could not read the PINDER ZIP central directory")
        lines = process.stdout
    for raw in lines:
        member = raw.rstrip("\n")
        member_count += 1
        if "test" in member.lower():
            test_named_member = True
        missing.discard(member)
    if process is not None:
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"PINDER ZIP inventory failed: {stderr.strip()}")
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(
            f"{len(missing)} required structures are absent from the archive; sample={sample}"
        )
    if test_named_member:
        raise RuntimeError("The selected PINDER archive unexpectedly contains test-named members")
    return {
        "archive_member_count": member_count,
        "required_member_count": len(required_members),
        "required_members_present": True,
        "test_named_member_present": False,
    }


def _deterministic_gzip(source: Path, output: Path) -> None:
    with source.open("rb") as input_handle, output.open("xb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
            for block in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                compressed.write(block)
        binary.flush()
        os.fsync(binary.fileno())


def _publish_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite PINDER receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--expected-duckdb-sha256", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--expected-materialization-receipt-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-row-count", type=int, required=True)
    parser.add_argument("--expected-cluster-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    source = _authenticate(Path(__file__), arguments.expected_source_sha256)
    duckdb = _authenticate(arguments.duckdb, arguments.expected_duckdb_sha256)
    index = _authenticate(arguments.index, arguments.expected_index_sha256)
    metadata = _authenticate(arguments.metadata, arguments.expected_metadata_sha256)
    protocol = _authenticate(arguments.protocol, arguments.expected_protocol_sha256)
    materialization_receipt = _authenticate(
        arguments.materialization_receipt,
        arguments.expected_materialization_receipt_sha256,
    )
    archive = arguments.archive.resolve()
    materialization = json.loads(materialization_receipt.read_text(encoding="utf-8"))
    if materialization.get("status") != "MATERIALIZED":
        raise RuntimeError("PINDER archive materialization did not pass")
    if any(materialization.get("test_access", {}).values()):
        raise RuntimeError("PINDER materialization receipt reports test access")
    if archive.is_symlink() or not archive.is_file():
        raise RuntimeError(f"Expected a regular PINDER archive: {archive}")
    if archive.stat().st_size != materialization["archive"]["size_bytes"]:
        raise RuntimeError("PINDER archive size differs from its authenticated receipt")
    output = arguments.output.resolve()
    receipt = arguments.receipt.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or receipt.exists():
        raise FileExistsError("Refusing to overwrite PINDER manifest artifacts")

    descriptor, raw_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tsv", dir=output.parent
    )
    os.close(descriptor)
    raw = Path(raw_name)
    compressed = output.with_name(f".{output.name}.partial")
    try:
        raw.unlink()
        _write_uncompressed_manifest(duckdb, index, metadata, raw)
        row_count, cluster_count, view_counts, required = _scan_manifest(raw)
        if row_count != arguments.expected_row_count:
            raise RuntimeError("PINDER train-only row count differs from frozen anchor")
        if cluster_count != arguments.expected_cluster_count:
            raise RuntimeError("PINDER train-only cluster count differs from frozen anchor")
        archive_audit = _validate_archive_members(archive, required)
        _deterministic_gzip(raw, compressed)
        os.replace(compressed, output)
        output.chmod(0o444)
    finally:
        raw.unlink(missing_ok=True)
        compressed.unlink(missing_ok=True)

    result = {
        "authorization": {
            "development_decontamination_complete": False,
            "license_review_complete": False,
            "stage_a_training": False,
        },
        "claim_boundary": "Authenticated PINDER 2024-02 train-only strict-quality system manifest. No validation or test rows are present. Training remains fail-closed pending data-specific license resolution and frozen development decontamination.",
        "format_version": "abpairfm_pinder_train_system_manifest_v1",
        "inputs": {
            "archive_sha256": materialization["archive"]["sha256"],
            "duckdb_sha256": arguments.expected_duckdb_sha256,
            "index_sha256": arguments.expected_index_sha256,
            "materialization_receipt_sha256": arguments.expected_materialization_receipt_sha256,
            "metadata_sha256": arguments.expected_metadata_sha256,
            "protocol_sha256": arguments.expected_protocol_sha256,
            "source_sha256": _sha256_file(source),
        },
        "inventory": {
            "strict_quality_cluster_count": cluster_count,
            "strict_quality_system_count": row_count,
            "view_availability": view_counts,
        },
        "output": {"bytes": output.stat().st_size, "sha256": _sha256_file(output)},
        "quality_predicate": " ".join(QUALITY_PREDICATE.split()),
        "status": "TRAIN_ONLY_SYSTEM_MANIFEST_NOT_AUTHORIZED",
        "structure_archive_audit": archive_audit,
        "test_access": {
            "pinder_test_rows_projected": False,
            "pinder_test_structure_archive_requested": False,
            "pinder_test_structure_bytes_accessed": False,
        },
    }
    _publish_json(receipt, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
