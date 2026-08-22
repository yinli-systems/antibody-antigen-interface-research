#!/usr/bin/env python3
"""Assemble strict-train PINDER holo sequences from selected mapping Parquets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def sequence_query(mapping_glob: str, output: Path) -> str:
    escaped_glob = mapping_glob.replace("'", "''")
    return f"""
    COPY (
      WITH residues AS (
        SELECT
          filename,
          resi,
          min(one_letter_code_can) AS canonical_code,
          count(DISTINCT one_letter_code_can) AS canonical_code_count,
          count(*) AS mapping_rows,
          max(resolved = 1) AS resolved
        FROM read_parquet('{escaped_glob}', filename=true)
        GROUP BY filename, resi
      )
      SELECT
        regexp_extract(filename, '([^/]+)\\.parquet$', 1) AS mapping_id,
        string_agg(canonical_code, '' ORDER BY resi) AS sequence,
        sum(mapping_rows) AS mapping_rows,
        count(*) AS residue_rows,
        count_if(resolved) AS resolved_residue_rows,
        count_if(
          resi IS NULL OR canonical_code_count != 1 OR
          canonical_code IS NULL OR
          NOT regexp_full_match(canonical_code, '[A-Z]')
        ) AS invalid_residue_indices,
        count_if(canonical_code_count > 1) AS ambiguous_residue_indices
      FROM residues
      GROUP BY filename
      ORDER BY mapping_id
    ) TO '{_sql_path(output)}'
    (FORMAT CSV, HEADER TRUE, DELIMITER '\t');
    """


def _publish_create_only(path: Path, payload: dict[str, object]) -> None:
    serialized = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def publish_sequences(
    raw_tsv: Path,
    output_fasta: Path,
    output_manifest: Path,
    expected_count: int,
) -> dict[str, int]:
    sequence_hashes: set[str] = set()
    mapping_count = 0
    residue_count = 0
    resolved_residue_count = 0
    mapping_row_count = 0
    minimum_sequence_length: int | None = None
    maximum_sequence_length = 0
    shorter_than_20 = 0
    longer_than_1_020 = 0
    longer_than_2_048 = 0
    unknown_x_residues = 0
    nonstandard_uppercase_residues = 0
    standard_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    previous_id: str | None = None
    with (
        raw_tsv.open("r", encoding="utf-8", newline="") as source,
        output_fasta.open("x", encoding="ascii", newline="\n") as fasta,
        gzip.open(output_manifest, "xt", encoding="ascii", newline="") as manifest,
    ):
        reader = csv.DictReader(source, delimiter="\t")
        expected_fields = (
            "mapping_id",
            "sequence",
            "mapping_rows",
            "residue_rows",
            "resolved_residue_rows",
            "invalid_residue_indices",
            "ambiguous_residue_indices",
        )
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError("Unexpected PINDER sequence aggregation schema")
        writer = csv.DictWriter(
            manifest,
            fieldnames=expected_fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            mapping_id = row["mapping_id"]
            sequence = row["sequence"]
            mapping_rows = int(row["mapping_rows"])
            rows = int(row["residue_rows"])
            resolved = int(row["resolved_residue_rows"])
            invalid = int(row["invalid_residue_indices"])
            ambiguous = int(row["ambiguous_residue_indices"])
            if not mapping_id or mapping_id.endswith(".parquet"):
                raise RuntimeError("Invalid PINDER mapping identifier")
            if previous_id is not None and mapping_id <= previous_id:
                raise RuntimeError("PINDER mapping identifiers are not unique and sorted")
            if invalid or ambiguous or mapping_rows < rows or len(sequence) != rows:
                raise RuntimeError(f"Invalid canonical residue mapping: {mapping_id}")
            if not sequence:
                raise RuntimeError(f"PINDER endpoint sequence is empty: {mapping_id}")
            previous_id = mapping_id
            mapping_count += 1
            mapping_row_count += mapping_rows
            residue_count += rows
            resolved_residue_count += resolved
            sequence_hashes.add(hashlib.sha256(sequence.encode("ascii")).hexdigest())
            length = len(sequence)
            minimum_sequence_length = (
                length
                if minimum_sequence_length is None
                else min(minimum_sequence_length, length)
            )
            maximum_sequence_length = max(maximum_sequence_length, length)
            shorter_than_20 += int(length < 20)
            longer_than_1_020 += int(length > 1_020)
            longer_than_2_048 += int(length > 2_048)
            unknown_x_residues += sequence.count("X")
            nonstandard_uppercase_residues += sum(
                residue not in standard_amino_acids and residue != "X"
                for residue in sequence
            )
            fasta.write(f">{mapping_id}\n{sequence}\n")
            writer.writerow(row)
    if mapping_count != expected_count:
        raise RuntimeError(
            f"PINDER mapping count differs: {mapping_count} != {expected_count}"
        )
    return {
        "canonical_residues": residue_count,
        "collapsed_mapping_rows": mapping_row_count - residue_count,
        "mapping_annotation_rows": mapping_row_count,
        "mapping_sequences": mapping_count,
        "maximum_sequence_length": maximum_sequence_length,
        "minimum_sequence_length": minimum_sequence_length or 0,
        "nonstandard_uppercase_residues": nonstandard_uppercase_residues,
        "sequences_longer_than_1020": longer_than_1_020,
        "sequences_longer_than_2048": longer_than_2_048,
        "sequences_shorter_than_20": shorter_than_20,
        "resolved_residues": resolved_residue_count,
        "unknown_x_residues": unknown_x_residues,
        "unique_exact_sequences": len(sequence_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--expected-duckdb-sha256", required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--mapping-extraction-receipt", type=Path, required=True)
    parser.add_argument("--expected-mapping-count", type=int, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    duckdb = arguments.duckdb.resolve()
    mapping_root = arguments.mapping_root.resolve()
    extraction_receipt = arguments.mapping_extraction_receipt.resolve()
    output_fasta = arguments.output_fasta.resolve()
    output_manifest = arguments.output_manifest.resolve()
    receipt = arguments.receipt.resolve()
    if (
        duckdb.is_symlink()
        or not duckdb.is_file()
        or _sha256(duckdb) != arguments.expected_duckdb_sha256
    ):
        raise RuntimeError("DuckDB executable differs from frozen anchor")
    if mapping_root.is_symlink() or not mapping_root.is_dir():
        raise RuntimeError("PINDER mapping root is unavailable")
    extraction_payload = json.loads(extraction_receipt.read_text(encoding="ascii"))
    if (
        extraction_payload.get("format_version")
        != "abpairfm_pinder_holo_mapping_extraction_v1"
        or extraction_payload.get("counts", {}).get(
            "required_holo_mapping_members"
        )
        != arguments.expected_mapping_count
        or extraction_payload.get("test_access") is not False
    ):
        raise RuntimeError("PINDER mapping extraction receipt differs")
    for path in (output_fasta, output_manifest, receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="abpairfm-pinder-sequences-",
        dir="/dev/shm",
    ) as temporary:
        raw_tsv = Path(temporary) / "sequences.tsv"
        result = subprocess.run(
            [
                str(duckdb),
                "-batch",
                "-c",
                sequence_query(str(mapping_root / "*.parquet"), raw_tsv),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"DuckDB PINDER sequence aggregation failed: {result.stderr.strip()}"
            )
        counts = publish_sequences(
            raw_tsv,
            output_fasta,
            output_manifest,
            arguments.expected_mapping_count,
        )

    payload = {
        "authorization": {
            "development_decontamination": False,
            "stage_a_training": False,
        },
        "claim_boundary": (
            "Canonical sequences for strict train-only PINDER holo endpoints; "
            "development homology exclusion and Stage A merge remain pending."
        ),
        "counts": counts,
        "format_version": "abpairfm_pinder_holo_sequences_v3",
        "inputs": {
            "duckdb_sha256": arguments.expected_duckdb_sha256,
            "mapping_extraction_receipt_sha256": _sha256(extraction_receipt),
        },
        "outputs": {
            "fasta_bytes": output_fasta.stat().st_size,
            "fasta_sha256": _sha256(output_fasta),
            "manifest_bytes": output_manifest.stat().st_size,
            "manifest_sha256": _sha256(output_manifest),
        },
        "status": "SEQUENCES_ASSEMBLED_NOT_DECONTAMINATED",
        "test_access": False,
    }
    _publish_create_only(receipt, payload)
    print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
