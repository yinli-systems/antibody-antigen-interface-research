#!/usr/bin/env python3
"""Extract the STRING sequences referenced by a frozen endpoint-id inventory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def read_sorted_ids(path: Path) -> set[str]:
    identifiers: set[str] = set()
    previous = ""
    with path.open("rt", encoding="utf-8") as handle:
        for number, raw_line in enumerate(handle, start=1):
            identifier = raw_line.rstrip("\n")
            if not identifier or any(character.isspace() for character in identifier):
                raise ValueError(f"Malformed endpoint id at line {number}")
            if previous and identifier <= previous:
                raise ValueError("Endpoint ids are not strictly sorted and unique")
            identifiers.add(identifier)
            previous = identifier
    if not identifiers:
        raise ValueError("Endpoint id inventory is empty")
    return identifiers


def extract_fasta(
    source_path: Path, target_ids: set[str], output_path: Path
) -> tuple[int, int, int]:
    remaining = set(target_ids)
    source_records = 0
    selected_records = 0
    selected_residues = 0
    current_id: str | None = None
    sequence: list[str] = []

    def emit(handle: object) -> None:
        nonlocal selected_records, selected_residues
        if current_id is None or current_id not in remaining:
            return
        value = "".join(sequence).upper()
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"Malformed sequence for {current_id}")
        handle.write(f">{current_id}\n{value}\n")  # type: ignore[attr-defined]
        remaining.remove(current_id)
        selected_records += 1
        selected_residues += len(value)

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wt", encoding="utf-8", newline="") as output_handle:
            with gzip.open(source_path, "rt", encoding="utf-8", newline="") as source_handle:
                for raw_line in source_handle:
                    if raw_line.startswith(">"):
                        emit(output_handle)
                        source_records += 1
                        current_id = raw_line[1:].split(maxsplit=1)[0]
                        sequence = []
                    else:
                        sequence.append(raw_line.strip())
                emit(output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if remaining:
            examples = sorted(remaining)[:5]
            raise ValueError(
                f"{len(remaining)} STRING endpoints lack sequences; examples={examples}"
            )
        os.link(temporary_path, output_path)
        output_path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite endpoint FASTA: {output_path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return source_records, selected_records, selected_residues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-ids", type=Path, required=True)
    parser.add_argument("--source-fasta", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    source_receipt_path = arguments.source_receipt.resolve()
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if source_receipt.get("status") != "MATERIALIZED_NOT_NORMALIZED":
        raise ValueError("STRING source receipt is not eligible")
    if source_receipt.get("test_access") is not False:
        raise ValueError("STRING source receipt crossed a test boundary")
    expected = source_receipt["files"]["protein.sequences.v12.0.fa.gz"]
    source_fasta = arguments.source_fasta.resolve()
    if source_fasta.stat().st_size != int(expected["bytes"]):
        raise ValueError("STRING sequence archive byte count differs")
    source_sha256 = sha256_file(source_fasta)
    if source_sha256 != expected["sha256"]:
        raise ValueError("STRING sequence archive SHA-256 differs")

    endpoint_ids = arguments.endpoint_ids.resolve()
    targets = read_sorted_ids(endpoint_ids)
    output_fasta = arguments.output_fasta.resolve()
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    source_records, selected_records, selected_residues = extract_fasta(
        source_fasta, targets, output_fasta
    )
    receipt = {
        "authorization": {"cluster_deduplication": False, "stage_a_training": False},
        "counts": {
            "endpoint_ids": len(targets),
            "selected_residues": selected_residues,
            "selected_sequence_records": selected_records,
            "source_sequence_records": source_records,
        },
        "endpoint_ids": {
            "bytes": endpoint_ids.stat().st_size,
            "sha256": sha256_file(endpoint_ids),
        },
        "format_version": "abpairfm_string_endpoint_fasta_v1",
        "output_fasta": {
            "bytes": output_fasta.stat().st_size,
            "sha256": sha256_file(output_fasta),
        },
        "source_materialization_receipt_sha256": sha256_file(source_receipt_path),
        "source_sequence_archive_sha256": source_sha256,
        "status": "ENDPOINT_SEQUENCES_EXTRACTED_NOT_CLUSTERED",
        "test_access": False,
    }
    publish_create_only(
        arguments.receipt.resolve(),
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
