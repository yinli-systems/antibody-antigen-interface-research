#!/usr/bin/env python3
"""Export native VH/VL chains and remove development-homologous pairs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_COLUMNS = (
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_sequence(value: str, label: str) -> str:
    sequence = value.strip().upper()
    if not sequence or not set(sequence).issubset(AMINO_ACIDS):
        raise ValueError(f"{label} is empty or noncanonical")
    return sequence


def checked_hash(value: str, label: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return digest


def load_source_receipt(input_path: Path, receipt_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != "abpairfm_oas_stage_b_candidates_v2"
        or receipt.get("status") != "NORMALIZED_NOT_SUFFICIENT_NOT_DECONTAMINATED"
        or receipt.get("test_access") is not False
        or receipt.get("output", {}).get("sha256") != sha256_file(input_path)
        or receipt.get("output", {}).get("bytes") != input_path.stat().st_size
        or receipt.get("counts", {}).get("unique_exact_pairs") != 2_585_593
    ):
        raise ValueError("OAS Stage-B source receipt differs")
    return receipt


def load_development_receipt(fasta_path: Path, receipt_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "FROZEN_VALIDATION_SEQUENCE_PANEL"
        or receipt.get("test_access") is not False
        or receipt.get("authorization", {}).get("training_decontamination") is not True
        or receipt.get("output_fasta", {}).get("sha256") != sha256_file(fasta_path)
    ):
        raise ValueError("Development sequence receipt differs")
    return receipt


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError("OAS Stage-B columns differ")
        yield from reader


def export_fasta(input_path: Path, output_path: Path, expected_rows: int) -> int:
    count = 0
    with output_path.open("wt", encoding="ascii", newline="\n") as handle:
        for row in rows(input_path):
            row_hash = checked_hash(row["row_hash"], "row_hash")
            heavy = checked_sequence(row["heavy_sequence"], f"{row_hash} heavy")
            light = checked_sequence(row["light_sequence"], f"{row_hash} light")
            handle.write(f">{row_hash}|heavy\n{heavy}\n>{row_hash}|light\n{light}\n")
            count += 1
    if count != expected_rows:
        raise ValueError(f"OAS Stage-B row count differs: {count}")
    return count


def excluded_pair_hashes(hits_path: Path) -> tuple[set[str], int]:
    excluded: set[str] = set()
    hit_count = 0
    with hits_path.open("rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8:
                raise ValueError(f"Malformed MMseqs hit at line {line_number}")
            target = fields[1]
            try:
                row_hash, chain = target.rsplit("|", 1)
            except ValueError as error:
                raise ValueError(f"Malformed MMseqs target at line {line_number}") from error
            checked_hash(row_hash, "MMseqs target row_hash")
            if chain not in {"heavy", "light"}:
                raise ValueError(f"Malformed MMseqs target chain at line {line_number}")
            excluded.add(row_hash)
            hit_count += 1
    return excluded, hit_count


def publish_create_only(path: Path, payload_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(payload_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite Stage-B artifact: {path}") from error
    path.chmod(0o444)


def filter_pairs(
    input_path: Path,
    output_path: Path,
    hits_path: Path,
    hits_output: Path,
    receipt_path: Path,
    source_receipt_path: Path,
    development_fasta: Path,
    development_receipt_path: Path,
    mmseqs_version: str,
) -> dict:
    source_receipt = load_source_receipt(input_path, source_receipt_path)
    development_receipt = load_development_receipt(
        development_fasta, development_receipt_path
    )
    excluded, hit_count = excluded_pair_hashes(hits_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    kept = 0
    observed = 0
    try:
        with temporary_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(
                        text, fieldnames=REQUIRED_COLUMNS, delimiter="\t", lineterminator="\n"
                    )
                    writer.writeheader()
                    for row in rows(input_path):
                        observed += 1
                        if row["row_hash"] in excluded:
                            continue
                        writer.writerow(row)
                        kept += 1
        if observed != source_receipt["counts"]["unique_exact_pairs"]:
            raise ValueError("OAS Stage-B source row count changed during filtering")
        publish_create_only(output_path, temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    descriptor, temporary_hits = tempfile.mkstemp(
        prefix=f".{hits_output.name}.", dir=hits_output.parent
    )
    os.close(descriptor)
    temporary_hits_path = Path(temporary_hits)
    try:
        with hits_path.open("rb") as source, temporary_hits_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    compressed.write(block)
        publish_create_only(hits_output, temporary_hits_path)
    finally:
        temporary_hits_path.unlink(missing_ok=True)

    receipt = {
        "authorization": {
            "full_5500000_pair_stage_b_training": False,
            "scale_restricted_stage_b_training": True,
            "test_access": False,
        },
        "claim_boundary": "Development-decontaminated OAS exact native VH/VL pairs only. The corpus remains below the preregistered 5.5M full Stage-B target and is not PairedAbNGS-complete; it may support only a scale-restricted CDR-specialization experiment.",
        "counts": {
            "excluded_pairs": len(excluded),
            "kept_pairs": kept,
            "mmseqs_hits": hit_count,
            "source_pairs": observed,
            "target_deficit_from_5500000": 5_500_000 - kept,
        },
        "decontamination": {
            "coverage": 0.8,
            "coverage_mode": 0,
            "minimum_sequence_identity": 0.5,
            "mmseqs_version": mmseqs_version.strip(),
            "query": "frozen AsEP and SAbDab2 development sequences",
            "target": "OAS native heavy and light chains",
        },
        "format_version": "abpairfm_oas_stage_b_devclean_pairs_v1",
        "inputs": {
            "development_fasta_sha256": sha256_file(development_fasta),
            "development_receipt_sha256": sha256_file(development_receipt_path),
            "source_receipt_sha256": sha256_file(source_receipt_path),
            "source_sha256": sha256_file(input_path),
        },
        "outputs": {
            "hits": {
                "bytes": hits_output.stat().st_size,
                "sha256": sha256_file(hits_output),
            },
            "pairs": {
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            },
        },
        "source_receipt_status": source_receipt["status"],
        "development_receipt_status": development_receipt["status"],
        "status": "PASS_SCALE_RESTRICTED_STAGE_B_DECONTAMINATION",
        "test_access": False,
    }
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_receipt = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.", dir=receipt_path.parent
    )
    temporary_receipt_path = Path(temporary_receipt)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        publish_create_only(receipt_path, temporary_receipt_path)
    finally:
        temporary_receipt_path.unlink(missing_ok=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export-fasta")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--source-receipt", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    filtered = subparsers.add_parser("filter")
    filtered.add_argument("--input", type=Path, required=True)
    filtered.add_argument("--source-receipt", type=Path, required=True)
    filtered.add_argument("--development-fasta", type=Path, required=True)
    filtered.add_argument("--development-receipt", type=Path, required=True)
    filtered.add_argument("--hits", type=Path, required=True)
    filtered.add_argument("--hits-output", type=Path, required=True)
    filtered.add_argument("--output", type=Path, required=True)
    filtered.add_argument("--receipt", type=Path, required=True)
    filtered.add_argument("--mmseqs-version", required=True)
    arguments = parser.parse_args()
    if arguments.command == "export-fasta":
        receipt = load_source_receipt(arguments.input, arguments.source_receipt)
        count = export_fasta(
            arguments.input,
            arguments.output,
            receipt["counts"]["unique_exact_pairs"],
        )
        print(json.dumps({"exported_pairs": count}, sort_keys=True))
        return
    receipt = filter_pairs(
        arguments.input,
        arguments.output,
        arguments.hits,
        arguments.hits_output,
        arguments.receipt,
        arguments.source_receipt,
        arguments.development_fasta,
        arguments.development_receipt,
        arguments.mmseqs_version,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
