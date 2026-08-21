#!/usr/bin/env python3
"""Build a quality-filtered exact-pair OAS Stage B candidate corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import io
from typing import Any


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
CDR_FIELDS = (
    "cdr1_aa_heavy",
    "cdr2_aa_heavy",
    "cdr3_aa_heavy",
    "cdr1_aa_light",
    "cdr2_aa_light",
    "cdr3_aa_light",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_create_only(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
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
        raise FileExistsError(f"Refusing to overwrite OAS Stage B receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _amino_acids(value: str) -> str:
    return value.upper().replace("-", "").replace(".", "").strip()


def _true(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "t", "true"}


def normalize_row_with_reason(
    row: dict[str, str],
) -> tuple[tuple[str, ...] | None, str]:
    if not _true(row.get("productive_heavy")) or not _true(row.get("productive_light")):
        return None, "not_productive"
    if not _true(row.get("vj_in_frame_heavy")) or not _true(row.get("vj_in_frame_light")):
        return None, "not_in_frame"
    if _true(row.get("stop_codon_heavy")) or _true(row.get("stop_codon_light")):
        return None, "stop_codon"
    if row.get("locus_heavy") not in {"H", "IGH"} or row.get("locus_light") not in {
        "K",
        "L",
        "IGK",
        "IGL",
    }:
        return None, "unexpected_locus"
    heavy = _amino_acids(row.get("sequence_alignment_aa_heavy", ""))
    light = _amino_acids(row.get("sequence_alignment_aa_light", ""))
    if not 80 <= len(heavy) <= 170 or not 70 <= len(light) <= 130:
        return None, "sequence_length"
    if not set(heavy).issubset(AMINO_ACIDS) or not set(light).issubset(AMINO_ACIDS):
        return None, "noncanonical_sequence"
    cdrs = tuple(_amino_acids(row.get(field, "")) for field in CDR_FIELDS)
    if any(not cdr or not set(cdr).issubset(AMINO_ACIDS) for cdr in cdrs):
        return None, "missing_or_noncanonical_cdr"
    if any(cdr not in heavy for cdr in cdrs[:3]):
        return None, "heavy_cdr_not_in_sequence"
    if any(cdr not in light for cdr in cdrs[3:]):
        return None, "light_cdr_not_in_sequence"
    return (heavy, light, *cdrs), "accepted"


def normalize_row(row: dict[str, str]) -> tuple[str, ...] | None:
    return normalize_row_with_reason(row)[0]


def _metadata(handle: Any) -> dict[str, Any]:
    row = next(csv.reader([handle.readline()]))
    if len(row) != 1:
        raise ValueError("Malformed OAS data-unit metadata")
    value = json.loads(row[0])
    if not isinstance(value, dict):
        raise ValueError("OAS metadata is not an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oas-root", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    source_receipt_path = arguments.materialization_receipt.resolve()
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if source_receipt.get("status") != "MATERIALIZED_NOT_DEDUPLICATED_NOT_DECONTAMINATED":
        raise ValueError("OAS materialization receipt is not eligible for normalization")
    if source_receipt.get("test_access") is not False:
        raise ValueError("OAS source receipt crossed a test boundary")
    oas_root = arguments.oas_root.resolve()
    pairs: dict[tuple[str, str], tuple[str, ...]] = {}
    source_rows = 0
    quality_rows = 0
    duplicate_rows = 0
    rejection_reasons: Counter[str] = Counter()
    source_file_count = 0
    for entry in source_receipt["files"]:
        relative = Path(entry["relative_path"])
        path = oas_root / relative
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"OAS data-unit byte count differs: {relative}")
        if _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"OAS data-unit SHA-256 differs: {relative}")
        source_file_count += 1
        study = relative.parts[0]
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            metadata = _metadata(handle)
            reader = csv.DictReader(handle)
            for row in reader:
                source_rows += 1
                normalized, reason = normalize_row_with_reason(row)
                if normalized is None:
                    rejection_reasons[reason] += 1
                    continue
                quality_rows += 1
                heavy, light, *cdrs = normalized
                pair = (heavy, light)
                row_identity = "|".join(
                    (
                        study,
                        str(metadata.get("Run", "")),
                        row.get("sequence_id_heavy", ""),
                        row.get("sequence_id_light", ""),
                    )
                )
                row_hash = hashlib.sha256(row_identity.encode("utf-8")).hexdigest()
                donor_identity = "|".join(
                    (study, str(metadata.get("Run", "")), str(metadata.get("Subject", "")))
                )
                donor_hash = hashlib.sha256(donor_identity.encode("utf-8")).hexdigest()
                record = (
                    heavy,
                    light,
                    *cdrs,
                    study,
                    str(metadata.get("Run", "")),
                    str(metadata.get("Species", "")),
                    donor_hash,
                    row_hash,
                )
                previous = pairs.get(pair)
                if previous is None or record[-1] < previous[-1]:
                    pairs[pair] = record
                if previous is not None:
                    duplicate_rows += 1
    if source_rows != int(source_receipt["total_sequence_rows"]):
        raise ValueError("Parsed OAS row count differs from materialization receipt")
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite OAS Stage B corpus: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary_path = Path(temporary)
    os.close(descriptor)
    header = (
        "heavy_sequence",
        "light_sequence",
        *CDR_FIELDS,
        "study",
        "run",
        "species",
        "donor_hash",
        "row_hash",
    )
    try:
        with temporary_path.open("wb") as raw_handle, gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", mtime=0
        ) as compressed_handle, io.TextIOWrapper(
            compressed_handle, encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            for record in sorted(pairs.values(), key=lambda item: (item[0], item[1])):
                writer.writerow(record)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary_path, output)
        output.chmod(0o444)
    finally:
        temporary_path.unlink(missing_ok=True)
    unique_pairs = len(pairs)
    target = 5_500_000
    receipt = {
        "authorization": {
            "development_decontamination": False,
            "stage_b_training": False,
        },
        "claim_boundary": "Exact-pair OAS quality inventory only. It is not 95%-clustered, merged with PairedAbNGS, or decontaminated from development data.",
        "counts": {
            "duplicate_quality_rows": duplicate_rows,
            "quality_rows_before_exact_dedup": quality_rows,
            "source_data_units": source_file_count,
            "source_rows": source_rows,
            "unique_exact_pairs": unique_pairs,
        },
        "format_version": "abpairfm_oas_stage_b_candidates_v2",
        "minimum_native_pair_target": target,
        "native_pair_target_deficit": max(0, target - unique_pairs),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "output": {
            "bytes": output.stat().st_size,
            "name": output.name,
            "sha256": _sha256_file(output),
        },
        "source_materialization_receipt_sha256": _sha256_file(source_receipt_path),
        "status": (
            "NORMALIZED_SUFFICIENT_NOT_DECONTAMINATED"
            if unique_pairs >= target
            else "NORMALIZED_NOT_SUFFICIENT_NOT_DECONTAMINATED"
        ),
        "test_access": False,
    }
    _publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
