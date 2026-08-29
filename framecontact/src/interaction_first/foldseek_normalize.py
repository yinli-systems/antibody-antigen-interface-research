"""Normalize hashed Foldseek chain identifiers back to SAbDab2 instances."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .sabdab2_split_matrix import structure_ids_sha256
from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file
from .structural_novelty import FOLDSEEK_RECEIPT_FORMAT


_HASH = re.compile(r"[0-9a-f]{64}")


def normalize_foldseek_alignments(
    *,
    raw_path: Path,
    raw_sha256: str,
    export_manifest_path: Path,
    export_manifest_sha256: str,
    output_path: Path,
    receipt_path: Path,
    foldseek_version: str,
    command: list[str],
) -> dict[str, Any]:
    """Write canonical instance-level qTM/tTM rows and a provenance receipt."""
    authenticated_raw = authenticate_file(raw_path, raw_sha256, "raw Foldseek output")
    authenticated_manifest = authenticate_file(
        export_manifest_path, export_manifest_sha256, "antigen export manifest"
    )
    with export_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"antigen_structure_id", "foldseek_structure_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Antigen export manifest lacks Foldseek identifier mapping")
    mapping = {row["foldseek_structure_id"]: row["antigen_structure_id"] for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("Foldseek structure IDs must be unique")
    instance_indices = {
        identifier: index for index, identifier in enumerate(sorted(mapping.values()))
    }
    structure_count = len(mapping)
    expected_pair_count = structure_count * structure_count
    observed_pairs = bytearray(expected_pair_count)

    def instance(value: str) -> str:
        match = _HASH.search(value)
        if match is None or match.group(0) not in mapping:
            raise ValueError(f"Unknown Foldseek identifier: {value!r}")
        return mapping[match.group(0)]

    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite normalized Foldseek outputs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = 0
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        temporary_path = Path(temporary_name)
        with raw_path.open("r", encoding="utf-8", newline="") as source, os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as target:
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")
            writer.writerow(("query", "target", "qtmscore", "ttmscore"))
            for line_number, line in enumerate(source, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 4:
                    raise ValueError(
                        f"Raw Foldseek row {line_number} has {len(fields)} fields"
                    )
                query_tm, target_tm = float(fields[2]), float(fields[3])
                if not (0 <= query_tm <= 1 and 0 <= target_tm <= 1):
                    raise ValueError("Foldseek TM scores must lie in [0, 1]")
                query, target_instance = instance(fields[0]), instance(fields[1])
                flat_index = (
                    instance_indices[query] * structure_count
                    + instance_indices[target_instance]
                )
                if observed_pairs[flat_index]:
                    raise ValueError(
                        f"Duplicate Foldseek pair at raw row {line_number}: "
                        f"{query!r}, {target_instance!r}"
                    )
                observed_pairs[flat_index] = 1
                writer.writerow((query, target_instance, fields[2], fields[3]))
                raw_rows += 1
            if raw_rows != expected_pair_count or 0 in observed_pairs:
                missing = observed_pairs.count(0)
                raise ValueError(
                    "Foldseek output is not a complete ordered Cartesian grid: "
                    f"rows={raw_rows}, expected={expected_pair_count}, missing={missing}"
                )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    receipt = {
        "schema_version": FOLDSEEK_RECEIPT_FORMAT,
        "foldseek_version": foldseek_version,
        "query_target_same_database": True,
        "alignment_type": 1,
        "exhaustive_search": True,
        "test_labels_read": False,
        "structure_count": structure_count,
        "structure_ids_sha256": structure_ids_sha256(mapping.values()),
        "format_output": ["query", "target", "qtmscore", "ttmscore"],
        "raw_row_count": raw_rows,
        "expected_ordered_pair_count": expected_pair_count,
        "complete_ordered_cartesian_grid": True,
        "raw_sha256": authenticated_raw,
        "export_manifest_sha256": authenticated_manifest,
        "normalized_sha256": sha256_file(output_path),
        "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
        "command": command,
        "claim_boundary": "Foldseek all-vs-all evidence only; no split or performance claim.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_descriptor, receipt_name = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.", suffix=".tmp", dir=receipt_path.parent
    )
    receipt_temporary = Path(receipt_name)
    try:
        with os.fdopen(receipt_descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(receipt_temporary, receipt_path)
    finally:
        receipt_temporary.unlink(missing_ok=True)
    return receipt


__all__ = ["normalize_foldseek_alignments"]
