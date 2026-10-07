#!/usr/bin/env python3
"""Authenticate Foldseek 3Di descriptors against the predicted-structure cache."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping


THREEDI_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
STRUCTURE_COLUMNS = {"member", "sequence", "sequence_sha256"}
OUTPUT_COLUMNS = ("member", "sequence_sha256", "three_di_tokens")


def allow_large_foldseek_fields() -> int:
    """Raise csv's parser limit for long per-residue descriptor fields."""
    limit = sys.maxsize
    while True:
        try:
            return csv.field_size_limit(limit)
        except OverflowError:
            limit //= 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_structures(path: Path) -> dict[str, tuple[str, str]]:
    structures: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not STRUCTURE_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("Predicted-structure manifest schema differs")
        for row in reader:
            member = row["member"]
            sequence = row["sequence"]
            sequence_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            if sequence_sha != row["sequence_sha256"]:
                raise ValueError("Predicted-structure sequence SHA-256 differs")
            if member in structures:
                raise ValueError("Predicted-structure member is duplicated")
            structures[member] = (sequence, sequence_sha)
    if not structures:
        raise ValueError("Predicted-structure manifest is empty")
    return structures


def parse_foldseek(
    path: Path, structures: Mapping[str, tuple[str, str]]
) -> list[dict[str, str]]:
    allow_large_foldseek_fields()
    rows: list[dict[str, str]] = []
    observed: set[str] = set()
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if len(fields) != 4:
                raise ValueError("Foldseek descriptor row must contain four columns")
            identifier, sequence, three_di, descriptor_text = fields
            if not identifier.endswith("_A"):
                raise ValueError("Foldseek descriptor identifier is not chain A")
            member = identifier[:-2]
            if member not in structures:
                raise ValueError(f"Unexpected Foldseek structure member: {member}")
            if member in observed:
                raise ValueError("Foldseek structure member is duplicated")
            expected_sequence, sequence_sha = structures[member]
            if sequence != expected_sequence:
                raise ValueError("Foldseek amino-acid sequence differs")
            if len(three_di) != len(sequence) or not set(three_di).issubset(THREEDI_ALPHABET):
                raise ValueError("Foldseek 3Di token axis or alphabet differs")
            descriptors = descriptor_text.split(",") if descriptor_text else []
            if len(descriptors) != 10 * len(sequence):
                raise ValueError("Foldseek continuous descriptor dimension differs")
            if not all(math.isfinite(float(value)) for value in descriptors):
                raise ValueError("Foldseek descriptor contains a non-finite value")
            observed.add(member)
            rows.append(
                {
                    "member": member,
                    "sequence_sha256": sequence_sha,
                    "three_di_tokens": three_di,
                }
            )
    missing = set(structures) - observed
    if missing:
        raise ValueError(f"Foldseek output is missing {len(missing)} structures")
    return sorted(rows, key=lambda row: row["member"])


def write_tokens(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(
                        text,
                        fieldnames=OUTPUT_COLUMNS,
                        delimiter="\t",
                        lineterminator="\n",
                        extrasaction="raise",
                    )
                    writer.writeheader()
                    writer.writerows(rows)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(arguments: argparse.Namespace) -> dict[str, object]:
    for path, expected, label in (
        (arguments.structure_manifest, arguments.structure_manifest_sha256, "structure manifest"),
        (arguments.foldseek_binary, arguments.foldseek_binary_sha256, "Foldseek binary"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 differs")
    if arguments.output_tokens.exists() or arguments.output_result.exists():
        raise FileExistsError("3Di output already exists")
    structures = load_structures(arguments.structure_manifest)
    rows = parse_foldseek(arguments.foldseek_output, structures)
    write_tokens(arguments.output_tokens, rows)
    payload: dict[str, object] = {
        "authorization": {"model_training": False, "structure_shard_materialization": True},
        "claim_boundary": "Foldseek 3Di descriptors from authenticated train-only predicted monomers; bound, validation, and test structures were not read.",
        "counts": {"predicted_structure_members": len(rows)},
        "format_version": "pinder_predicted_structure_3di_result_v1",
        "inputs": {
            "foldseek_binary_sha256": arguments.foldseek_binary_sha256,
            "foldseek_output_sha256": sha256_file(arguments.foldseek_output),
            "structure_manifest_sha256": arguments.structure_manifest_sha256,
        },
        "outputs": {"three_di_manifest_sha256": sha256_file(arguments.output_tokens)},
        "status": "PASS_AUTHENTICATED_PREDICTED_STRUCTURE_3DI",
        "test_access": {
            "bound_coordinates_read": False,
            "pinder_test_structures_read": False,
            "pinder_validation_structures_read": False,
        },
    }
    create_only(arguments.output_result, canonical_json_bytes(payload))
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--foldseek-binary", type=Path, required=True)
    parser.add_argument("--foldseek-binary-sha256", required=True)
    parser.add_argument("--foldseek-output", type=Path, required=True)
    parser.add_argument("--output-tokens", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_arguments()), sort_keys=True))
