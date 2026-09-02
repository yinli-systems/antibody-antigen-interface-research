#!/usr/bin/env python3
"""Create a train-only allowlist for independent predicted structure inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Iterable, Mapping


REQUIRED_COLUMNS = {
    "split",
    "id",
    "cluster_id",
    "predicted_R_pdb",
    "predicted_L_pdb",
    "sequence_R",
    "sequence_L",
    "sequence_sha256_R",
    "sequence_sha256_L",
    "system_weight",
}
PDB_MEMBER = re.compile(r"^[A-Za-z0-9_.-]+\.pdb$")
OUTPUT_COLUMNS = (
    "id",
    "cluster_id",
    "predicted_R_pdb",
    "predicted_L_pdb",
    "sequence_R",
    "sequence_L",
    "sequence_sha256_R",
    "sequence_sha256_L",
    "system_weight",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def validate_member(value: str) -> None:
    if not PDB_MEMBER.fullmatch(value) or Path(value).name != value:
        raise ValueError(f"Unsafe predicted structure member: {value!r}")


def write_inventory(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
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


def build(arguments: argparse.Namespace) -> dict[str, object]:
    observed_manifest_sha = sha256_file(arguments.curriculum_manifest)
    if observed_manifest_sha != arguments.curriculum_manifest_sha256:
        raise ValueError("Curriculum manifest SHA-256 differs")
    if sha256_file(arguments.protocol) != arguments.protocol_sha256:
        raise ValueError("Multimodal protocol SHA-256 differs")

    selected: list[dict[str, str]] = []
    identifiers: set[str] = set()
    structure_members: set[str] = set()
    input_count = 0
    with gzip.open(
        arguments.curriculum_manifest,
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("Curriculum manifest schema differs")
        for row in reader:
            input_count += 1
            if row["split"] != "train":
                raise ValueError("Multimodal inventory must remain train-only")
            identifier = row["id"]
            if not identifier or identifier in identifiers:
                raise ValueError("Curriculum identifiers are empty or duplicated")
            identifiers.add(identifier)
            right = row["predicted_R_pdb"]
            left = row["predicted_L_pdb"]
            if not right or not left:
                continue
            validate_member(right)
            validate_member(left)
            if hashlib.sha256(row["sequence_R"].encode("ascii")).hexdigest() != row[
                "sequence_sha256_R"
            ]:
                raise ValueError("Right sequence SHA-256 differs")
            if hashlib.sha256(row["sequence_L"].encode("ascii")).hexdigest() != row[
                "sequence_sha256_L"
            ]:
                raise ValueError("Left sequence SHA-256 differs")
            weight = float(row["system_weight"])
            if not weight > 0.0:
                raise ValueError("System weights must be positive")
            selected.append({column: row[column] for column in OUTPUT_COLUMNS})
            structure_members.update((right, left))

    if input_count != arguments.expected_input_systems:
        raise ValueError("Curriculum system count differs")
    if len(selected) != arguments.expected_selected_systems:
        raise ValueError("Both-predicted system count differs")
    if len(structure_members) != arguments.expected_structure_members:
        raise ValueError("Predicted structure-member count differs")
    write_inventory(arguments.output_inventory, selected)
    payload: dict[str, object] = {
        "authorization": {
            "model_training": False,
            "structure_cache_materialization": True,
        },
        "claim_boundary": "Train-only predicted-structure inventory; no structure bytes, contact labels, validation rows, or test rows were read.",
        "counts": {
            "input_train_systems": input_count,
            "selected_both_predicted_systems": len(selected),
            "unique_predicted_structure_members": len(structure_members),
        },
        "format_version": "pinder_multimodal_inventory_result_v1",
        "inputs": {
            "curriculum_manifest_sha256": observed_manifest_sha,
            "protocol_sha256": arguments.protocol_sha256,
        },
        "outputs": {
            "inventory_sha256": sha256_file(arguments.output_inventory),
        },
        "status": "PASS_TRAIN_ONLY_PREDICTED_STRUCTURE_INVENTORY",
        "test_access": {
            "pinder_test_rows_read": False,
            "pinder_test_structures_read": False,
            "pinder_validation_rows_read": False,
        },
    }
    create_only(arguments.output_result, canonical_json_bytes(payload))
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--curriculum-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--expected-input-systems", type=int, required=True)
    parser.add_argument("--expected-selected-systems", type=int, required=True)
    parser.add_argument("--expected-structure-members", type=int, required=True)
    parser.add_argument("--output-inventory", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_arguments()), sort_keys=True))
