#!/usr/bin/env python3
"""Audit sequence-only alignment from PINDER holo axes to predicted monomers."""

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
from statistics import median
from typing import Iterable, Mapping


INVENTORY_COLUMNS = {
    "id",
    "predicted_R_pdb",
    "predicted_L_pdb",
    "sequence_R",
    "sequence_L",
    "sequence_sha256_R",
    "sequence_sha256_L",
}
STRUCTURE_COLUMNS = {"member", "sequence", "sequence_sha256"}
OUTPUT_COLUMNS = (
    "member",
    "holo_sequence_sha256",
    "predicted_sequence_sha256",
    "holo_length",
    "predicted_length",
    "non_x_residues",
    "exact_mapped_residues",
    "exact_non_x_coverage",
    "ambiguous_optimum",
    "usable_structure_modality",
    "holo_to_predicted_zero_based",
)


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


def load_structure_sequences(path: Path) -> dict[str, tuple[str, str]]:
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


def load_associations(path: Path) -> dict[tuple[str, str], str]:
    associations: dict[tuple[str, str], str] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not INVENTORY_COLUMNS.issubset(reader.fieldnames):
            raise ValueError("Multimodal inventory schema differs")
        for row in reader:
            for side in ("R", "L"):
                member = row[f"predicted_{side}_pdb"]
                sequence = row[f"sequence_{side}"]
                sequence_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                if sequence_sha != row[f"sequence_sha256_{side}"]:
                    raise ValueError("Holo sequence SHA-256 differs")
                key = (member, sequence_sha)
                previous = associations.setdefault(key, sequence)
                if previous != sequence:
                    raise ValueError("A sequence SHA-256 collision was observed")
    if not associations:
        raise ValueError("No member-sequence associations were selected")
    return associations


def configure_aligner():
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -3.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    aligner.end_insertion_score = 0.0
    aligner.end_deletion_score = 0.0
    aligner.wildcard = "X"
    return aligner


def exact_mapping(alignment, predicted: str, holo: str) -> tuple[int, ...]:
    mapping = [-1] * len(holo)
    target_blocks, query_blocks = alignment.aligned
    if len(target_blocks) != len(query_blocks):
        raise ValueError("Alignment block count differs")
    for target_block, query_block in zip(target_blocks, query_blocks):
        target_start, target_end = (int(value) for value in target_block)
        query_start, query_end = (int(value) for value in query_block)
        if target_end - target_start != query_end - query_start:
            raise ValueError("Ungapped alignment block length differs")
        for target_index, query_index in zip(
            range(target_start, target_end), range(query_start, query_end)
        ):
            if holo[query_index] != "X" and holo[query_index] == predicted[target_index]:
                mapping[query_index] = target_index
    return tuple(mapping)


def align_association(aligner, predicted: str, holo: str) -> tuple[tuple[int, ...], bool]:
    alignments = iter(aligner.align(predicted, holo))
    try:
        first = next(alignments)
    except StopIteration as error:
        raise ValueError("PairwiseAligner returned no alignment") from error
    mapping = exact_mapping(first, predicted, holo)
    ambiguous = False
    for index, candidate in enumerate(alignments, start=1):
        if exact_mapping(candidate, predicted, holo) != mapping:
            ambiguous = True
            break
        if index >= 16:
            ambiguous = True
            break
    if ambiguous:
        mapping = tuple(-1 for _ in holo)
    return mapping, ambiguous


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty list")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def write_alignment_manifest(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
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


def count_both_usable_systems(
    inventory: Path, usable: Mapping[tuple[str, str], bool]
) -> tuple[int, int]:
    systems = 0
    both_usable = 0
    with gzip.open(inventory, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            systems += 1
            keys = [
                (row[f"predicted_{side}_pdb"], row[f"sequence_sha256_{side}"])
                for side in ("R", "L")
            ]
            both_usable += all(usable.get(key, False) for key in keys)
    return systems, both_usable


def run(arguments: argparse.Namespace) -> dict[str, object]:
    for path, expected, label in (
        (arguments.inventory, arguments.inventory_sha256, "inventory"),
        (arguments.structure_manifest, arguments.structure_manifest_sha256, "structure manifest"),
        (arguments.protocol_amendment, arguments.protocol_amendment_sha256, "protocol amendment"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 differs")
    if arguments.output_manifest.exists() or arguments.output_result.exists():
        raise FileExistsError("Alignment output already exists")

    structures = load_structure_sequences(arguments.structure_manifest)
    associations = load_associations(arguments.inventory)
    aligner = configure_aligner()
    rows: list[dict[str, object]] = []
    usable: dict[tuple[str, str], bool] = {}
    coverages: list[float] = []
    ambiguous_count = 0
    for member, holo_sha in sorted(associations):
        if member not in structures:
            raise ValueError(f"Predicted structure is absent: {member}")
        holo = associations[(member, holo_sha)]
        predicted, predicted_sha = structures[member]
        mapping, ambiguous = align_association(aligner, predicted, holo)
        non_x = sum(residue != "X" for residue in holo)
        exact = sum(index >= 0 for index in mapping)
        coverage = exact / non_x if non_x else 0.0
        is_usable = (
            not ambiguous
            and exact >= arguments.minimum_exact_mapped_residues
            and coverage >= arguments.minimum_exact_coverage
        )
        ambiguous_count += ambiguous
        coverages.append(coverage)
        usable[(member, holo_sha)] = is_usable
        rows.append(
            {
                "member": member,
                "holo_sequence_sha256": holo_sha,
                "predicted_sequence_sha256": predicted_sha,
                "holo_length": len(holo),
                "predicted_length": len(predicted),
                "non_x_residues": non_x,
                "exact_mapped_residues": exact,
                "exact_non_x_coverage": f"{coverage:.12f}",
                "ambiguous_optimum": str(ambiguous).lower(),
                "usable_structure_modality": str(is_usable).lower(),
                "holo_to_predicted_zero_based": ",".join(str(index) for index in mapping),
            }
        )
    write_alignment_manifest(arguments.output_manifest, rows)
    systems, both_usable = count_both_usable_systems(arguments.inventory, usable)
    median_coverage = median(coverages)
    gate_pass = (
        both_usable >= arguments.minimum_both_usable_systems
        and median_coverage >= arguments.minimum_median_coverage
    )
    payload: dict[str, object] = {
        "authorization": {
            "model_training": False,
            "structure_feature_materialization": gate_pass,
        },
        "claim_boundary": "Train-only amino-acid alignment audit between holo sequence axes and independently predicted full-UniProt monomers; no bound coordinates, resolved flags, validation rows, or test rows were read.",
        "counts": {
            "ambiguous_associations": ambiguous_count,
            "both_endpoint_usable_systems": both_usable,
            "systems": systems,
            "unique_member_holo_sequence_associations": len(rows),
            "usable_associations": sum(usable.values()),
        },
        "coverage": {
            "median_exact_non_x": median_coverage,
            "p05_exact_non_x": percentile(coverages, 0.05),
            "minimum": min(coverages),
            "maximum": max(coverages),
        },
        "format_version": "pinder_holo_predicted_sequence_alignment_result_v1",
        "inputs": {
            "inventory_sha256": arguments.inventory_sha256,
            "protocol_amendment_sha256": arguments.protocol_amendment_sha256,
            "structure_manifest_sha256": arguments.structure_manifest_sha256,
        },
        "outputs": {"alignment_manifest_sha256": sha256_file(arguments.output_manifest)},
        "status": (
            "PASS_SEQUENCE_ONLY_ALIGNMENT_GATE_PENDING_PARENT"
            if gate_pass
            else "FAIL_SEQUENCE_ONLY_ALIGNMENT_GATE"
        ),
        "test_access": {
            "bound_coordinates_read": False,
            "pinder_test_rows_read": False,
            "pinder_validation_rows_read": False,
            "resolved_residue_flags_read": False,
        },
        "thresholds": {
            "minimum_both_endpoint_usable_systems": arguments.minimum_both_usable_systems,
            "minimum_exact_coverage": arguments.minimum_exact_coverage,
            "minimum_exact_mapped_residues": arguments.minimum_exact_mapped_residues,
            "minimum_median_coverage": arguments.minimum_median_coverage,
        },
    }
    create_only(arguments.output_result, canonical_json_bytes(payload))
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--protocol-amendment", type=Path, required=True)
    parser.add_argument("--protocol-amendment-sha256", required=True)
    parser.add_argument("--minimum-exact-mapped-residues", type=int, default=8)
    parser.add_argument("--minimum-exact-coverage", type=float, default=0.5)
    parser.add_argument("--minimum-both-usable-systems", type=int, default=250000)
    parser.add_argument("--minimum-median-coverage", type=float, default=0.9)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_arguments()), sort_keys=True))
