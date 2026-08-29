#!/usr/bin/env python3
"""Audit only the training member of the authenticated AbCDR OAS archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile


TRAIN_MEMBER = "OAS_paired_set/train.csv"
SEALED_MEMBERS = {
    "OAS_paired_set/validation.csv",
    "OAS_paired_set/test.csv",
}
EXPECTED_COLUMNS = (
    "sequence_alignment_aa_heavy",
    "sequence_alignment_aa_light",
    "cdr1_aa_heavy",
    "cdr2_aa_heavy",
    "cdr3_aa_heavy",
    "cdr1_aa_light",
    "cdr2_aa_light",
    "cdr3_aa_light",
    "cdr1_aa_heavy_inds",
    "cdr2_aa_heavy_inds",
    "cdr3_aa_heavy_inds",
    "cdr1_aa_light_inds",
    "cdr2_aa_light_inds",
    "cdr3_aa_light_inds",
)
RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--provider-metadata", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--target-native-pairs", type=int, default=5_500_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode("ascii"), digest_size=16).digest()


def _quantile(values: list[int], q: float) -> int:
    if not values:
        raise ValueError("Cannot summarize empty lengths")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _publish_create_only(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite audit receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _provider_evidence(path: Path, archive_name: str) -> dict[str, object]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("id") != 18762309:
        raise RuntimeError("Provider record ID differs from frozen AbCDR source")
    license_id = metadata.get("metadata", {}).get("license", {}).get("id")
    if license_id != "cc-by-4.0":
        raise RuntimeError("Provider license is not CC-BY-4.0")
    matches = [item for item in metadata.get("files", []) if item.get("key") == archive_name]
    if len(matches) != 1:
        raise RuntimeError("Provider metadata does not identify exactly one archive")
    return {
        "doi": metadata.get("doi"),
        "license": license_id,
        "provider_checksum": matches[0].get("checksum"),
        "provider_size_bytes": matches[0].get("size"),
        "record_id": metadata.get("id"),
    }


def audit(arguments: argparse.Namespace) -> dict[str, object]:
    archive = arguments.archive.resolve()
    if archive.is_symlink() or not archive.is_file():
        raise RuntimeError("Archive must be a regular file")
    if archive.stat().st_size != arguments.expected_size:
        raise RuntimeError("Archive size differs from frozen anchor")
    observed_sha256 = _sha256(archive)
    if observed_sha256 != arguments.expected_sha256:
        raise RuntimeError("Archive SHA-256 differs from frozen anchor")
    provider = _provider_evidence(arguments.provider_metadata.resolve(), archive.name)
    if provider["provider_size_bytes"] != arguments.expected_size:
        raise RuntimeError("Provider and frozen archive sizes disagree")

    exact_pairs: set[bytes] = set()
    exact_heavy: set[bytes] = set()
    exact_light: set[bytes] = set()
    heavy_lengths: list[int] = []
    light_lengths: list[int] = []
    invalid_residue_rows = 0
    invalid_residue_examples: list[int] = []
    malformed_cdr_rows = 0
    malformed_cdr_examples: list[int] = []
    ambiguous_sequence_rows = 0
    row_count = 0

    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        names = {item.filename for item in members}
        unsafe = [
            item.filename
            for item in members
            if item.filename.startswith(("/", "\\")) or ".." in Path(item.filename).parts
        ]
        if unsafe:
            raise RuntimeError(f"Archive contains unsafe paths: {unsafe[:3]}")
        if TRAIN_MEMBER not in names or not SEALED_MEMBERS.issubset(names):
            raise RuntimeError("Archive split members differ from expected layout")
        # Deliberately open exactly one member. Validation and test contents stay sealed.
        with handle.open(TRAIN_MEMBER, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                raise RuntimeError("Training CSV columns differ from frozen schema")
            for row_count, row in enumerate(reader, start=1):
                heavy = row["sequence_alignment_aa_heavy"].strip().upper()
                light = row["sequence_alignment_aa_light"].strip().upper()
                heavy_lengths.append(len(heavy))
                light_lengths.append(len(light))
                invalid = (set(heavy) | set(light)) - RESIDUES
                if not heavy or not light or invalid:
                    invalid_residue_rows += 1
                    if len(invalid_residue_examples) < 10:
                        invalid_residue_examples.append(row_count)
                if "X" in heavy or "X" in light:
                    ambiguous_sequence_rows += 1
                exact_pairs.add(_digest(heavy + "\0" + light))
                exact_heavy.add(_digest(heavy))
                exact_light.add(_digest(light))
                cdr_valid = True
                for locus, sequence in (("heavy", heavy), ("light", light)):
                    for number in (1, 2, 3):
                        cdr = row[f"cdr{number}_aa_{locus}"].strip().upper()
                        try:
                            indices = json.loads(row[f"cdr{number}_aa_{locus}_inds"])
                        except json.JSONDecodeError:
                            cdr_valid = False
                            continue
                        if (
                            not isinstance(indices, list)
                            or len(indices) != 2
                            or any(type(value) is not int for value in indices)
                        ):
                            cdr_valid = False
                            continue
                        start, end = indices
                        if start < 0 or end <= start or end > len(sequence):
                            cdr_valid = False
                        elif sequence[start:end] != cdr:
                            cdr_valid = False
                if not cdr_valid:
                    malformed_cdr_rows += 1
                    if len(malformed_cdr_examples) < 10:
                        malformed_cdr_examples.append(row_count)

    unique_pairs = len(exact_pairs)
    duplicate_pairs = row_count - unique_pairs
    deficits = {
        "native_pair_target_deficit": max(0, arguments.target_native_pairs - unique_pairs),
        "row_level_donor_study_clone_provenance_absent": True,
        "development_decontamination_pending": True,
    }
    admitted = (
        invalid_residue_rows == 0
        and malformed_cdr_rows == 0
        and deficits["native_pair_target_deficit"] == 0
        and not deficits["row_level_donor_study_clone_provenance_absent"]
        and not deficits["development_decontamination_pending"]
    )
    return {
        "archive": {
            "sha256": observed_sha256,
            "size_bytes": arguments.expected_size,
        },
        "authorization": {
            "normalization": invalid_residue_rows == 0 and malformed_cdr_rows == 0,
            "stage_b_training": admitted,
        },
        "claim_boundary": (
            "Authenticated, train-member-only AbCDR/OAS inventory and sequence/CDR "
            "quality evidence. Validation and test contents were not opened."
        ),
        "deficits": deficits,
        "format_version": "abcdr_oas_paired_train_audit_v1",
        "provider": provider,
        "sealed_members_present_but_not_opened": sorted(SEALED_MEMBERS),
        "status": "ADMITTED" if admitted else "MATERIALIZED_NOT_SUFFICIENT_FOR_STAGE_B",
        "test_access": {
            "abcdr_test_member_opened": False,
            "abcdr_validation_member_opened": False,
            "asep_test_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "training_inventory": {
            "ambiguous_sequence_row_count": ambiguous_sequence_rows,
            "duplicate_exact_pair_count": duplicate_pairs,
            "heavy_length": {
                "maximum": max(heavy_lengths),
                "minimum": min(heavy_lengths),
                "p50": _quantile(heavy_lengths, 0.5),
                "p95": _quantile(heavy_lengths, 0.95),
                "p99": _quantile(heavy_lengths, 0.99),
            },
            "invalid_residue_row_count": invalid_residue_rows,
            "invalid_residue_row_examples": invalid_residue_examples,
            "light_length": {
                "maximum": max(light_lengths),
                "minimum": min(light_lengths),
                "p50": _quantile(light_lengths, 0.5),
                "p95": _quantile(light_lengths, 0.95),
                "p99": _quantile(light_lengths, 0.99),
            },
            "malformed_cdr_row_count": malformed_cdr_rows,
            "malformed_cdr_row_examples": malformed_cdr_examples,
            "row_count": row_count,
            "unique_exact_heavy_count": len(exact_heavy),
            "unique_exact_light_count": len(exact_light),
            "unique_exact_pair_count": unique_pairs,
        },
        "training_member": TRAIN_MEMBER,
    }


def main() -> None:
    arguments = _arguments()
    receipt = audit(arguments)
    _publish_create_only(arguments.output, receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if receipt["status"] != "ADMITTED" and not arguments.allow_blocked:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
