#!/usr/bin/env python3
"""Audit the public PPLM pair corpus without extracting archive members."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Iterable


MANIFESTS = (
    "trainset.PDB.txt",
    "trainset.STRING.txt",
    "validset.PDB.txt",
    "validset.PDB_hetero.txt",
    "validset.PDB_homo.txt",
    "validset.STRING.txt",
)
EXPECTED_ROOT = "PPLM_Dataset"
ASCII_RESIDUES = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class ArchiveAuditError(RuntimeError):
    """Report an archive, schema, or provenance violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        raise ArchiveAuditError("Cannot summarize an empty sequence collection")
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def _parse_fasta(payload: bytes, member_name: str) -> str:
    try:
        lines = [line.strip() for line in payload.decode("ascii").splitlines() if line.strip()]
    except UnicodeDecodeError as error:
        raise ArchiveAuditError(f"Non-ASCII FASTA member: {member_name}") from error
    if len(lines) < 2 or not lines[0].startswith(">"):
        raise ArchiveAuditError(f"Malformed FASTA member: {member_name}")
    sequence = "".join(lines[1:]).upper()
    if not sequence:
        raise ArchiveAuditError(f"Empty FASTA sequence: {member_name}")
    invalid = sorted(set(sequence) - ASCII_RESIDUES)
    if invalid:
        raise ArchiveAuditError(
            f"Unsupported non-letter FASTA symbols in {member_name}: {invalid}"
        )
    return sequence


def _pair_endpoints(
    pair_id: str, source: str, sequence_ids: set[str]
) -> tuple[str, str]:
    if source == "PDB":
        fields = pair_id.rsplit("_", 2)
        if len(fields) != 3 or not all(fields):
            raise ArchiveAuditError(f"Malformed PDB pair identifier: {pair_id}")
        return f"{fields[0]}_{fields[1]}", f"{fields[0]}_{fields[2]}"
    if source == "STRING":
        candidates = [
            (pair_id[:index], pair_id[index + 1 :])
            for index, character in enumerate(pair_id)
            if character == "-"
            and pair_id[:index] in sequence_ids
            and pair_id[index + 1 :] in sequence_ids
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates and "." in pair_id:
            taxonomy_id = pair_id.split(".", 1)[0]
            marker = f"-{taxonomy_id}."
            split_indices = [
                index for index in range(len(pair_id)) if pair_id.startswith(marker, index)
            ]
            if len(split_indices) == 1:
                index = split_indices[0]
                return pair_id[:index], pair_id[index + 1 :]
        if len(candidates) != 1:
            raise ArchiveAuditError(
                f"STRING pair identifier has {len(candidates)} inventory-valid splits: {pair_id}"
            )
    raise ArchiveAuditError(f"Unsupported manifest source: {source}")


def _canonical_pair_digest(first: str, second: str) -> bytes:
    ordered = sorted((first, second))
    return hashlib.blake2b(
        f"{ordered[0]}\0{ordered[1]}".encode("ascii"), digest_size=16
    ).digest()


def _manifest_summary(
    payload: bytes,
    *,
    name: str,
    sequence_ids: set[str],
    training_pair_digests: set[bytes],
    admissible_training_pair_digests: set[bytes],
) -> dict[str, Any]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ArchiveAuditError(f"Non-ASCII manifest: {name}") from error
    source = "PDB" if ".PDB" in name else "STRING"
    is_training = name.startswith("trainset.")
    cluster_count = 0
    pair_count = 0
    singleton_count = 0
    maximum_cluster_pairs = 0
    missing_endpoint_count = 0
    missing_endpoint_examples: list[str] = []
    pair_with_missing_endpoint_count = 0
    duplicate_pair_count = 0
    duplicate_pair_examples: list[str] = []
    validation_training_overlap_count = 0
    validation_training_overlap_examples: list[str] = []
    local_pair_digests: set[bytes] = set()
    for raw_line in text.splitlines():
        pairs = raw_line.split()
        if not pairs:
            continue
        cluster_count += 1
        pair_count += len(pairs)
        singleton_count += int(len(pairs) == 1)
        maximum_cluster_pairs = max(maximum_cluster_pairs, len(pairs))
        for pair_id in pairs:
            first, second = _pair_endpoints(pair_id, source, sequence_ids)
            pair_missing = False
            for endpoint in (first, second):
                if endpoint not in sequence_ids:
                    pair_missing = True
                    missing_endpoint_count += 1
                    if len(missing_endpoint_examples) < 10:
                        missing_endpoint_examples.append(f"{pair_id}|{endpoint}")
            pair_with_missing_endpoint_count += int(pair_missing)
            digest = _canonical_pair_digest(first, second)
            if digest in local_pair_digests:
                duplicate_pair_count += 1
                if len(duplicate_pair_examples) < 10:
                    duplicate_pair_examples.append(pair_id)
            local_pair_digests.add(digest)
            if is_training:
                training_pair_digests.add(digest)
                if not pair_missing:
                    admissible_training_pair_digests.add(digest)
            else:
                if digest in training_pair_digests:
                    validation_training_overlap_count += 1
                    if len(validation_training_overlap_examples) < 10:
                        validation_training_overlap_examples.append(pair_id)
    return {
        "cluster_count": cluster_count,
        "duplicate_pair_count": duplicate_pair_count,
        "duplicate_pair_examples": duplicate_pair_examples,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "maximum_cluster_pair_count": maximum_cluster_pairs,
        "missing_endpoint_count": missing_endpoint_count,
        "missing_endpoint_examples": missing_endpoint_examples,
        "pair_count": pair_count,
        "pair_with_missing_endpoint_count": pair_with_missing_endpoint_count,
        "singleton_cluster_count": singleton_count,
        "validation_training_exact_pair_overlap_count": validation_training_overlap_count,
        "validation_training_exact_pair_overlap_examples": validation_training_overlap_examples,
    }


def audit_archive(path: Path, *, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArchiveAuditError("Archive must be a regular file")
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise ArchiveAuditError(
            f"Archive size mismatch: observed {observed_size}, expected {expected_size}"
        )
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ArchiveAuditError("Archive SHA-256 differs from the frozen source anchor")

    category_file_counts = {"PDB_seqs": 0, "STRING_seqs": 0}
    fasta_lengths: list[int] = []
    unique_sequence_digests: set[bytes] = set()
    sequence_ids: set[str] = set()
    duplicate_sequence_id_count = 0
    ambiguous_residue_sequence_count = 0
    total_residues = 0
    manifests: dict[str, bytes] = {}
    root_files: set[str] = set()
    member_count = 0
    regular_file_count = 0
    directory_count = 0

    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            member_count += 1
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ArchiveAuditError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                directory_count += 1
                continue
            if not member.isfile():
                raise ArchiveAuditError(f"Non-regular archive member: {member.name}")
            regular_file_count += 1
            if not member_path.parts or member_path.parts[0] != EXPECTED_ROOT:
                raise ArchiveAuditError(f"Member outside expected root: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ArchiveAuditError(f"Cannot read archive member: {member.name}")
            payload = extracted.read()
            if len(payload) != member.size:
                raise ArchiveAuditError(f"Short archive member read: {member.name}")
            relative = member_path.relative_to(EXPECTED_ROOT)
            if len(relative.parts) == 1:
                root_files.add(relative.name)
                if relative.name in MANIFESTS:
                    manifests[relative.name] = payload
                continue
            category = relative.parts[0]
            if category not in category_file_counts or len(relative.parts) != 2:
                raise ArchiveAuditError(f"Unexpected archive member layout: {member.name}")
            if not relative.name.endswith(".fa"):
                raise ArchiveAuditError(f"Unexpected non-FASTA sequence member: {member.name}")
            category_file_counts[category] += 1
            sequence_id = relative.name[:-3]
            duplicate_sequence_id_count += int(sequence_id in sequence_ids)
            sequence_ids.add(sequence_id)
            sequence = _parse_fasta(payload, member.name)
            length = len(sequence)
            fasta_lengths.append(length)
            total_residues += length
            ambiguous_residue_sequence_count += int(bool(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")))
            unique_sequence_digests.add(hashlib.sha1(sequence.encode("ascii")).digest())

    expected_root_files = set(MANIFESTS) | {"README.txt"}
    if root_files != expected_root_files:
        raise ArchiveAuditError(
            f"Unexpected root files: observed {sorted(root_files)}, expected {sorted(expected_root_files)}"
        )
    if set(manifests) != set(MANIFESTS):
        raise ArchiveAuditError("One or more frozen pair manifests are missing")

    training_pair_digests: set[bytes] = set()
    admissible_training_pair_digests: set[bytes] = set()
    manifest_summaries: dict[str, dict[str, Any]] = {}
    # Training must be visited before validation for exact-overlap accounting.
    manifest_order: Iterable[str] = (
        "trainset.PDB.txt",
        "trainset.STRING.txt",
        "validset.PDB.txt",
        "validset.PDB_hetero.txt",
        "validset.PDB_homo.txt",
        "validset.STRING.txt",
    )
    for name in manifest_order:
        manifest_summaries[name] = _manifest_summary(
            manifests[name],
            name=name,
            sequence_ids=sequence_ids,
            training_pair_digests=training_pair_digests,
            admissible_training_pair_digests=admissible_training_pair_digests,
        )
    schema_defects = {
        "duplicate_pair_count": sum(
            item["duplicate_pair_count"] for item in manifest_summaries.values()
        ),
        "duplicate_sequence_id_count": duplicate_sequence_id_count,
        "missing_endpoint_count": sum(
            item["missing_endpoint_count"] for item in manifest_summaries.values()
        ),
        "pair_with_missing_endpoint_count": sum(
            item["pair_with_missing_endpoint_count"] for item in manifest_summaries.values()
        ),
        "validation_training_exact_pair_overlap_count": sum(
            item["validation_training_exact_pair_overlap_count"]
            for name, item in manifest_summaries.items()
            if name in {"validset.PDB.txt", "validset.STRING.txt"}
        ),
    }
    has_schema_defects = any(schema_defects.values())

    sorted_lengths = sorted(fasta_lengths)
    training_pair_count = sum(
        manifest_summaries[name]["pair_count"]
        for name in ("trainset.PDB.txt", "trainset.STRING.txt")
    )
    validation_pair_count = sum(
        manifest_summaries[name]["pair_count"]
        for name in ("validset.PDB.txt", "validset.STRING.txt")
    )
    admissible_training_pair_count = sum(
        manifest_summaries[name]["pair_count"]
        - manifest_summaries[name]["pair_with_missing_endpoint_count"]
        for name in ("trainset.PDB.txt", "trainset.STRING.txt")
    )
    return {
        "archive": {
            "member_count": member_count,
            "regular_file_count": regular_file_count,
            "sha256": observed_sha256,
            "size_bytes": observed_size,
            "top_level_directory_count": directory_count,
        },
        "authorization": {
            "normalization": True,
            "scientific_claim": False,
            "training": False,
        },
        "claim_boundary": (
            "Public PPLM archive integrity and internal-schema evidence only. "
            "Training remains fail-closed pending source-license admission, "
            "development decontamination, normalization, and corpus admission."
        ),
        "format_version": "pplm_public_archive_audit_v2",
        "manifests": manifest_summaries,
        "pair_totals": {
            "admissible_training_pair_count_after_missing_endpoint_exclusion": admissible_training_pair_count,
            "admissible_training_unique_exact_pair_count_after_missing_endpoint_exclusion_and_order_invariant_deduplication": len(admissible_training_pair_digests),
            "paper_claim_more_than_3_3m_not_reproduced_by_this_archive": training_pair_count <= 3_300_000,
            "training_exact_pair_count": training_pair_count,
            "training_unique_exact_pair_count": len(training_pair_digests),
            "validation_primary_pair_count": validation_pair_count,
        },
        "schema_defects": schema_defects,
        "sequences": {
            "ambiguous_residue_sequence_count": ambiguous_residue_sequence_count,
            "category_file_counts": category_file_counts,
            "duplicate_sequence_id_count": duplicate_sequence_id_count,
            "length_maximum": max(sorted_lengths),
            "length_minimum": min(sorted_lengths),
            "length_p50": _quantile(sorted_lengths, 0.50),
            "length_p95": _quantile(sorted_lengths, 0.95),
            "length_p99": _quantile(sorted_lengths, 0.99),
            "sequence_file_count": len(fasta_lengths),
            "total_residue_count": total_residues,
            "unique_exact_sequence_count": len(unique_sequence_digests),
        },
        "source_license_status": "unresolved_fail_closed_for_training",
        "status": (
            "MATERIALIZED_WITH_SCHEMA_DEFECTS_NOT_ADMITTED"
            if has_schema_defects
            else "MATERIALIZED_NOT_ADMITTED"
        ),
        "test_access": {
            "asep_test_accessed": False,
            "pinder_test_rows_used": False,
            "pinder_test_structure_bytes_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    data = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    payload = audit_archive(
        args.archive,
        expected_size=args.expected_size,
        expected_sha256=args.expected_sha256,
    )
    _write_create_only(args.output, payload)
    if payload["status"] == "MATERIALIZED_WITH_SCHEMA_DEFECTS_NOT_ADMITTED" and not args.allow_blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
