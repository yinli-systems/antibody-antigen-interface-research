#!/usr/bin/env python3
"""Canonicalize and authenticate one MMseqs2 all-versus-all relation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta_ids(path: Path) -> set[str]:
    identifiers: set[str] = set()
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                identifier = line[1:].strip()
                if not identifier or identifier in identifiers:
                    raise ValueError("FASTA identifier is empty or duplicated")
                identifiers.add(identifier)
    if not identifiers:
        raise ValueError("FASTA has no identifiers")
    return identifiers


def fraction(value: str) -> float:
    result = float(value)
    if result > 1.0:
        result /= 100.0
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("MMseqs metric is not a finite fraction")
    return result


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(arguments.fasta) != arguments.fasta_sha256:
        raise ValueError("FASTA SHA mismatch")
    if sha256_file(arguments.mmseqs) != arguments.mmseqs_sha256:
        raise ValueError("MMseqs binary SHA mismatch")
    identifiers = read_fasta_ids(arguments.fasta)
    qualified: dict[tuple[str, str], tuple[float, float, float]] = {}
    observed: set[str] = set()
    raw_rows = 0
    with arguments.raw_tsv.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_rows += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"raw MMseqs row {line_number} does not have five fields")
            query, target = fields[:2]
            if query not in identifiers or target not in identifiers:
                raise ValueError("raw MMseqs relation references an unknown digest")
            observed.update((query, target))
            identity, qcov, tcov = map(fraction, fields[2:])
            if (
                identity + 1e-12 < arguments.identity_threshold
                or qcov + 1e-12 < arguments.coverage_threshold
                or tcov + 1e-12 < arguments.coverage_threshold
            ):
                continue
            key = (query, target)
            metrics = (identity, qcov, tcov)
            previous = qualified.get(key)
            if previous is None or metrics > previous:
                qualified[key] = metrics
    if observed != identifiers:
        raise ValueError("not every FASTA digest occurs in the raw all-versus-all relation")
    for identifier in identifiers:
        qualified.setdefault((identifier, identifier), (1.0, 1.0, 1.0))
    lines = [
        f"{query}\t{target}\t{identity:.12g}\t{qcov:.12g}\t{tcov:.12g}\n"
        for (query, target), (identity, qcov, tcov) in sorted(qualified.items())
    ]
    atomic_create(arguments.output_tsv, "".join(lines).encode("ascii"))
    receipt: dict[str, Any] = {
        "format_version": "pinder_antibody_s2_mmseqs_relation_receipt_v1",
        "status": "PASS_PINDER_ANTIBODY_S2_MMSEQS_RELATION",
        "role": arguments.role,
        "source": {
            "fasta": {
                "path": str(arguments.fasta.resolve()),
                "sha256": arguments.fasta_sha256,
                "sequence_count": len(identifiers),
            },
            "raw_tsv_sha256": sha256_file(arguments.raw_tsv),
            "mmseqs_binary_sha256": arguments.mmseqs_sha256,
            "mmseqs_version": arguments.mmseqs_version,
        },
        "thresholds": {
            "identity": arguments.identity_threshold,
            "reciprocal_coverage": arguments.coverage_threshold,
        },
        "raw_row_count": raw_rows,
        "canonical_row_count": len(lines),
        "canonical_relation": {
            "path": str(arguments.output_tsv.resolve()),
            "sha256": sha256_file(arguments.output_tsv),
        },
        "checks": {
            "all_fasta_digests_observed": True,
            "all_output_rows_meet_threshold": True,
            "self_rows_complete": True,
        },
        "test_access": False,
        "training_authorized": False,
    }
    atomic_create(
        arguments.receipt,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return receipt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("concatenated_antibody", "antigen"), required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--fasta-sha256", required=True)
    parser.add_argument("--raw-tsv", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--mmseqs-sha256", required=True)
    parser.add_argument("--mmseqs-version", required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.8)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
