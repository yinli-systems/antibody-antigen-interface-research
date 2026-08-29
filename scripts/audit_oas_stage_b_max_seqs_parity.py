#!/usr/bin/env python3
"""Audit whether a capped MMseqs target list changes OAS pair exclusion."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import tempfile

try:
    from scripts.decontaminate_oas_stage_b_pairs import (
        checked_hash,
        checked_sequence,
        excluded_pair_hashes,
        load_development_receipt,
        load_source_receipt,
        publish_create_only,
        rows,
        sha256_file,
    )
except ModuleNotFoundError:
    from decontaminate_oas_stage_b_pairs import (
        checked_hash,
        checked_sequence,
        excluded_pair_hashes,
        load_development_receipt,
        load_source_receipt,
        publish_create_only,
        rows,
        sha256_file,
    )


def selected(row_hash: str, modulus: int, remainder: int) -> bool:
    if modulus < 2 or not 0 <= remainder < modulus:
        raise ValueError("Invalid deterministic sample modulus or remainder")
    digest = checked_hash(row_hash, "row_hash")
    return int(digest[:16], 16) % modulus == remainder


def export_sample_fasta(
    input_path: Path,
    output_path: Path,
    *,
    expected_rows: int,
    modulus: int,
    remainder: int,
    minimum_pairs: int,
) -> dict[str, int]:
    observed = 0
    sampled = 0
    with output_path.open("wt", encoding="ascii", newline="\n") as handle:
        for row in rows(input_path):
            observed += 1
            row_hash = checked_hash(row["row_hash"], "row_hash")
            if not selected(row_hash, modulus, remainder):
                continue
            heavy = checked_sequence(row["heavy_sequence"], f"{row_hash} heavy")
            light = checked_sequence(row["light_sequence"], f"{row_hash} light")
            handle.write(f">{row_hash}|heavy\n{heavy}\n>{row_hash}|light\n{light}\n")
            sampled += 1
    if observed != expected_rows:
        raise ValueError(f"OAS Stage-B row count differs: {observed}")
    if sampled < minimum_pairs:
        raise ValueError(
            f"Deterministic parity sample is too small: {sampled} < {minimum_pairs}"
        )
    return {"source_pairs": observed, "sampled_pairs": sampled}


def _publish_gzip(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with source.open("rb") as source_handle, temporary_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                for block in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
                    compressed.write(block)
        publish_create_only(output, temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def compare_exclusions(
    capped_hits: Path,
    exhaustive_hits: Path,
    capped_hits_output: Path,
    exhaustive_hits_output: Path,
    receipt_path: Path,
    *,
    input_path: Path,
    source_receipt_path: Path,
    development_fasta: Path,
    development_receipt_path: Path,
    sample_fasta: Path,
    source_pairs: int,
    sampled_pairs: int,
    modulus: int,
    remainder: int,
    capped_maximum_target_candidates: int,
    exhaustive_maximum_target_candidates: int,
    mmseqs_version: str,
) -> dict:
    load_source_receipt(input_path, source_receipt_path)
    load_development_receipt(development_fasta, development_receipt_path)
    capped_pairs, capped_hit_count = excluded_pair_hashes(
        capped_hits, oas_column="query"
    )
    exhaustive_pairs, exhaustive_hit_count = excluded_pair_hashes(
        exhaustive_hits, oas_column="query"
    )
    missed_by_cap = exhaustive_pairs - capped_pairs
    cap_only = capped_pairs - exhaustive_pairs
    status = (
        "PASS_MAX_SEQS_PAIR_EXCLUSION_PARITY"
        if not missed_by_cap and not cap_only
        else "FAIL_MAX_SEQS_PAIR_EXCLUSION_PARITY"
    )
    _publish_gzip(capped_hits, capped_hits_output)
    _publish_gzip(exhaustive_hits, exhaustive_hits_output)
    receipt = {
        "authorization": {
            "development_decontamination_audit": True,
            "stage_b_training": False,
            "test_access": False,
        },
        "claim_boundary": "Deterministic development-only canary of pair-level exclusion parity. A pass validates the 64-candidate cap only for this frozen sample and exact MMseqs configuration.",
        "comparison": {
            "cap_only_pairs": len(cap_only),
            "capped_excluded_pairs": len(capped_pairs),
            "capped_hits": capped_hit_count,
            "capped_maximum_target_candidates": capped_maximum_target_candidates,
            "exhaustive_excluded_pairs": len(exhaustive_pairs),
            "exhaustive_hits": exhaustive_hit_count,
            "exhaustive_maximum_target_candidates": exhaustive_maximum_target_candidates,
            "missed_by_cap_pairs": len(missed_by_cap),
        },
        "format_version": "abpairfm_oas_stage_b_max_seqs_parity_v1",
        "inputs": {
            "development_fasta_sha256": sha256_file(development_fasta),
            "development_receipt_sha256": sha256_file(development_receipt_path),
            "sample_fasta_sha256": sha256_file(sample_fasta),
            "source_pairs_sha256": sha256_file(input_path),
            "source_receipt_sha256": sha256_file(source_receipt_path),
        },
        "mmseqs": {
            "coverage": 0.8,
            "coverage_mode": 0,
            "minimum_sequence_identity": 0.5,
            "sequence_masking": False,
            "version": mmseqs_version.strip(),
        },
        "outputs": {
            "capped_hits": {
                "bytes": capped_hits_output.stat().st_size,
                "sha256": sha256_file(capped_hits_output),
            },
            "exhaustive_hits": {
                "bytes": exhaustive_hits_output.stat().st_size,
                "sha256": sha256_file(exhaustive_hits_output),
            },
        },
        "sample": {
            "modulus": modulus,
            "remainder": remainder,
            "sampled_pairs": sampled_pairs,
            "source_pairs": source_pairs,
        },
        "status": status,
        "test_access": False,
    }
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{receipt_path.name}.", dir=receipt_path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        publish_create_only(receipt_path, temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample-fasta")
    sample.add_argument("--input", type=Path, required=True)
    sample.add_argument("--source-receipt", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--modulus", type=int, required=True)
    sample.add_argument("--remainder", type=int, required=True)
    sample.add_argument("--minimum-pairs", type=int, required=True)
    compare = subparsers.add_parser("compare")
    for name in (
        "input",
        "source-receipt",
        "development-fasta",
        "development-receipt",
        "sample-fasta",
        "capped-hits",
        "exhaustive-hits",
        "capped-hits-output",
        "exhaustive-hits-output",
        "receipt",
    ):
        compare.add_argument(f"--{name}", type=Path, required=True)
    compare.add_argument("--source-pairs", type=int, required=True)
    compare.add_argument("--sampled-pairs", type=int, required=True)
    compare.add_argument("--modulus", type=int, required=True)
    compare.add_argument("--remainder", type=int, required=True)
    compare.add_argument("--capped-maximum-target-candidates", type=int, required=True)
    compare.add_argument(
        "--exhaustive-maximum-target-candidates", type=int, required=True
    )
    compare.add_argument("--mmseqs-version", required=True)
    arguments = parser.parse_args()
    if arguments.command == "sample-fasta":
        source_receipt = load_source_receipt(arguments.input, arguments.source_receipt)
        result = export_sample_fasta(
            arguments.input,
            arguments.output,
            expected_rows=source_receipt["counts"]["unique_exact_pairs"],
            modulus=arguments.modulus,
            remainder=arguments.remainder,
            minimum_pairs=arguments.minimum_pairs,
        )
        print(json.dumps(result, sort_keys=True))
        return
    receipt = compare_exclusions(
        arguments.capped_hits,
        arguments.exhaustive_hits,
        arguments.capped_hits_output,
        arguments.exhaustive_hits_output,
        arguments.receipt,
        input_path=arguments.input,
        source_receipt_path=arguments.source_receipt,
        development_fasta=arguments.development_fasta,
        development_receipt_path=arguments.development_receipt,
        sample_fasta=arguments.sample_fasta,
        source_pairs=arguments.source_pairs,
        sampled_pairs=arguments.sampled_pairs,
        modulus=arguments.modulus,
        remainder=arguments.remainder,
        capped_maximum_target_candidates=arguments.capped_maximum_target_candidates,
        exhaustive_maximum_target_candidates=(
            arguments.exhaustive_maximum_target_candidates
        ),
        mmseqs_version=arguments.mmseqs_version,
    )
    print(json.dumps(receipt, sort_keys=True))
    if not receipt["status"].startswith("PASS_"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
