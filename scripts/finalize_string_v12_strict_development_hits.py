#!/usr/bin/env python3
"""Authenticate the 30%-identity STRING-to-development homology hit asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterator


MINIMUM_IDENTITY = 0.30
MINIMUM_QUERY_COVERAGE = 0.80
MINIMUM_TARGET_COVERAGE = 0.80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_lines(path: Path) -> Iterator[str]:
    if path.suffix != ".zst":
        with path.open("rt", encoding="utf-8", newline="") as handle:
            yield from handle
        return
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        yield from process.stdout
    finally:
        process.stdout.close()
        error = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")


def publish_create_only(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite strict-hit receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def audit_hits(path: Path) -> dict[str, int]:
    rows = 0
    query_ids: set[str] = set()
    target_ids: set[str] = set()
    for line_number, raw_line in enumerate(text_lines(path), start=1):
        fields = raw_line.rstrip("\n").split("\t")
        if len(fields) != 8 or not fields[0] or not fields[1]:
            raise ValueError(f"Malformed strict development-hit row {line_number}")
        identity = float(fields[2])
        query_coverage = float(fields[4])
        target_coverage = float(fields[5])
        if (
            identity + 1e-12 < MINIMUM_IDENTITY
            or query_coverage + 1e-12 < MINIMUM_QUERY_COVERAGE
            or target_coverage + 1e-12 < MINIMUM_TARGET_COVERAGE
        ):
            raise ValueError(f"Strict development-hit row {line_number} violates the gate")
        rows += 1
        query_ids.add(fields[0])
        target_ids.add(fields[1])
    if rows == 0:
        raise ValueError("Strict development-hit asset is empty")
    return {
        "hit_rows": rows,
        "unique_development_queries_with_hits": len(query_ids),
        "unique_string_endpoints_with_hits": len(target_ids),
    }


def build_receipt(arguments: argparse.Namespace) -> dict[str, object]:
    endpoint_fasta = arguments.endpoint_fasta.resolve()
    cluster_receipt_path = arguments.cluster_receipt.resolve()
    cluster_receipt = json.loads(cluster_receipt_path.read_text(encoding="utf-8"))
    published = {
        item["name"]: item for item in cluster_receipt.get("published_assets", [])
    }
    if (
        cluster_receipt.get("format_version")
        != "abpairfm_string_high_confidence_cluster_assets_v1"
        or cluster_receipt.get("status")
        != "CLUSTERED_NOT_COLLAPSED_NOT_DECONTAMINATED"
        or cluster_receipt.get("test_access") is not False
        or published.get(endpoint_fasta.name, {}).get("sha256")
        != sha256_file(endpoint_fasta)
    ):
        raise ValueError("STRING endpoint-cluster evidence differs")

    development_fasta = arguments.development_fasta.resolve()
    development_receipt_path = arguments.development_receipt.resolve()
    development_receipt = json.loads(
        development_receipt_path.read_text(encoding="utf-8")
    )
    if (
        development_receipt.get("format_version")
        != "abpairfm_development_sequence_panel_v1"
        or development_receipt.get("status") != "FROZEN_VALIDATION_SEQUENCE_PANEL"
        or development_receipt.get("test_access") is not False
        or development_receipt.get("output_fasta", {}).get("sha256")
        != sha256_file(development_fasta)
    ):
        raise ValueError("Frozen development-sequence evidence differs")

    hits = arguments.hits.resolve()
    if hits.suffix != ".zst" or not hits.is_file() or hits.stat().st_size <= 0:
        raise ValueError("Strict development-hit asset must be a nonempty zstd file")
    counts = audit_hits(hits)
    return {
        "authorization": {
            "pair_collapse_with_strict_hits": True,
            "production_pretraining": False,
        },
        "claim_boundary": "Thirty-percent-identity, bidirectional-80%-coverage STRING endpoint hits against frozen AsEP/SAbDab2 development sequences. This asset authorizes strict pair collapse only; production training, development evaluation, test access, and SOTA claims remain closed.",
        "counts": counts,
        "format_version": "abpairfm_string_strict_development_hits_v1",
        "inputs": {
            "cluster_receipt_sha256": sha256_file(cluster_receipt_path),
            "development_fasta_sha256": sha256_file(development_fasta),
            "development_receipt_sha256": sha256_file(development_receipt_path),
            "endpoint_fasta_sha256": sha256_file(endpoint_fasta),
        },
        "mmseqs": {
            "alignment_mode": 3,
            "coverage_mode": 0,
            "minimum_bidirectional_coverage": MINIMUM_QUERY_COVERAGE,
            "minimum_sequence_identity": MINIMUM_IDENTITY,
            "version": arguments.mmseqs_version,
        },
        "output": {
            "bytes": hits.stat().st_size,
            "name": arguments.output_name or hits.name,
            "sha256": sha256_file(hits),
        },
        "status": "STRICT_DEVELOPMENT_HITS_READY_NOT_COLLAPSED",
        "test_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-fasta", type=Path, required=True)
    parser.add_argument("--cluster-receipt", type=Path, required=True)
    parser.add_argument("--development-fasta", type=Path, required=True)
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--output-name")
    parser.add_argument("--mmseqs-version", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = build_receipt(arguments)
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
