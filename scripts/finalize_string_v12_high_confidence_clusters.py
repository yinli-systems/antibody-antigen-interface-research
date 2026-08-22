#!/usr/bin/env python3
"""Finalize immutable MMseqs assets for the STRING high-confidence scale tier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite cluster receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-ids", type=Path, required=True)
    parser.add_argument("--endpoint-receipt", type=Path, required=True)
    parser.add_argument("--endpoint-fasta-receipt", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--membership-count", type=Path, required=True)
    parser.add_argument("--development-hits", type=Path, required=True)
    parser.add_argument("--mmseqs-version", required=True)
    parser.add_argument("--asset", type=Path, action="append", default=[])
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    endpoint_ids = arguments.endpoint_ids.resolve()
    endpoint_receipt_path = arguments.endpoint_receipt.resolve()
    endpoint_receipt = json.loads(endpoint_receipt_path.read_text(encoding="utf-8"))
    if (
        endpoint_receipt.get("format_version")
        != "abpairfm_string_high_confidence_endpoints_v1"
        or endpoint_receipt.get("status")
        != "ENDPOINT_IDS_EXTRACTED_NOT_SEQUENCED_NOT_CLUSTERED"
        or endpoint_receipt.get("test_access") is not False
        or endpoint_receipt["output"]["sha256"] != sha256_file(endpoint_ids)
    ):
        raise ValueError("STRING high-confidence endpoint evidence differs")

    fasta_receipt_path = arguments.endpoint_fasta_receipt.resolve()
    fasta_receipt = json.loads(fasta_receipt_path.read_text(encoding="utf-8"))
    expected_endpoints = int(endpoint_receipt["counts"]["unique_endpoint_ids"])
    if (
        fasta_receipt.get("status") != "ENDPOINT_SEQUENCES_EXTRACTED_NOT_CLUSTERED"
        or fasta_receipt.get("test_access") is not False
        or int(fasta_receipt["counts"]["endpoint_ids"]) != expected_endpoints
        or int(fasta_receipt["counts"]["selected_sequence_records"])
        != expected_endpoints
    ):
        raise ValueError("STRING endpoint FASTA evidence differs")
    membership_text = arguments.membership_count.resolve().read_text(
        encoding="utf-8"
    ).strip()
    if not membership_text.isdigit() or int(membership_text) != expected_endpoints:
        raise ValueError("MMseqs membership count differs from the endpoint inventory")
    clusters = arguments.clusters.resolve()
    hits = arguments.development_hits.resolve()
    if clusters.suffix != ".zst" or clusters.stat().st_size <= 0:
        raise ValueError("MMseqs cluster asset is missing")
    if not hits.is_file():
        raise ValueError("Development-homology hit asset is missing")
    assets = []
    for path in arguments.asset:
        resolved = path.resolve()
        assets.append(
            {
                "bytes": resolved.stat().st_size,
                "name": resolved.name,
                "sha256": sha256_file(resolved),
            }
        )
    receipt = {
        "authorization": {
            "development_decontamination": True,
            "pair_collapse": True,
            "production_pretraining": False,
        },
        "claim_boundary": "Authenticated high-confidence STRING endpoint sequences, 50%-identity MMseqs memberships, and development-homology hits only. Pair collapse, development decontamination, training, and SOTA claims remain pending.",
        "cluster": {
            "coverage": 0.8,
            "coverage_mode": 0,
            "membership_count": expected_endpoints,
            "minimum_sequence_identity": 0.5,
            "mmseqs_version": arguments.mmseqs_version,
        },
        "format_version": "abpairfm_string_high_confidence_cluster_assets_v1",
        "inputs": {
            "endpoint_fasta_receipt_sha256": sha256_file(fasta_receipt_path),
            "endpoint_receipt_sha256": sha256_file(endpoint_receipt_path),
        },
        "published_assets": assets,
        "status": "CLUSTERED_NOT_COLLAPSED_NOT_DECONTAMINATED",
        "test_access": False,
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
