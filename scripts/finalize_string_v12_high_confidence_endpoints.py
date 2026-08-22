#!/usr/bin/env python3
"""Finalize unique endpoint IDs from the STRING high-confidence scale tier."""

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
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite endpoint receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--edge-receipt", type=Path, required=True)
    parser.add_argument("--endpoint-ids", type=Path, required=True)
    parser.add_argument("--endpoint-count", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()

    edges = arguments.edges.resolve()
    edge_receipt_path = arguments.edge_receipt.resolve()
    edge_receipt = json.loads(edge_receipt_path.read_text(encoding="utf-8"))
    if (
        edge_receipt.get("format_version")
        != "abpairfm_string_high_confidence_filter_v2"
        or edge_receipt.get("status")
        != "FILTERED_HIGH_CONFIDENCE_NOT_CLUSTERED_NOT_DECONTAMINATED"
        or edge_receipt.get("test_access") is not False
        or edge_receipt["output"]["sha256"] != sha256_file(edges)
    ):
        raise ValueError("STRING high-confidence edge evidence differs")
    count_text = arguments.endpoint_count.resolve().read_text(encoding="utf-8").strip()
    if not count_text.isdigit():
        raise ValueError("Malformed endpoint count")
    endpoint_count = int(count_text)
    edge_count = int(edge_receipt["counts"]["high_confidence_rows"])
    if not 1 <= endpoint_count <= 2 * edge_count:
        raise ValueError("Endpoint count is inconsistent with the edge count")
    endpoint_ids = arguments.endpoint_ids.resolve()
    if endpoint_ids.suffix != ".zst" or endpoint_ids.stat().st_size <= 0:
        raise ValueError("Endpoint ID artifact is not a non-empty Zstandard file")
    receipt = {
        "authorization": {
            "endpoint_sequence_extraction": True,
            "production_pretraining": False,
        },
        "claim_boundary": "Unique STRING identifiers from the high-confidence physical-link scale tier only. Sequence extraction, clustering, decontamination, training, and SOTA claims remain pending.",
        "counts": {
            "high_confidence_edges": edge_count,
            "unique_endpoint_ids": endpoint_count,
        },
        "format_version": "abpairfm_string_high_confidence_endpoints_v1",
        "input": {
            "edge_receipt_sha256": sha256_file(edge_receipt_path),
        },
        "output": {
            "bytes": endpoint_ids.stat().st_size,
            "name": endpoint_ids.name,
            "sha256": sha256_file(endpoint_ids),
        },
        "status": "ENDPOINT_IDS_EXTRACTED_NOT_SEQUENCED_NOT_CLUSTERED",
        "test_access": False,
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
