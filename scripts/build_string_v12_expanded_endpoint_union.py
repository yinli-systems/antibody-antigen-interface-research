#!/usr/bin/env python3
"""Freeze the endpoint union for expanded STRING v12 clustering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


EXPECTED_HEADER = [
    "protein1", "protein2", "homology", "experiments", "experiments_transferred",
    "database", "database_transferred", "textmining", "textmining_transferred", "combined_score",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def endpoints(path: Path) -> tuple[set[str], int]:
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    values: set[str] = set()
    rows = 0
    try:
        header = process.stdout.readline().rstrip("\n").split()
        if header != EXPECTED_HEADER:
            raise ValueError("Unexpected STRING endpoint-union header")
        for line in process.stdout:
            row = line.rstrip("\n").split()
            if len(row) != len(EXPECTED_HEADER):
                raise ValueError("Malformed STRING endpoint-union edge")
            values.update(row[:2])
            rows += 1
    finally:
        process.stdout.close()
        error = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")
    return values, rows


def publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise FileExistsError(f"Refusing to overwrite expanded STRING endpoint evidence: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--direct-receipt", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, required=True)
    parser.add_argument("--supplemental-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    direct = arguments.direct.resolve()
    direct_receipt_path = arguments.direct_receipt.resolve()
    direct_receipt = json.loads(direct_receipt_path.read_text(encoding="utf-8"))
    supplemental = arguments.supplemental.resolve()
    supplemental_receipt_path = arguments.supplemental_receipt.resolve()
    supplemental_receipt = json.loads(supplemental_receipt_path.read_text(encoding="utf-8"))
    if (
        direct_receipt.get("status") != "FILTERED_NOT_CLUSTERED_NOT_DECONTAMINATED"
        or direct_receipt.get("test_access") is not False
        or sha256_file(direct) != direct_receipt["output"]["sha256"]
    ):
        raise ValueError("STRING direct evidence differs")
    if (
        supplemental_receipt.get("status") != "FILTERED_UNIVERSE_NOT_SELECTED_NOT_CLUSTERED"
        or supplemental_receipt.get("test_access") is not False
        or sha256_file(supplemental) != supplemental_receipt["output"]["sha256"]
    ):
        raise ValueError("STRING supplemental evidence differs")
    direct_ids, direct_rows = endpoints(direct)
    supplemental_ids, supplemental_rows = endpoints(supplemental)
    if direct_rows != int(direct_receipt["counts"]["direct_evidence_kept_rows"]):
        raise ValueError("STRING direct edge count differs")
    if supplemental_rows != int(supplemental_receipt["counts"]["candidate_rows"]):
        raise ValueError("STRING supplemental edge count differs")
    union = direct_ids | supplemental_ids
    output = arguments.output.resolve()
    publish_create_only(output, "".join(f"{value}\n" for value in sorted(union)).encode("utf-8"))
    receipt = {
        "authorization": {"sequence_extraction": True, "stage_a_training": False},
        "claim_boundary": "Authenticated endpoint union for 50%-identity clustering of STRING tier-1 and preregistered supplemental edges. It does not select a supplemental threshold or authorize training.",
        "counts": {
            "direct_edge_rows": direct_rows,
            "direct_unique_endpoints": len(direct_ids),
            "expanded_unique_endpoints": len(union),
            "supplemental_added_endpoints": len(union - direct_ids),
            "supplemental_edge_rows": supplemental_rows,
            "supplemental_unique_endpoints": len(supplemental_ids),
        },
        "format_version": "abpairfm_string_expanded_endpoint_union_v1",
        "inputs": {
            "direct_receipt_sha256": sha256_file(direct_receipt_path),
            "supplemental_receipt_sha256": sha256_file(supplemental_receipt_path),
        },
        "output": {"bytes": output.stat().st_size, "sha256": sha256_file(output)},
        "status": "EXPANDED_ENDPOINT_UNION_AUTHENTICATED_NOT_CLUSTERED",
        "test_access": False,
    }
    publish_create_only(
        arguments.receipt.resolve(),
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
