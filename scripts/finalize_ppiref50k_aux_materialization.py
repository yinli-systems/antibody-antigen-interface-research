#!/usr/bin/env python3
"""Freeze the PPIRef50K split and matching wwPDB SEQRES snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


PINNED_PPIREF_COMMIT = "a2a5b6d01f958acac39418bda1a5a9e3b4d71b53"
PINNED_SPLIT_SHA256 = "a77239af16efd8c4a83353405ba2e6670720270955d9831febacbcdbad92ce57"
PINNED_SPLIT_COUNT = 51_755


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_create_only(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
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
        raise FileExistsError(f"Refusing to overwrite materialization receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_final_headers(path: Path) -> dict[str, str]:
    blocks = path.read_text(encoding="iso-8859-1").replace("\r\n", "\n").split("\n\n")
    candidates = [block for block in blocks if block.startswith("HTTP/")]
    if not candidates:
        raise ValueError(f"No HTTP response found in {path}")
    lines = candidates[-1].splitlines()
    if " 200 " not in f" {lines[0]} ":
        raise ValueError(f"Final HTTP response was not 200: {lines[0]}")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return headers


def validate_split(path: Path) -> dict[str, object]:
    if sha256_file(path) != PINNED_SPLIT_SHA256:
        raise ValueError("PPIRef50K split SHA-256 differs from the pinned commit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source") != "data/ppiref/ppi_6A":
        raise ValueError("PPIRef50K split source differs")
    folds = payload.get("folds")
    if not isinstance(folds, dict) or set(folds) != {"whole"}:
        raise ValueError("PPIRef50K split fold schema differs")
    identifiers = folds["whole"]
    if not isinstance(identifiers, list) or len(identifiers) != PINNED_SPLIT_COUNT:
        raise ValueError("PPIRef50K split count differs")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("PPIRef50K split contains duplicate interface ids")
    return {"interface_ids": len(identifiers), "unique_interface_ids": len(set(identifiers))}


def validate_seqres(path: Path) -> dict[str, int]:
    records = 0
    residues = 0
    current = False
    with gzip.open(path, "rt", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                records += 1
                current = True
            else:
                if not current or not line.strip().isalpha():
                    raise ValueError("Malformed wwPDB SEQRES FASTA")
                residues += len(line.strip())
    if records <= 0 or residues <= 0:
        raise ValueError("wwPDB SEQRES snapshot is empty")
    return {"records": records, "residues": residues}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppiref-split", type=Path, required=True)
    parser.add_argument("--ppiref-headers", type=Path, required=True)
    parser.add_argument("--seqres", type=Path, required=True)
    parser.add_argument("--seqres-headers", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument(
        "--transport-mode",
        choices=("compute_node_https", "authenticated_local_https_scp_recovery"),
        default="compute_node_https",
    )
    parser.add_argument("--failed-job-id", type=int)
    arguments = parser.parse_args()

    split_path = arguments.ppiref_split.resolve()
    seqres_path = arguments.seqres.resolve()
    split_headers = parse_final_headers(arguments.ppiref_headers.resolve())
    seqres_headers = parse_final_headers(arguments.seqres_headers.resolve())
    split_counts = validate_split(split_path)
    seqres_counts = validate_seqres(seqres_path)
    declared_length = seqres_headers.get("content-length")
    if declared_length is not None and int(declared_length) != seqres_path.stat().st_size:
        raise ValueError("wwPDB SEQRES Content-Length differs from the downloaded bytes")

    receipt = {
        "authorization": {"normalization": True, "stage_a_training": False},
        "claim_boundary": "Authenticated PPIRef50K interface-id split and wwPDB full-chain SEQRES snapshot only; no normalized examples, decontamination, training readiness, model quality, or test claim.",
        "execution": {
            "allocated_gpu_count": 1,
            "allocated_gpu_name": arguments.gpu_name,
            "gpu_compute_used": False,
            "job_id": arguments.job_id,
            "partition": "gpu_4090",
            "reason_for_gpu_allocation": "The gpu_4090 partition rejects jobs that request zero GPUs.",
            "transport_mode": arguments.transport_mode,
            "superseded_failed_job_id": arguments.failed_job_id,
        },
        "files": {
            split_path.name: {
                "bytes": split_path.stat().st_size,
                "counts": split_counts,
                "etag": split_headers.get("etag"),
                "sha256": sha256_file(split_path),
                "source_commit": PINNED_PPIREF_COMMIT,
            },
            seqres_path.name: {
                "bytes": seqres_path.stat().st_size,
                "counts": seqres_counts,
                "etag": seqres_headers.get("etag"),
                "last_modified": seqres_headers.get("last-modified"),
                "sha256": sha256_file(seqres_path),
            },
        },
        "format_version": "abpairfm_ppiref50k_aux_materialization_v1",
        "sources": {
            "ppiref_split": f"https://raw.githubusercontent.com/anton-bushuiev/PPIRef/{PINNED_PPIREF_COMMIT}/ppiref/data/splits/ppiref_6A_filtered_clustered_04.json",
            "wwpdb_seqres": "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz",
        },
        "status": "MATERIALIZED_NOT_NORMALIZED",
        "test_access": False,
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
