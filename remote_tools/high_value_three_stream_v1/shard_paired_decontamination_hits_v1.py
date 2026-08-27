#!/usr/bin/env python3
"""Stage a huge frozen MMseqs hit stream into bounded deterministic shards.

This is the resumability boundary for the paired/CDR study.  It reads no task
labels or test data.  Downstream shard auditors operate independently and can
be retried without replaying or reusing the failed legacy SQLite database.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def query_digest(line: bytes, line_number: int) -> bytes:
    endpoint = line.split(b"\t", 1)[0]
    try:
        digest, chain = endpoint.rsplit(b"|", 1)
        raw = bytes.fromhex(digest.decode("ascii"))
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"malformed query endpoint at line {line_number}") from error
    if len(raw) != 32 or chain not in {b"heavy", b"light"}:
        raise ValueError(f"malformed query endpoint at line {line_number}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--hits-sha256", required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=64)
    args = parser.parse_args()
    if args.shards <= 0 or args.shards > 256 or 256 % args.shards:
        raise ValueError("shards must be a positive divisor of 256")
    if sha256_file(args.hits) != args.hits_sha256:
        raise ValueError("hits SHA-256 differs")
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol SHA-256 differs")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("format_version") != "abpairfm_esmc_cdr_study_protocol_v1"
        or protocol.get("status") != "FROZEN_DATA_GATE_BEFORE_ESMC_MODEL_BINDING"
        or protocol.get("decontamination", {}).get("sharder_sha256")
        != sha256_file(Path(__file__))
        or protocol.get("decontamination", {}).get("shard_count") != args.shards
    ):
        raise ValueError("paired/CDR protocol differs")
    source_receipt = json.loads(args.source_receipt.read_text())
    if (
        source_receipt.get("format_version") != "abpairfm_oas_stage_b_candidates_v2"
        or source_receipt.get("test_access") is not False
        or not isinstance(source_receipt.get("counts", {}).get("unique_exact_pairs"), int)
    ):
        raise ValueError("source receipt differs")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    temporary = args.output_root.with_name(args.output_root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    raw_handles = []
    gzip_handles = []
    counts = [0] * args.shards
    try:
        for index in range(args.shards):
            raw = (temporary / f"hits-shard-{index:03d}.tsv.gz").open("xb")
            compressed = gzip.GzipFile(
                filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0
            )
            raw_handles.append(raw)
            gzip_handles.append(compressed)
        with gzip.open(args.hits, "rb") as source:
            for line_number, line in enumerate(source, start=1):
                if line.count(b"\t") != 7 or not line.endswith(b"\n"):
                    raise ValueError(f"malformed MMseqs line {line_number}")
                raw_digest = query_digest(line, line_number)
                shard = raw_digest[0] * args.shards // 256
                gzip_handles[shard].write(line)
                counts[shard] += 1
                if line_number % 5_000_000 == 0:
                    print(json.dumps({"phase": "shard_hits", "lines": line_number}), flush=True)
    finally:
        for handle in gzip_handles:
            handle.close()
        for handle in raw_handles:
            handle.close()
    if not all(counts):
        raise ValueError("one or more deterministic shards are empty")
    shards = [
        {
            "index": index,
            "path": f"hits-shard-{index:03d}.tsv.gz",
            "line_count": counts[index],
            "sha256": sha256_file(temporary / f"hits-shard-{index:03d}.tsv.gz"),
        }
        for index in range(args.shards)
    ]
    manifest = {
        "format_version": "oas_pairwise_hits_shards_v1",
        "status": "PASS_BOUNDED_SHARD_STAGING",
        "protocol_sha256": args.protocol_sha256,
        "inputs": {
            "hits_sha256": args.hits_sha256,
            "source_receipt_sha256": sha256_file(args.source_receipt),
        },
        "shard_rule": "floor(first_query_sha256_byte * shard_count / 256)",
        "shard_count": args.shards,
        "total_hit_lines": sum(counts),
        "shards": shards,
        "audit": {
            "task_labels_read": False,
            "test_access": False,
            "legacy_partial_sqlite_reused": False,
            "downstream_each_shard_independently_retryable": True,
        },
    }
    atomic_write(temporary / "PAIRWISE_HITS_SHARD_MANIFEST_V1.json", canonical_bytes(manifest))
    os.replace(temporary, args.output_root)
    result = args.output_root / "PAIRWISE_HITS_SHARD_MANIFEST_V1.json"
    receipt = {
        "format_version": "oas_pairwise_hits_shards_receipt_v1",
        "status": manifest["status"],
        "manifest_sha256": sha256_file(result),
        "protocol_sha256": args.protocol_sha256,
        "shard_count": args.shards,
        "total_hit_lines": sum(counts),
        "task_labels_read": False,
        "test_access": False,
    }
    atomic_write(
        args.output_root / "PAIRWISE_HITS_SHARD_RECEIPT_V1.json",
        canonical_bytes(receipt),
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
