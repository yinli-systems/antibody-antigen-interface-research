#!/usr/bin/env python3
"""Reduce one frozen MMseqs hit shard to unique excluded OAS pair hashes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_query(fields: list[str], line_number: int) -> tuple[str, str]:
    if len(fields) != 8:
        raise ValueError(f"malformed MMseqs hit at line {line_number}")
    try:
        row_hash, chain = fields[0].rsplit("|", 1)
    except ValueError as error:
        raise ValueError(f"malformed MMseqs query at line {line_number}") from error
    if (
        len(row_hash) != 64
        or any(character not in "0123456789abcdef" for character in row_hash)
        or chain not in {"heavy", "light"}
    ):
        raise ValueError(f"malformed MMseqs query at line {line_number}")
    try:
        fident = float(fields[2])
        alnlen = int(fields[3])
        qcov = float(fields[4])
        tcov = float(fields[5])
        evalue = float(fields[6])
        bits = float(fields[7])
    except ValueError as error:
        raise ValueError(f"non-numeric MMseqs hit at line {line_number}") from error
    if (
        not all(math.isfinite(value) for value in (fident, qcov, tcov, evalue, bits))
        or fident < 0.5
        or qcov < 0.8
        or alnlen <= 0
        or evalue < 0.0
    ):
        raise ValueError(f"MMseqs threshold/finite contract differs at line {line_number}")
    return row_hash, chain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.plan) != args.plan_sha256:
        raise ValueError("execution plan SHA-256 differs")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if (
        plan.get("format_version") != "paired_esmc_cdr_decontamination_plan_v1"
        or plan.get("status") != "FROZEN_BEFORE_SHARD_REDUCTION"
        or plan.get("audit", {}).get("test_access") is not False
        or plan.get("scripts", {}).get("reduce_paired_hit_shard_v1.py")
        != sha256_file(Path(__file__))
    ):
        raise ValueError("execution plan contract differs")
    shard_count = plan["shards"]["count"]
    if not 0 <= args.shard_index < shard_count:
        raise ValueError("shard index is outside the frozen range")
    manifest_path = Path(plan["shards"]["manifest"])
    if sha256_file(manifest_path) != plan["shards"]["manifest_sha256"]:
        raise ValueError("shard manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS_BOUNDED_SHARD_STAGING"
        or manifest.get("shard_count") != shard_count
        or manifest.get("audit", {}).get("test_access") is not False
    ):
        raise ValueError("shard manifest contract differs")
    shard_record = manifest["shards"][args.shard_index]
    if shard_record["index"] != args.shard_index:
        raise ValueError("shard manifest ordering differs")
    shard_path = manifest_path.parent / shard_record["path"]
    if sha256_file(shard_path) != shard_record["sha256"]:
        raise ValueError("input shard SHA-256 differs")

    final_root = args.output_root / f"shard-{args.shard_index:03d}"
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError(final_root)
    temporary = final_root.with_name(final_root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    excluded: set[str] = set()
    chain_hit_counts = {"heavy": 0, "light": 0}
    line_count = 0
    try:
        with gzip.open(shard_path, "rt", encoding="utf-8", newline="") as handle:
            for line_count, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(f"unterminated MMseqs hit at line {line_count}")
                row_hash, chain = checked_query(line[:-1].split("\t"), line_count)
                observed_shard = int(row_hash[:2], 16) * shard_count // 256
                if observed_shard != args.shard_index:
                    raise ValueError(f"query assigned to wrong shard at line {line_count}")
                excluded.add(row_hash)
                chain_hit_counts[chain] += 1
        if line_count != shard_record["line_count"]:
            raise ValueError("input shard line count differs")

        output = temporary / "excluded-row-hashes.tsv.gz"
        with output.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0) as compressed:
                for row_hash in sorted(excluded):
                    compressed.write(row_hash.encode("ascii") + b"\n")
        result = {
            "format_version": "paired_esmc_cdr_excluded_hash_shard_v1",
            "status": "PASS_INDEPENDENT_SHARD_REDUCTION",
            "plan_sha256": args.plan_sha256,
            "shard_manifest_sha256": plan["shards"]["manifest_sha256"],
            "shard_index": args.shard_index,
            "input": {
                "path": str(shard_path),
                "sha256": shard_record["sha256"],
                "hit_lines": line_count,
            },
            "counts": {
                "excluded_unique_pairs": len(excluded),
                "heavy_hit_lines": chain_hit_counts["heavy"],
                "light_hit_lines": chain_hit_counts["light"],
            },
            "output": {
                "path": "excluded-row-hashes.tsv.gz",
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
            "audit": {
                "thresholds_revalidated": True,
                "shard_partition_revalidated": True,
                "task_labels_read": False,
                "test_access": False,
                "legacy_partial_sqlite_reused": False,
            },
        }
        result_path = temporary / "SHARD_REDUCTION_RESULT_V1.json"
        atomic_write(result_path, canonical_bytes(result))
        receipt = {
            "format_version": "paired_esmc_cdr_excluded_hash_shard_receipt_v1",
            "status": result["status"],
            "result_sha256": sha256_file(result_path),
            "output_sha256": result["output"]["sha256"],
            "plan_sha256": args.plan_sha256,
            "shard_index": args.shard_index,
            "test_access": False,
        }
        atomic_write(
            temporary / "SHARD_REDUCTION_RESULT_V1_RECEIPT.json",
            canonical_bytes(receipt),
        )
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final_root)
    except BaseException:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
        raise
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
