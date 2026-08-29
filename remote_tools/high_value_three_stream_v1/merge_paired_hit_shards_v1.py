#!/usr/bin/env python3
"""Audit and merge all independently reduced paired-decontamination shards."""

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
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--reductions-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.plan) != args.plan_sha256:
        raise ValueError("execution plan SHA-256 differs")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if (
        plan.get("format_version") != "paired_esmc_cdr_decontamination_plan_v1"
        or plan.get("status") != "FROZEN_BEFORE_SHARD_REDUCTION"
        or plan.get("scripts", {}).get("merge_paired_hit_shards_v1.py")
        != sha256_file(Path(__file__))
        or plan.get("audit", {}).get("test_access") is not False
    ):
        raise ValueError("execution plan contract differs")
    manifest_path = Path(plan["shards"]["manifest"])
    if sha256_file(manifest_path) != plan["shards"]["manifest_sha256"]:
        raise ValueError("shard manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_count = plan["shards"]["count"]
    if manifest.get("shard_count") != shard_count:
        raise ValueError("shard count differs")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    temporary = args.output_root.with_name(args.output_root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    output_path = temporary / "excluded-row-hashes.tsv.gz"
    total_hit_lines = 0
    total_excluded = 0
    shard_evidence = []
    previous_hash: str | None = None
    try:
        with output_path.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0) as compressed:
                for index in range(shard_count):
                    root = args.reductions_root / f"shard-{index:03d}"
                    result_path = root / "SHARD_REDUCTION_RESULT_V1.json"
                    receipt_path = root / "SHARD_REDUCTION_RESULT_V1_RECEIPT.json"
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if (
                        result.get("status") != "PASS_INDEPENDENT_SHARD_REDUCTION"
                        or result.get("plan_sha256") != args.plan_sha256
                        or result.get("shard_index") != index
                        or result.get("audit", {}).get("test_access") is not False
                        or receipt.get("result_sha256") != sha256_file(result_path)
                        or receipt.get("output_sha256") != result.get("output", {}).get("sha256")
                        or receipt.get("test_access") is not False
                    ):
                        raise ValueError(f"shard {index} result/receipt contract differs")
                    excluded_path = root / result["output"]["path"]
                    if sha256_file(excluded_path) != result["output"]["sha256"]:
                        raise ValueError(f"shard {index} excluded-list SHA-256 differs")
                    observed = 0
                    with gzip.open(excluded_path, "rt", encoding="ascii", newline="") as handle:
                        for line_number, line in enumerate(handle, start=1):
                            if not line.endswith("\n"):
                                raise ValueError(f"shard {index} has unterminated hash")
                            row_hash = line[:-1]
                            if (
                                len(row_hash) != 64
                                or any(c not in "0123456789abcdef" for c in row_hash)
                                or int(row_hash[:2], 16) * shard_count // 256 != index
                                or (previous_hash is not None and row_hash <= previous_hash)
                            ):
                                raise ValueError(f"shard {index} ordering/partition differs")
                            compressed.write(row_hash.encode("ascii") + b"\n")
                            previous_hash = row_hash
                            observed += 1
                    if observed != result["counts"]["excluded_unique_pairs"]:
                        raise ValueError(f"shard {index} excluded count differs")
                    expected_hit_lines = manifest["shards"][index]["line_count"]
                    if result["input"]["hit_lines"] != expected_hit_lines:
                        raise ValueError(f"shard {index} hit count differs")
                    total_hit_lines += expected_hit_lines
                    total_excluded += observed
                    shard_evidence.append(
                        {
                            "index": index,
                            "result_sha256": receipt["result_sha256"],
                            "receipt_sha256": sha256_file(receipt_path),
                            "excluded_unique_pairs": observed,
                            "hit_lines": expected_hit_lines,
                        }
                    )
        if total_hit_lines != manifest["total_hit_lines"]:
            raise ValueError("global hit-line total differs")
        result = {
            "format_version": "paired_esmc_cdr_excluded_hashes_merged_v1",
            "status": "PASS_ALL_SHARDS_MERGED_VERIFIED",
            "plan_sha256": args.plan_sha256,
            "shard_manifest_sha256": plan["shards"]["manifest_sha256"],
            "counts": {
                "shards": shard_count,
                "hit_lines": total_hit_lines,
                "excluded_unique_pairs": total_excluded,
            },
            "output": {
                "path": "excluded-row-hashes.tsv.gz",
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            },
            "shard_evidence": shard_evidence,
            "audit": {
                "all_shard_results_and_receipts_verified": True,
                "global_sort_and_cross_shard_uniqueness_verified": True,
                "task_labels_read": False,
                "test_access": False,
                "legacy_partial_sqlite_reused": False,
            },
        }
        result_path = temporary / "MERGED_EXCLUSIONS_RESULT_V1.json"
        atomic_write(result_path, canonical_bytes(result))
        receipt = {
            "format_version": "paired_esmc_cdr_excluded_hashes_merged_receipt_v1",
            "status": result["status"],
            "result_sha256": sha256_file(result_path),
            "output_sha256": result["output"]["sha256"],
            "plan_sha256": args.plan_sha256,
            "excluded_unique_pairs": total_excluded,
            "test_access": False,
        }
        atomic_write(
            temporary / "MERGED_EXCLUSIONS_RESULT_V1_RECEIPT.json",
            canonical_bytes(receipt),
        )
        args.output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, args.output_root)
    except BaseException:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
        raise
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
