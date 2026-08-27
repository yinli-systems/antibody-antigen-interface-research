#!/usr/bin/env python3
"""Audit and merge all protocol-v3 exclusion shard results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


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
        raise ValueError("plan SHA-256 differs")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("format_version") != "paired_esmc_cdr_decontamination_plan_v2"
        or plan.get("status") != "FROZEN_PROTOCOL_V3_EXCLUSION_BEFORE_REDUCTION"
        or plan.get("scripts", {}).get(Path(__file__).name) != sha256_file(Path(__file__))
        or plan.get("audit", {}).get("test_access") is not False
    ):
        raise ValueError("plan contract differs")
    manifest_path = Path(plan["shards"]["manifest"])
    if sha256_file(manifest_path) != plan["shards"]["manifest_sha256"]:
        raise ValueError("shard manifest differs")
    manifest = json.loads(manifest_path.read_text())
    count = plan["shards"]["count"]
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    temporary = args.output_root.with_name(args.output_root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    output = temporary / "excluded-row-hashes.tsv.gz"
    totals = {
        "hit_lines": 0,
        "query_pairs_with_any_hit": 0,
        "single_chain_90pct_pairs": 0,
        "paired_same_instance_80pct_pairs": 0,
        "excluded_union_pairs": 0,
    }
    previous: str | None = None
    evidence = []
    try:
        with output.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
                for index in range(count):
                    root = args.reductions_root / f"shard-{index:03d}"
                    result_path = root / "SHARD_REDUCTION_RESULT_V2.json"
                    receipt_path = root / "SHARD_REDUCTION_RESULT_V2_RECEIPT.json"
                    result = json.loads(result_path.read_text())
                    receipt = json.loads(receipt_path.read_text())
                    if (
                        result.get("status") != "PASS_PROTOCOL_V3_SHARD_REDUCTION"
                        or result.get("plan_sha256") != args.plan_sha256
                        or result.get("shard_index") != index
                        or result.get("audit", {}).get("test_access") is not False
                        or receipt.get("result_sha256") != sha256_file(result_path)
                        or receipt.get("output_sha256") != result["output"]["sha256"]
                    ):
                        raise ValueError(f"shard {index} result differs")
                    excluded_path = root / result["output"]["path"]
                    if sha256_file(excluded_path) != result["output"]["sha256"]:
                        raise ValueError(f"shard {index} output differs")
                    observed = 0
                    with gzip.open(excluded_path, "rt", encoding="ascii", newline="") as handle:
                        for line in handle:
                            if not line.endswith("\n"):
                                raise ValueError(f"shard {index} unterminated hash")
                            row_hash = line[:-1]
                            if (
                                len(row_hash) != 64
                                or int(row_hash[:2], 16) * count // 256 != index
                                or (previous is not None and row_hash <= previous)
                            ):
                                raise ValueError(f"shard {index} order/partition differs")
                            out.write(row_hash.encode() + b"\n")
                            previous = row_hash
                            observed += 1
                    if observed != result["counts"]["excluded_union_pairs"]:
                        raise ValueError(f"shard {index} count differs")
                    totals["hit_lines"] += result["input"]["hit_lines"]
                    for key in totals:
                        if key != "hit_lines":
                            totals[key] += result["counts"][key]
                    evidence.append(
                        {
                            "index": index,
                            "result_sha256": receipt["result_sha256"],
                            "receipt_sha256": sha256_file(receipt_path),
                        }
                    )
        if totals["hit_lines"] != manifest["total_hit_lines"]:
            raise ValueError("global hit-line count differs")
        result = {
            "format_version": "paired_esmc_cdr_protocol_v3_exclusions_merged_v1",
            "status": "PASS_PROTOCOL_V3_ALL_SHARDS_MERGED",
            "plan_sha256": args.plan_sha256,
            "counts": totals,
            "output": {
                "path": output.name,
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
            "shard_evidence": evidence,
            "audit": {
                "all_64_shards_verified": True,
                "global_uniqueness_verified": True,
                "task_labels_read": False,
                "test_access": False,
                "legacy_partial_sqlite_reused": False,
            },
        }
        result_path = temporary / "MERGED_EXCLUSIONS_RESULT_V2.json"
        atomic_write(result_path, canonical_bytes(result))
        receipt = {
            "format_version": "paired_esmc_cdr_protocol_v3_exclusions_merged_receipt_v1",
            "status": result["status"],
            "plan_sha256": args.plan_sha256,
            "result_sha256": sha256_file(result_path),
            "output_sha256": result["output"]["sha256"],
            "excluded_union_pairs": totals["excluded_union_pairs"],
            "test_access": False,
        }
        atomic_write(
            temporary / "MERGED_EXCLUSIONS_RESULT_V2_RECEIPT.json",
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
