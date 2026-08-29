#!/usr/bin/env python3
"""Reduce one hit shard with the frozen paired ESM-C v3 exclusion rule."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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


def parse_hit(
    line: str, line_number: int
) -> tuple[str, str, str, str, float, float, float]:
    fields = line.split("\t")
    if len(fields) != 8:
        raise ValueError(f"malformed MMseqs hit at line {line_number}")
    try:
        row_hash, query_chain = fields[0].rsplit("|", 1)
        target_prefix, target_chain, target_sha = fields[1].rsplit("|", 2)
    except ValueError as error:
        raise ValueError(f"malformed endpoint at line {line_number}") from error
    if (
        len(row_hash) != 64
        or any(c not in "0123456789abcdef" for c in row_hash)
        or query_chain not in {"heavy", "light"}
        or target_chain not in {"heavy", "light", "antibody", "antigen"}
        or not target_sha.startswith("sha256=")
        or len(target_prefix.split("|", 1)) != 2
    ):
        raise ValueError(f"endpoint contract differs at line {line_number}")
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
        not all(math.isfinite(x) for x in (fident, qcov, tcov, evalue, bits))
        or fident < 0.5
        or qcov < 0.8
        or alnlen <= 0
        or evalue < 0
    ):
        raise ValueError(f"frozen hit-generation contract differs at line {line_number}")
    return row_hash, query_chain, target_prefix, target_chain, fident, qcov, tcov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
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
    shard_count = plan["shards"]["count"]
    if not 0 <= args.shard_index < shard_count:
        raise ValueError("shard index differs")
    manifest_path = Path(plan["shards"]["manifest"])
    if sha256_file(manifest_path) != plan["shards"]["manifest_sha256"]:
        raise ValueError("shard manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text())
    shard = manifest["shards"][args.shard_index]
    input_path = manifest_path.parent / shard["path"]
    if shard["index"] != args.shard_index or sha256_file(input_path) != shard["sha256"]:
        raise ValueError("input shard differs")

    final_root = args.output_root / f"shard-{args.shard_index:03d}"
    if final_root.exists() or final_root.is_symlink():
        raise FileExistsError(final_root)
    temporary = final_root.with_name(final_root.name + f".tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    excluded: list[str] = []
    single_high_count = 0
    paired_same_count = 0
    current: str | None = None
    high_single = False
    paired_targets = {"heavy": set(), "light": set()}
    seen: set[str] = set()

    def flush() -> None:
        nonlocal single_high_count, paired_same_count, high_single, paired_targets
        if current is None:
            return
        paired_same = bool(paired_targets["heavy"].intersection(paired_targets["light"]))
        if high_single or paired_same:
            excluded.append(current)
        single_high_count += int(high_single)
        paired_same_count += int(paired_same)
        high_single = False
        paired_targets = {"heavy": set(), "light": set()}

    line_count = 0
    try:
        with gzip.open(input_path, "rt", encoding="utf-8", newline="") as handle:
            for line_count, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(f"unterminated line {line_count}")
                row_hash, chain, target, target_chain, identity, qcov, tcov = parse_hit(
                    line[:-1], line_count
                )
                if int(row_hash[:2], 16) * shard_count // 256 != args.shard_index:
                    raise ValueError(f"wrong shard at line {line_count}")
                if row_hash != current:
                    flush()
                    if row_hash in seen:
                        raise ValueError("row hash is not contiguous in frozen shard")
                    seen.add(row_hash)
                    current = row_hash
                reciprocal = qcov + 1e-12 >= 0.8 and tcov + 1e-12 >= 0.8
                if reciprocal and identity + 1e-12 >= 0.9:
                    high_single = True
                if (
                    reciprocal
                    and identity + 1e-12 >= 0.8
                    and target_chain == chain
                ):
                    paired_targets[chain].add(target)
        flush()
        if line_count != shard["line_count"]:
            raise ValueError("shard hit-line count differs")
        excluded.sort()
        output = temporary / "excluded-row-hashes.tsv.gz"
        with output.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
                for row_hash in excluded:
                    out.write(row_hash.encode("ascii") + b"\n")
        result = {
            "format_version": "paired_esmc_cdr_protocol_v3_exclusion_shard_v1",
            "status": "PASS_PROTOCOL_V3_SHARD_REDUCTION",
            "plan_sha256": args.plan_sha256,
            "shard_index": args.shard_index,
            "input": {"sha256": shard["sha256"], "hit_lines": line_count},
            "counts": {
                "query_pairs_with_any_hit": len(seen),
                "single_chain_90pct_pairs": single_high_count,
                "paired_same_instance_80pct_pairs": paired_same_count,
                "excluded_union_pairs": len(excluded),
            },
            "output": {
                "path": output.name,
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
            "audit": {
                "reciprocal_coverage_applied": True,
                "protocol_v3_rule_applied": True,
                "task_labels_read": False,
                "test_access": False,
                "legacy_partial_sqlite_reused": False,
            },
        }
        result_path = temporary / "SHARD_REDUCTION_RESULT_V2.json"
        atomic_write(result_path, canonical_bytes(result))
        receipt = {
            "format_version": "paired_esmc_cdr_protocol_v3_exclusion_shard_receipt_v1",
            "status": result["status"],
            "plan_sha256": args.plan_sha256,
            "shard_index": args.shard_index,
            "result_sha256": sha256_file(result_path),
            "output_sha256": result["output"]["sha256"],
            "test_access": False,
        }
        atomic_write(
            temporary / "SHARD_REDUCTION_RESULT_V2_RECEIPT.json",
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
