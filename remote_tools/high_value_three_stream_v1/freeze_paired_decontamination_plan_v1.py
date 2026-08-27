#!/usr/bin/env python3
"""Freeze the resumable paired ESM-C decontamination execution plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCRIPT_NAMES = (
    "reduce_paired_hit_shard_v1.py",
    "merge_paired_hit_shards_v1.py",
    "materialize_paired_devclean_v1.py",
)


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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--shard-receipt", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol SHA-256 differs")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("format_version") != "abpairfm_esmc_cdr_study_protocol_v3"
        or protocol.get("data", {}).get("provided_test_split_access") is not False
        or protocol.get("data", {}).get("legacy_partial_sqlite_reused") is not False
        or protocol.get("decontamination", {}).get("shard_count") != 64
    ):
        raise ValueError("paired ESM-C protocol contract differs")
    manifest = json.loads(args.shard_manifest.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(args.shard_manifest)
    shard_receipt = json.loads(args.shard_receipt.read_text(encoding="utf-8"))
    if (
        manifest.get("format_version") != "oas_pairwise_hits_shards_v1"
        or manifest.get("status") != "PASS_BOUNDED_SHARD_STAGING"
        or manifest.get("shard_count") != 64
        or manifest.get("inputs", {}).get("hits_sha256")
        != protocol["data"]["downstream_decontamination_hits_sha256"]
        or manifest.get("audit", {}).get("test_access") is not False
        or shard_receipt.get("manifest_sha256") != manifest_sha
        or shard_receipt.get("test_access") is not False
    ):
        raise ValueError("shard manifest/receipt contract differs")
    source_receipt = json.loads(args.source_receipt.read_text(encoding="utf-8"))
    source_sha = sha256_file(args.source_pairs)
    if (
        source_receipt.get("format_version") != "abpairfm_oas_stage_b_candidates_v2"
        or source_receipt.get("counts", {}).get("unique_exact_pairs")
        != protocol["data"]["source_pairs"]
        or source_receipt.get("output", {}).get("sha256") != source_sha
        or source_receipt.get("test_access") is not False
    ):
        raise ValueError("source pair receipt contract differs")
    scripts = {}
    for name in SCRIPT_NAMES:
        path = args.source_root / name
        scripts[name] = sha256_file(path)
    plan = {
        "format_version": "paired_esmc_cdr_decontamination_plan_v1",
        "status": "FROZEN_BEFORE_SHARD_REDUCTION",
        "study_protocol_sha256": args.protocol_sha256,
        "hits": {
            "sha256": protocol["data"]["downstream_decontamination_hits_sha256"],
            "search_orientation": "oas_query_development_target",
            "identifier_column": "query",
        },
        "shards": {
            "count": 64,
            "manifest": str(args.shard_manifest),
            "manifest_sha256": manifest_sha,
            "receipt": str(args.shard_receipt),
            "receipt_sha256": sha256_file(args.shard_receipt),
            "rule": manifest["shard_rule"],
            "total_hit_lines": manifest["total_hit_lines"],
        },
        "source_pairs": {
            "path": str(args.source_pairs),
            "sha256": source_sha,
            "receipt": str(args.source_receipt),
            "receipt_sha256": sha256_file(args.source_receipt),
            "count": protocol["data"]["source_pairs"],
        },
        "decontamination": {
            "minimum_sequence_identity": 0.5,
            "minimum_query_coverage": 0.8,
            "coverage_mode": 0,
            "maximum_target_candidates": 64,
            "sequence_masking": False,
            "mmseqs_revision": "f71d0a6b57c19e3a8954d8557f065c0fe1f46142",
            "pair_exclusion_rule": "exclude the complete VH/VL pair if either chain has at least one frozen MMseqs hit",
        },
        "scripts": scripts,
        "outputs": {
            "reduction_root": str(args.output.parent.parent / "shard-reductions-v1"),
            "merged_root": str(args.output.parent.parent / "merged-exclusions-v1"),
            "filtered_pairs": str(
                args.source_pairs.parent
                / "oas_stage_b_devclean_50id80cov_sharded_v1.tsv.gz"
            ),
            "filtered_receipt": str(
                args.source_pairs.parent
                / "oas_stage_b_devclean_50id80cov_sharded_v1.receipt.json"
            ),
        },
        "audit": {
            "plan_frozen_after_hit_sharding_before_reduction": True,
            "independently_retryable_shards": True,
            "task_labels_read": False,
            "test_access": False,
            "legacy_partial_sqlite_reused": False,
            "training_started": False,
        },
    }
    atomic_write(args.output, canonical_bytes(plan))
    receipt = {
        "format_version": "paired_esmc_cdr_decontamination_plan_receipt_v1",
        "status": plan["status"],
        "plan_sha256": sha256_file(args.output),
        "study_protocol_sha256": args.protocol_sha256,
        "shard_manifest_sha256": manifest_sha,
        "script_sha256": scripts,
        "test_access": False,
        "training_started": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
