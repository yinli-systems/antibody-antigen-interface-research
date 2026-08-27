#!/usr/bin/env python3
"""Freeze corrected paired decontamination matching protocol v3 exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCRIPTS = (
    "reduce_paired_hit_shard_protocol_v3.py",
    "merge_paired_hit_shards_protocol_v3.py",
    "materialize_paired_devclean_protocol_v3.py",
    "slurm_reduce_paired_hits_protocol_v3.sbatch",
    "slurm_merge_paired_hits_protocol_v3.sbatch",
    "slurm_materialize_paired_devclean_protocol_v3.sbatch",
)


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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--old-plan-sha256", required=True)
    parser.add_argument("--overexcluded-receipt", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--shard-receipt", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    for path, expected, label in (
        (args.protocol, args.protocol_sha256, "protocol"),
        (args.old_plan, args.old_plan_sha256, "old plan"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 differs")
    protocol = json.loads(args.protocol.read_text())
    old_plan = json.loads(args.old_plan.read_text())
    overexcluded = json.loads(args.overexcluded_receipt.read_text())
    manifest = json.loads(args.shard_manifest.read_text())
    shard_receipt = json.loads(args.shard_receipt.read_text())
    source_receipt = json.loads(args.source_receipt.read_text())
    manifest_sha = sha256_file(args.shard_manifest)
    source_sha = sha256_file(args.source_pairs)
    if (
        protocol.get("format_version") != "abpairfm_esmc_cdr_study_protocol_v3"
        or protocol.get("decontamination", {}).get("single_chain_exclusion_identity")
        != 0.9
        or protocol.get("decontamination", {}).get("paired_same_development_instance_identity")
        != 0.8
        or protocol.get("decontamination", {}).get("minimum_reciprocal_coverage")
        != 0.8
        or old_plan.get("status") != "FROZEN_BEFORE_SHARD_REDUCTION"
        or overexcluded.get("counts", {}).get("kept_pairs") != 3
        or overexcluded.get("audit", {}).get("test_access") is not False
        or manifest.get("status") != "PASS_BOUNDED_SHARD_STAGING"
        or manifest.get("shard_count") != 64
        or shard_receipt.get("manifest_sha256") != manifest_sha
        or source_receipt.get("output", {}).get("sha256") != source_sha
        or source_receipt.get("test_access") is not False
    ):
        raise ValueError("corrected decontamination authority differs")
    scripts = {name: sha256_file(args.source_root / name) for name in SCRIPTS}
    plan = {
        "format_version": "paired_esmc_cdr_decontamination_plan_v2",
        "status": "FROZEN_PROTOCOL_V3_EXCLUSION_BEFORE_REDUCTION",
        "study_protocol_sha256": args.protocol_sha256,
        "supersedes_plan_sha256": args.old_plan_sha256,
        "overexclusion_diagnostic": {
            "receipt": str(args.overexcluded_receipt),
            "receipt_sha256": sha256_file(args.overexcluded_receipt),
            "kept_pairs": 3,
            "training_use_forbidden": True,
            "reason": "v1 reducer applied any >=50% hit instead of the frozen v3 compound rule",
        },
        "hits": {
            "sha256": protocol["data"]["downstream_decontamination_hits_sha256"],
            "generation_floor_identity": 0.5,
            "generation_floor_query_coverage": 0.8,
            "search_orientation": "oas_query_development_target",
        },
        "shards": {
            "count": 64,
            "manifest": str(args.shard_manifest),
            "manifest_sha256": manifest_sha,
            "receipt": str(args.shard_receipt),
            "receipt_sha256": sha256_file(args.shard_receipt),
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
            "minimum_reciprocal_coverage": 0.8,
            "single_chain_exclusion_identity": 0.9,
            "paired_same_development_instance_identity": 0.8,
            "rule": "exclude if either chain has any reciprocal-coverage development hit at >=0.9 identity OR heavy-to-heavy and light-to-light reciprocal-coverage hits each reach >=0.8 identity to the same frozen development instance",
            "pair_safe": True,
        },
        "scripts": scripts,
        "audit": {
            "frozen_before_corrected_reduction": True,
            "thresholds_copied_from_protocol_v3": True,
            "task_labels_read": False,
            "test_access": False,
            "legacy_partial_sqlite_reused": False,
            "training_started": False,
        },
    }
    atomic_write(args.output, canonical_bytes(plan))
    receipt = {
        "format_version": "paired_esmc_cdr_decontamination_plan_receipt_v2",
        "status": plan["status"],
        "plan_sha256": sha256_file(args.output),
        "study_protocol_sha256": args.protocol_sha256,
        "supersedes_plan_sha256": args.old_plan_sha256,
        "script_sha256": scripts,
        "test_access": False,
        "training_started": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
