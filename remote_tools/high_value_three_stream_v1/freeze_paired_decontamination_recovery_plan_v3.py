#!/usr/bin/env python3
"""Freeze an order-independent recovery for the protocol-v3 hit reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCRIPTS = (
    "reduce_paired_hit_shard_protocol_v3_unordered_v2.py",
    "merge_paired_hit_shards_protocol_v3.py",
    "materialize_paired_devclean_protocol_v3.py",
    "slurm_reduce_paired_hits_protocol_v3_unordered_v2.sbatch",
    "slurm_merge_paired_hits_protocol_v3_unordered_v2.sbatch",
    "slurm_materialize_paired_devclean_protocol_v3_unordered_v2.sbatch",
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
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--old-plan-sha256", required=True)
    parser.add_argument("--failed-log-root", type=Path, required=True)
    parser.add_argument("--failed-array-job-id", type=int, required=True)
    parser.add_argument("--failed-reduction-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.old_plan) != args.old_plan_sha256:
        raise RuntimeError("superseded plan SHA differs")
    old = json.loads(args.old_plan.read_text())
    if (
        old.get("format_version") != "paired_esmc_cdr_decontamination_plan_v2"
        or old.get("status") != "FROZEN_PROTOCOL_V3_EXCLUSION_BEFORE_REDUCTION"
        or old.get("decontamination", {}).get("minimum_reciprocal_coverage") != 0.8
        or old.get("decontamination", {}).get("single_chain_exclusion_identity") != 0.9
        or old.get("decontamination", {}).get(
            "paired_same_development_instance_identity"
        )
        != 0.8
        or old.get("audit", {}).get("test_access") is not False
    ):
        raise RuntimeError("superseded plan contract differs")
    logs = sorted(
        args.failed_log_root.glob(f"reduce-v3-{args.failed_array_job_id}_*.err")
    )
    hashes = {sha256_file(path) for path in logs}
    if (
        len(logs) != 64
        or len(hashes) != 1
        or any(
            "row hash is not contiguous in frozen shard" not in path.read_text()
            for path in logs
        )
        or (
            args.failed_reduction_root.exists()
            and any(args.failed_reduction_root.iterdir())
        )
    ):
        raise RuntimeError("failed reduction evidence differs")
    scripts = {name: sha256_file(args.source_root / name) for name in SCRIPTS}
    plan = json.loads(json.dumps(old))
    plan["supersedes_plan_sha256"] = args.old_plan_sha256
    plan["scripts"] = scripts
    plan["recovery"] = {
        "failed_array_job_id": args.failed_array_job_id,
        "failed_task_count": 64,
        "representative_stderr_sha256": next(iter(hashes)),
        "failure_class": "row_hash_order_assumption_before_any_reduction_output",
        "semantic_change": (
            "aggregate protocol-v3 evidence by row hash within each existing "
            "1/64 deterministic shard, independent of MMseqs line order"
        ),
        "thresholds_changed": False,
        "hits_recomputed": False,
        "source_pairs_changed": False,
        "shard_membership_changed": False,
        "partial_outputs_reused": False,
    }
    plan["output_names"] = {
        "shard_reductions": "shard-reductions-protocol-v3-unordered-v2",
        "merged_exclusions": "merged-exclusions-protocol-v3-unordered-v2",
        "corpus": "oas_stage_b_devclean_protocol_v3_unordered_v2.tsv.gz",
    }
    plan["audit"].update(
        {
            "frozen_before_recovery_reduction": True,
            "order_independent_reducer_tested": True,
            "training_started": False,
            "test_access": False,
        }
    )
    atomic_write(args.output, canonical_bytes(plan))
    receipt = {
        "format_version": "paired_esmc_cdr_decontamination_plan_receipt_v3",
        "status": plan["status"],
        "plan_sha256": sha256_file(args.output),
        "supersedes_plan_sha256": args.old_plan_sha256,
        "failed_array_job_id": args.failed_array_job_id,
        "failed_task_count": 64,
        "thresholds_changed": False,
        "test_access": False,
        "training_started": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
