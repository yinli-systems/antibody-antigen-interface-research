#!/usr/bin/env python3
"""Fail-closed archival for an interrupted ABPairFM Stage-A0 segment.

This utility never authorizes a successor.  It verifies that the requested
segment has no result or checkpoint, records immutable hashes for the failed
job evidence, and moves the partial progress file out of the production path
so the same segment can be replayed from the preceding verified DCP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--manifest-job-id", type=int, required=True)
    parser.add_argument("--stale-watchdog-job-id", type=int, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--input-generation", required=True)
    parser.add_argument("--expected-input-content-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()

    segment = f"segment-{args.segment_index:06d}"
    input_receipt = (
        args.campaign_root
        / "checkpoints"
        / args.input_generation
        / "receipt.json"
    )
    input_result = args.campaign_root / "results" / f"{args.input_generation}.json"
    progress_path = args.campaign_root / "progress" / f"{segment}.json"
    result_path = args.campaign_root / "results" / f"{segment}.json"
    checkpoint_path = args.campaign_root / "checkpoints" / segment
    manifest_path = (
        args.data_root / f"string_seed1701_{segment}_22400_v1.jsonl.zst"
    )
    manifest_receipt_path = (
        args.data_root / f"string_seed1701_{segment}_22400_v1.receipt.json"
    )
    stdout_path = args.log_root / f"abpairfm-3b-a0-pretrain-{args.job_id}.out"
    stderr_path = args.log_root / f"abpairfm-3b-a0-pretrain-{args.job_id}.err"
    archive = (
        args.campaign_root
        / "failed"
        / f"{segment}-job-{args.job_id}-fail-closed-v1"
    )

    required = [
        input_receipt,
        input_result,
        progress_path,
        manifest_path,
        manifest_receipt_path,
        stdout_path,
        stderr_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Required failure evidence missing: {missing}")
    if result_path.exists() or checkpoint_path.exists():
        raise SystemExit("Interrupted segment unexpectedly has a result or checkpoint")
    if archive.exists():
        raise SystemExit(f"Archive already exists: {archive}")

    receipt = load_json(input_receipt)
    previous_result = load_json(input_result)
    progress = load_json(progress_path)
    manifest_receipt = load_json(manifest_receipt_path)
    if (
        receipt.get("status") != "complete"
        or receipt.get("generation") != args.input_generation
        or receipt.get("content_sha256")
        != args.expected_input_content_sha256
        or receipt.get("provenance_hashes", {}).get("campaign_protocol")
        != args.expected_protocol_sha256
        or receipt.get("provenance_hashes", {}).get("training_source")
        != args.expected_source_sha256
        or previous_result.get("status") != "PASS_CHECKPOINT_VERIFIED"
        or previous_result.get("checkpoint", {}).get("content_sha256")
        != args.expected_input_content_sha256
        or previous_result.get("training_state", {}).get("next_segment_index")
        != args.segment_index
        or previous_result.get("test_access") is not False
    ):
        raise SystemExit("Preceding DCP/result does not authorize same-segment replay")
    numeric_fields = (
        "latest_global_mean_mlm_loss",
        "latest_gradient_norm",
        "learning_rate",
    )
    if (
        progress.get("format_version")
        != "abpairfm_3b_stage_a0_progress_v1"
        or progress.get("segment_index") != args.segment_index
        or progress.get("status") != "RUNNING"
        or progress.get("test_access") is not False
        or not all(
            isinstance(progress.get(field), (int, float))
            and math.isfinite(float(progress[field]))
            for field in numeric_fields
        )
    ):
        raise SystemExit("Partial progress is not a finite Stage-A0 training state")
    if (
        manifest_receipt.get("format_version")
        != "abpairfm_3b_stage_a0_segment_manifest_v1"
        or manifest_receipt.get("status")
        != "PRODUCTION_STAGE_A0_SEGMENT_READY"
        or manifest_receipt.get("segment", {}).get("index")
        != args.segment_index
        or manifest_receipt.get("inputs", {}).get("protocol_sha256")
        != args.expected_protocol_sha256
        or manifest_receipt.get("test_access") is not False
    ):
        raise SystemExit("Interrupted segment manifest receipt differs")

    payload = {
        "classification": (
            "UNCLASSIFIED_RANK_EXIT_WITH_TRUNCATED_TORCHELASTIC_STDERR"
        ),
        "claim_boundary": (
            "Failure evidence only. No result/checkpoint exists for this segment; "
            "the partial update state is not resumable or successor-authorizing."
        ),
        "failed_job": {
            "exit_code": "120:0",
            "job_id": args.job_id,
            "manifest_job_id": args.manifest_job_id,
            "slurm_state": "FAILED",
            "stale_watchdog_job_id": args.stale_watchdog_job_id,
        },
        "format_version": "abpairfm_3b_stage_a0_failure_receipt_v1",
        "input_checkpoint": {
            "content_sha256": args.expected_input_content_sha256,
            "generation": args.input_generation,
            "receipt_sha256": sha256(input_receipt),
            "result_sha256": sha256(input_result),
        },
        "partial_progress": progress,
        "protocol_sha256": args.expected_protocol_sha256,
        "segment_index": args.segment_index,
        "source_sha256": args.expected_source_sha256,
        "test_access": False,
        "unrecoverable_partial_state": True,
        "evidence": {
            "manifest_receipt_sha256": sha256(manifest_receipt_path),
            "manifest_sha256": sha256(manifest_path),
            "progress_sha256": sha256(progress_path),
            "stderr_bytes": stderr_path.stat().st_size,
            "stderr_sha256": sha256(stderr_path),
            "stdout_bytes": stdout_path.stat().st_size,
            "stdout_sha256": sha256(stdout_path),
        },
    }

    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(stdout_path, archive / "stdout.log")
    shutil.copy2(stderr_path, archive / "stderr.log")
    shutil.copy2(manifest_receipt_path, archive / "manifest.receipt.json")
    os.replace(progress_path, archive / "progress.json")
    atomic_json(archive / "failure-receipt.json", payload)
    print(json.dumps({"archive": str(archive), **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
