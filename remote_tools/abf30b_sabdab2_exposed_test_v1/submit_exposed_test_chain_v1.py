#!/usr/bin/env python3
"""Submit the frozen label-free preparation and one exposed test opening chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value); descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle: handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def submit(path: Path, dependency: str | None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency: command.append(f"--dependency=afterok:{dependency}")
    command.append(str(path)); output = subprocess.check_output(command, text=True).strip(); job = output.split(";", 1)[0]
    if not job.isdecimal(): raise RuntimeError(f"unexpected sbatch output: {output}")
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True); parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.campaign_root / "authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1.json"
    receipt_path = args.campaign_root / "authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1_RECEIPT.json"
    output = args.campaign_root / "authority/EXPOSED_TEST_CHAIN_SUBMISSION_RECEIPT_V1.json"
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text()); receipt = json.loads(receipt_path.read_text()); protocol_sha = sha256_file(protocol_path)
    if (
        protocol.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ"
        or receipt.get("protocol_sha256") != protocol_sha
        or receipt.get("frozen_before_test_label_artifact_read") is not True
        or protocol.get("claim_boundary", {}).get("cumulative_test_access_ordinal") != 3
    ):
        raise ValueError("preparation authority differs")
    scripts = {
        "inputs": args.tools_root / "build_test_inputs_v1.sbatch",
        "sdh_af3": args.tools_root / "run_test_sdh_af3_array_v1.sbatch",
        "structure": args.tools_root / "materialize_test_structure_cache_v1.sbatch",
        "embeddings": args.tools_root / "materialize_test_embeddings_v1.sbatch",
        "measurement_freeze": args.tools_root / "freeze_test_measurement_v1.sbatch",
        "evaluation": args.tools_root / "evaluate_exposed_test_v1.sbatch",
        "analyzer": args.tools_root / "analyze_exposed_test_v1.sbatch",
    }
    for path in scripts.values():
        if path.is_symlink() or not path.is_file(): raise ValueError(f"script missing: {path}")
    jobs: dict[str, str] = {}; dependencies: dict[str, str] = {}
    previous = None
    for name in scripts:
        jobs[name] = submit(scripts[name], previous)
        if previous: dependencies[jobs[name]] = f"afterok:{previous}"
        previous = jobs[name]
    payload = {
        "format_version": "abf30b_sabdab2_exposed_test_chain_submission_receipt_v1",
        "status": "SUBMITTED_FROZEN_EXPOSED_HISTORICAL_TEST_CHAIN",
        "prep_protocol_sha256": protocol_sha,
        "scripts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in scripts.items()},
        "jobs": jobs, "dependencies": dependencies,
        "sdh_af3_batches": 3, "evaluation_components": 6,
        "checkpoint_or_model_selected_on_test": False,
        "fresh_blind_test": False, "cumulative_test_access_ordinal": 3,
        "sable_repeated": False, "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(output, payload)
    print(json.dumps({"receipt_sha256": receipt_sha, "jobs": jobs}, sort_keys=True))


if __name__ == "__main__":
    main()
