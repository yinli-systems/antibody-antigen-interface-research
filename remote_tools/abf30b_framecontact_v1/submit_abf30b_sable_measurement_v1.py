#!/usr/bin/env python3
"""Submit the frozen ABF-30B analyst-exposed SABLE measurement chain."""

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
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def submit(path: Path, dependency: str) -> str:
    output = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency=afterok:{dependency}", str(path)],
        text=True,
    ).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdecimal():
        raise RuntimeError(f"unexpected sbatch output: {output!r}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--development-analyzer-job", default="1496570")
    parser.add_argument("--sable-embedding-job", default="1496583")
    args = parser.parse_args()
    if not args.development_analyzer_job.isdecimal() or not args.sable_embedding_job.isdecimal():
        raise ValueError("upstream job ID differs")

    authority = args.campaign_root / "authority/sable_measurement_v1"
    plan_path = authority / "ABF30B_SABLE_MEASUREMENT_PLAN_V1.json"
    plan_receipt_path = authority / "ABF30B_SABLE_MEASUREMENT_PLAN_V1_RECEIPT.json"
    label_free_submission_path = (
        args.campaign_root
        / "authority/sable_label_free_v1/ABF30B_SABLE_LABEL_FREE_SUBMISSION_RECEIPT_V1.json"
    )
    output_path = authority / "ABF30B_SABLE_MEASUREMENT_SUBMISSION_RECEIPT_V1.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    plan_receipt = json.loads(plan_receipt_path.read_text())
    label_free = json.loads(label_free_submission_path.read_text())
    if (
        sha256_file(plan_path) != plan_receipt.get("plan_sha256")
        or plan_receipt.get("frozen_before_new_model_measurement") is not True
        or plan_receipt.get("fresh_blind_test") is not False
        or plan_receipt.get("measurement_only") is not True
        or label_free.get("jobs", {}).get("embeddings") != args.sable_embedding_job
        or label_free.get("bound_labels_read") is not False
        or label_free.get("model_specific_measurement_submitted") is not False
    ):
        raise ValueError("measurement submission authority differs")

    scripts = {
        "bind_authorize": args.tools_root / "bind_authorize_abf30b_sable_measurement_v1.sbatch",
        "evaluate": args.tools_root / "evaluate_abf30b_sable_component_v1.sbatch",
        "analyze": args.tools_root / "analyze_abf30b_sable_measurement_v1.sbatch",
    }
    for path in scripts.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"submission script missing or unsafe: {path}")
    upstream = f"{args.development_analyzer_job}:{args.sable_embedding_job}"
    bind_job = submit(scripts["bind_authorize"], upstream)
    evaluate_job = submit(scripts["evaluate"], bind_job)
    analyze_job = submit(scripts["analyze"], evaluate_job)
    payload = {
        "format_version": "abf30b_sable_analyst_exposed_measurement_submission_receipt_v1",
        "status": "SUBMITTED_MODEL_SPECIFIC_ANALYST_EXPOSED_SABLE_MEASUREMENT",
        "measurement_plan_sha256": plan_receipt["plan_sha256"],
        "scripts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in scripts.items()
        },
        "jobs": {
            "bind_authorize": bind_job,
            "evaluation_array": evaluate_job,
            "analyzer": analyze_job,
        },
        "dependencies": {
            bind_job: f"afterok:{upstream}",
            evaluate_job: f"afterok:{bind_job}",
            analyze_job: f"afterok:{evaluate_job}",
        },
        "six_run_mapping": {
            "0": ["baseline", 17],
            "1": ["baseline", 29],
            "2": ["baseline", 43],
            "3": ["edge_topk2_supervised", 17],
            "4": ["edge_topk2_supervised", 29],
            "5": ["edge_topk2_supervised", 43],
        },
        "fresh_blind_test": False,
        "analyst_has_seen_prior_model_results_on_dataset": True,
        "model_unseen_before_this_measurement": True,
        "measurement_only": True,
        "new_bound_label_opening": False,
        "selection_or_tuning": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(output_path, payload)
    print(json.dumps({"receipt_sha256": receipt_sha, **payload["jobs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
