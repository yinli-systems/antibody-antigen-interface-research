#!/usr/bin/env python3
"""Submit contact-adapter preparation, six fixed tests, and analyzer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def submit(path: Path, dependency: str | None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(path))
    output = subprocess.check_output(command, text=True).strip()
    job = output.split(";", 1)[0]
    if not job.isdecimal():
        raise RuntimeError(output)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    recovery_path = args.campaign_root / "authority/CONTACT_AUTHORITY_ADAPTER_RECEIPT_RECOVERY_V1.json"
    recovery_receipt_path = args.campaign_root / "authority/CONTACT_AUTHORITY_ADAPTER_RECEIPT_RECOVERY_V1_RECEIPT.json"
    output = args.campaign_root / "authority/EXPOSED_TEST_CONTACT_AUTHORITY_RECOVERY_SUBMISSION_RECEIPT_V2.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery_receipt = json.loads(recovery_receipt_path.read_text(encoding="utf-8"))
    recovery_sha = sha256_file(recovery_path)
    if (
        recovery.get("status") != "FROZEN_AFTER_STRUCTURE_RECEIPT_FIELD_ALIAS_FAILURE_FOR_SAME_OPENING"
        or recovery_receipt.get("recovery_sha256") != recovery_sha
        or recovery.get("same_opening_retry") is not True
        or recovery.get("scientific_protocol_changed") is not False
    ):
        raise ValueError("contact-authority adapter recovery differs")
    prepare = args.tools_root / "prepare_contact_authority_recovery_v1.sbatch"
    evaluation = args.tools_root / "evaluate_exposed_test_contact_authority_recovery_v1.sbatch"
    analyzer = args.tools_root / "analyze_exposed_test_v1.sbatch"
    prepare_job = submit(prepare, None)
    evaluation_job = submit(evaluation, prepare_job)
    analyzer_job = submit(analyzer, evaluation_job)
    payload = {
        "format_version": "abf30b_sabdab2_exposed_test_contact_authority_recovery_submission_receipt_v1",
        "status": "SUBMITTED_SAME_OPENING_CONTACT_AUTHORITY_RECOVERY",
        "adapter_recovery_sha256": recovery_sha,
        "failed_prepare_job_id": "1501359",
        "cancelled_unstarted_evaluation_job_id": "1501360",
        "cancelled_unstarted_analyzer_job_id": "1501361",
        "jobs": {"prepare": prepare_job, "evaluation": evaluation_job, "analyzer": analyzer_job},
        "dependencies": {
            evaluation_job: f"afterok:{prepare_job}",
            analyzer_job: f"afterok:{evaluation_job}",
        },
        "scripts": {
            "prepare": {"path": str(prepare), "sha256": sha256_file(prepare)},
            "evaluation": {"path": str(evaluation), "sha256": sha256_file(evaluation)},
            "analyzer": {"path": str(analyzer), "sha256": sha256_file(analyzer)},
        },
        "same_opening_retry": True,
        "scientific_protocol_changed": False,
        "selection_performed_on_test": False,
        "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3,
    }
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    raw = canonical_bytes(payload)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
    print(json.dumps({
        "status": payload["status"],
        "jobs": payload["jobs"],
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
