#!/usr/bin/env python3
"""Freeze the registered-module recovery after the structure wrapper failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FAILED_STDERR_SHA256 = "a6e6d40d49ea2786f1d3bc82efb59afd0ec0499eb6cac1cc764b5c6e9ff161c3"
AF3_SUM_SHA256 = (
    "099e22c006e280214e0ea1b76330a56e179cac11a1949cfcbb35cf345e66cc34",
    "e954a4058de0f9d9b3c241b9157b3c4ea8052fe9e4bec8b0050254208d06151b",
    "04b6e0efe66e06ed7e3e4ba49abccd508934ecc5bb54dd10f1bef072c2d66929",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-protocol", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, required=True)
    parser.add_argument("--failed-output-root", type=Path, required=True)
    parser.add_argument("--af3-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    prep_sha = sha256_file(args.prep_protocol)
    prep = json.loads(args.prep_protocol.read_text(encoding="utf-8"))
    stderr_text = args.failed_stderr.read_text(encoding="utf-8")
    if (
        prep.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ"
        or prep.get("task_labels_read_during_freeze") is not False
        or sha256_file(args.failed_stderr) != FAILED_STDERR_SHA256
        or "sys.modules.get(cls.__module__).__dict__" not in stderr_text
        or "AttributeError: 'NoneType' object has no attribute '__dict__'" not in stderr_text
        or args.failed_output_root.exists()
        or args.failed_output_root.is_symlink()
    ):
        raise ValueError("structure failure evidence differs")
    observed_af3 = []
    expected_counts = (20, 20, 19)
    for index, expected_count in enumerate(expected_counts):
        batch = args.af3_root / "batches" / f"batch-{index:03d}"
        sums = batch / "RESULT_SHA256SUMS.txt"
        models = list((batch / "results").glob("*/*_model.cif"))
        if sha256_file(sums) != AF3_SUM_SHA256[index] or len(models) != expected_count:
            raise ValueError(f"AF3 batch authority differs: {index}")
        observed_af3.append({"batch_index": index, "model_count": len(models), "result_sha256s_sha256": sha256_file(sums)})
    sources = []
    for path in sorted(args.source, key=lambda item: str(item.resolve())):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"recovery source missing: {path}")
        sources.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    protocol = {
        "format_version": "abf30b_sabdab2_exposed_test_structure_exec_recovery_v1",
        "status": "FROZEN_AFTER_STRUCTURE_EXEC_FAILURE_BEFORE_LABEL_ARTIFACT_READ",
        "prep_protocol_sha256": prep_sha,
        "failed_job_id": "1500379",
        "failed_job_exit_code": "1:0",
        "failed_stderr_sha256": FAILED_STDERR_SHA256,
        "failure_class": "Python dynamic-module registration required by dataclasses",
        "failure_before_structure_iteration": True,
        "failed_output_root_created": False,
        "af3_batches": observed_af3,
        "af3_model_count": 59,
        "recovery_change": "register transformed source in sys.modules during execution",
        "scientific_protocol_changed": False,
        "cohort_changed": False,
        "prediction_files_changed": False,
        "checkpoint_or_routing_changed": False,
        "test_label_artifacts_read": False,
        "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3,
        "sources": sources,
    }
    protocol_sha = publish(args.output, protocol)
    receipt_sha = publish(args.receipt_output, {
        "format_version": "abf30b_sabdab2_exposed_test_structure_exec_recovery_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": protocol_sha,
        "failed_stderr_sha256": FAILED_STDERR_SHA256,
        "all_af3_batches_verified": True,
        "scientific_protocol_changed": False,
        "test_label_artifacts_read": False,
    })
    print(json.dumps({"status": protocol["status"], "protocol_sha256": protocol_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
