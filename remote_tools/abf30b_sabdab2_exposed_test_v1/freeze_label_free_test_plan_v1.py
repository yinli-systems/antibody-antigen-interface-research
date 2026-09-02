#!/usr/bin/env python3
"""Freeze label-free preparation for one exposed SAbDab2 test measurement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
INSTANCE_MANIFEST_SHA256 = "214b9c6afb5b716751f61ccc5fcfff2e3897dfae9c49ca995b710c24e17dd7da"
FINAL_ANALYSIS_SHA256 = "c83a025c0f895ceecf61d0a079095dd182f3a8610cc419b1482604176af3b68d"
FINAL_ANALYSIS_RECEIPT_SHA256 = "2ce0e8586703dd1fd5166f35cba9b3a3eb416bc3bc5bf27c485f94d483205126"
FINAL_MODEL_SHA256 = "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--instance-manifest", type=Path, required=True)
    parser.add_argument("--final-analysis", type=Path, required=True)
    parser.add_argument("--final-analysis-receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.split, SPLIT_SHA256),
        (args.instance_manifest, INSTANCE_MANIFEST_SHA256),
        (args.final_analysis, FINAL_ANALYSIS_SHA256),
        (args.final_analysis_receipt, FINAL_ANALYSIS_RECEIPT_SHA256),
    ):
        require_sha(path, expected)
    for path in args.source:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source missing or symlinked: {path}")
    with args.split.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [row for row in rows if row["role"] == "test"]
    if len(rows) != 3663 or len(test_rows) != 549 or len({row["instance_id"] for row in test_rows}) != 549:
        raise ValueError("test split cardinality differs")
    analysis = json.loads(args.final_analysis.read_text(encoding="utf-8"))
    receipt = json.loads(args.final_analysis_receipt.read_text(encoding="utf-8"))
    if (
        analysis.get("status") != "PASS_ABF30B_ANTIBODY_REPRESENTATION_DEVELOPMENT"
        or analysis.get("abf30b_checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or receipt.get("analysis_sha256") != FINAL_ANALYSIS_SHA256
        or receipt.get("all_six_runs_verified") is not True
        or receipt.get("test_loaded") is not False
    ):
        raise ValueError("final development checkpoint authority differs")
    protocol = {
        "format_version": "abf30b_sabdab2_exposed_test_label_free_plan_v1",
        "status": "FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ",
        "measurement_role": "previously exposed historical test; model-unseen, analyst-exposed",
        "test_instance_ids_sha256": hashlib.sha256(
            canonical_bytes(sorted(row["instance_id"] for row in test_rows))
        ).hexdigest(),
        "test_complex_count": 549,
        "split_sha256": SPLIT_SHA256,
        "instance_manifest_sha256": INSTANCE_MANIFEST_SHA256,
        "final_development_analysis_sha256": FINAL_ANALYSIS_SHA256,
        "final_development_analysis_receipt_sha256": FINAL_ANALYSIS_RECEIPT_SHA256,
        "checkpoint_model_sha256": FINAL_MODEL_SHA256,
        "arms": ["baseline", "edge_topk2_supervised"],
        "seeds": [17, 29, 43],
        "task_routing": {
            "contact": "baseline same seed",
            "epitope": "edge_topk2_supervised same seed",
            "paratope": "baseline same seed",
        },
        "structure_input": {
            "source": "independent monomer AF3 predictions from sequence",
            "templates": "disabled",
            "cross_partner_geometry": False,
            "experimental_bound_coordinates_as_model_input": False,
        },
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted(args.source, key=lambda item: str(item.resolve()))
        ],
        "claim_boundary": {
            "fresh_blind_test": False,
            "old_test_previously_exposed": True,
            "historical_measurement_only": True,
            "model_unseen": True,
            "analyst_exposed": True,
            "selection_or_tuning_from_this_measurement": False,
            "result_may_not_select_checkpoint_or_model": True,
            "cumulative_test_access_ordinal": 3,
            "sable_repeated": False,
        },
        "test_sequence_metadata_already_audited": True,
        "task_labels_read_during_freeze": False,
        "temporal_holdout_loaded": False,
    }
    protocol_sha = publish(args.output, protocol)
    receipt_payload = {
        "format_version": "abf30b_sabdab2_exposed_test_label_free_plan_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": protocol_sha,
        "frozen_before_test_structure_generation": True,
        "frozen_before_test_label_artifact_read": True,
        "checkpoint_and_routing_frozen": True,
        "test_sequence_metadata_already_audited": True,
        "task_labels_read": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(args.receipt_output, receipt_payload)
    print(json.dumps({"status": protocol["status"], "protocol_sha256": protocol_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
