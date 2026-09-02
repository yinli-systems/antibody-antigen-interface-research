#!/usr/bin/env python3
"""Build the frozen test sequence view and query-only AF3 SD-H inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SEEDS = (42, 100, 18, 37, 79)
TIMESTAMP_SUFFIX = re.compile(r"_[0-9]{8}_[0-9]{6}$")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--instance-manifest", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--antigen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink() or args.batch_size != 20:
        raise ValueError("output exists or batch size differs")
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol SHA differs")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ"
        or protocol.get("test_sequence_metadata_already_audited") is not True
        or protocol.get("task_labels_read_during_freeze") is not False
    ):
        raise ValueError("label-free protocol differs")
    if sha256_file(args.split) != protocol["split_sha256"] or sha256_file(args.instance_manifest) != protocol["instance_manifest_sha256"]:
        raise ValueError("sequence source authority differs")
    with args.split.open(newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    test_by_id = {row["instance_id"]: row for row in split_rows if row["role"] == "test"}
    with args.instance_manifest.open(newline="") as handle:
        manifest_rows = [row for row in csv.DictReader(handle) if row["instance_id"] in test_by_id]
    if len(test_by_id) != 549 or len(manifest_rows) != 549:
        raise ValueError("test cohort differs")
    args.output_root.mkdir(parents=True)
    sequence_path = args.output_root / "SABDAB2_TEST_SEQUENCE_VIEW_V1.jsonl"
    sequence_hasher = hashlib.sha256()
    rows_by_id = {row["instance_id"]: row for row in manifest_rows}
    with sequence_path.open("xb") as handle:
        for identifier in sorted(test_by_id):
            row = rows_by_id[identifier]
            ab = (row["vh_seq"] + row["vl_seq"]).strip().upper()
            ag = row["ag_seq"].strip().upper()
            split_row = test_by_id[identifier]
            if hashlib.sha256(ab.encode()).hexdigest() != split_row["antibody_sequence_sha256"] or hashlib.sha256(ag.encode()).hexdigest() != split_row["antigen_sequence_sha256"]:
                raise ValueError(f"test sequence SHA differs: {identifier}")
            payload = canonical_bytes({
                "instance_id": identifier, "role": "test", "ab_seq": ab,
                "ab_length": len(ab), "ab_sequence_sha256": split_row["antibody_sequence_sha256"],
                "ag_seq": ag, "ag_length": len(ag), "ag_sequence_sha256": split_row["antigen_sequence_sha256"],
            })
            handle.write(payload); sequence_hasher.update(payload)
    sequence_sha = sequence_hasher.hexdigest()

    source_index: dict[str, list[Path]] = {}
    for directory in args.paired_root.iterdir():
        if directory.is_dir():
            source_index.setdefault(TIMESTAMP_SUFFIX.sub("", directory.name).lower(), []).append(directory)
    paired_count = 0
    for row in manifest_rows:
        if row["type"] == "SD-H":
            continue
        paired_count += 1
        key = f"{row['instance_id']}_{row['type']}".lower()
        if len(source_index.get(key, [])) != 1:
            raise ValueError(f"paired prediction coverage differs: {key}")
        if not (args.antigen_root / row["ag_seq_hash"].strip().lower()).is_dir():
            raise ValueError(f"antigen prediction missing: {row['instance_id']}")
    if paired_count != 445:
        raise ValueError("paired test count differs")

    by_sequence: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        if row["type"] != "SD-H":
            continue
        sequence = row["vh_seq"].strip().upper()
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        item = by_sequence.setdefault(digest, {"sequence": sequence, "instance_ids": []})
        if item["sequence"] != sequence:
            raise AssertionError("sequence SHA collision")
        item["instance_ids"].append(row["instance_id"])
    if len(by_sequence) != 59 or sum(len(item["instance_ids"]) for item in by_sequence.values()) != 104:
        raise ValueError("SD-H test count differs")
    records = []
    for index, digest in enumerate(sorted(by_sequence)):
        item = by_sequence[digest]
        batch_index = index // args.batch_size
        name = f"sabdab2_test_sdh_{digest[:24]}"
        input_path = args.output_root / "sdh_af3" / "batches" / f"batch-{batch_index:03d}" / "inputs" / f"{name}.json"
        fold_input = {
            "dialect": "alphafold3", "version": 1, "name": name,
            "sequences": [{"protein": {"id": "A", "sequence": item["sequence"], "modifications": [], "unpairedMsa": "", "pairedMsa": "", "templates": []}}],
            "modelSeeds": list(SEEDS), "bondedAtomPairs": None, "userCCD": None,
        }
        input_sha = publish(input_path, fold_input)
        records.append({
            "index": index, "batch_index": batch_index, "name": name,
            "sequence_sha256": digest, "sequence_length": len(item["sequence"]),
            "instance_count": len(item["instance_ids"]), "instance_ids": sorted(item["instance_ids"]),
            "input_path": str(input_path), "input_sha256": input_sha,
        })
    plan = {
        "format_version": "sabdab2_exposed_test_sdh_af3_input_plan_v1",
        "status": "PASS_EXPOSED_TEST_LABEL_FREE_INPUT_PLAN",
        "protocol_sha256": args.protocol_sha256,
        "allowed_roles": ["test"], "test_access": True, "task_labels_loaded": False,
        "temporal_holdout_access": False, "templates": "disabled", "unpaired_msa": "empty", "paired_msa": "empty",
        "model_seeds": list(SEEDS), "batch_size": args.batch_size, "batch_count": 3,
        "unique_sequence_count": len(records), "covered_instance_count": 104,
        "paired_existing_prediction_count": paired_count, "antigen_existing_prediction_count": 549,
        "sequence_view_path": str(sequence_path), "sequence_view_sha256": sequence_sha,
        "records": records,
    }
    plan_sha = publish(args.output_root / "TEST_INPUT_PLAN_V1.json", plan)
    receipt_sha = publish(args.output_root / "TEST_INPUT_PLAN_V1_RECEIPT.json", {
        "format_version": "sabdab2_exposed_test_input_plan_receipt_v1", "status": plan["status"],
        "plan_sha256": plan_sha, "sequence_view_sha256": sequence_sha,
        "input_count": 59, "batch_count": 3, "task_labels_loaded": False,
        "test_access": True, "temporal_holdout_access": False,
    })
    print(json.dumps({"status": plan["status"], "plan_sha256": plan_sha, "receipt_sha256": receipt_sha, "sdh_unique": 59, "batches": 3}, sort_keys=True))


if __name__ == "__main__":
    main()
