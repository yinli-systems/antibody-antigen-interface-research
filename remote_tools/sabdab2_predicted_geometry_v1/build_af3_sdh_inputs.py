#!/usr/bin/env python3
"""Build label-free AlphaFold 3 inputs for unique SAbDab2 SD-H sequences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (42, 100, 18, 37, 79)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allowed-role",
        action="append",
        choices=("train", "validation"),
        dest="allowed_roles",
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.seeds:
        raise ValueError("at least one model seed is required")

    with args.split_csv.open(newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    split_required = {"instance_id", "role"}
    split_missing = split_required.difference(split_rows[0] if split_rows else {})
    if split_missing:
        raise ValueError(f"split is missing columns: {sorted(split_missing)}")
    role_by_id: dict[str, str] = {}
    for row in split_rows:
        instance_id = row["instance_id"].strip()
        role = row["role"].strip()
        if instance_id in role_by_id or role not in {"train", "validation", "test"}:
            raise ValueError("split contains duplicate IDs or invalid roles")
        role_by_id[instance_id] = role
    allowed_roles = frozenset(args.allowed_roles)
    if allowed_roles != {"train", "validation"}:
        raise ValueError("this development-only builder requires train and validation roles")

    with args.manifest_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"instance_id", "type", "vh_seq", "vh_seq_hash"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    by_sequence: dict[str, dict[str, Any]] = {}
    for row in rows:
        instance_id = row["instance_id"].strip()
        if instance_id not in role_by_id:
            raise ValueError(f"manifest instance is absent from split: {instance_id}")
        # Fail closed before using any sequence field from a non-development row.
        if role_by_id[instance_id] not in allowed_roles:
            continue
        if row["type"] != "SD-H":
            continue
        sequence = row["vh_seq"].strip().upper()
        sequence_hash = row["vh_seq_hash"].strip().lower()
        if not sequence or not sequence_hash:
            raise ValueError(f"empty SD-H sequence/hash for {row['instance_id']}")
        sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
        item = by_sequence.setdefault(
            sequence,
            {
                "sequence_sha256": sequence_sha256,
                "sequence": sequence,
                "manifest_vh_seq_hashes": [],
                "instance_ids": [],
            },
        )
        item["manifest_vh_seq_hashes"].append(sequence_hash)
        item["instance_ids"].append(instance_id)

    items = sorted(by_sequence.values(), key=lambda item: item["sequence_sha256"])
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        items = items[: args.limit]

    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        batch_index = index // args.batch_size
        name = f"sabdab2_sdh_{item['sequence_sha256'][:24]}"
        input_path = (
            args.output_root
            / "batches"
            / f"batch-{batch_index:03d}"
            / "inputs"
            / f"{name}.json"
        )
        fold_input = {
            "dialect": "alphafold3",
            "version": 1,
            "name": name,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": item["sequence"],
                        "modifications": [],
                        # Empty MSA/templates are explicit: this is a sequence-only,
                        # no-template structure prediction input.
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            ],
            "modelSeeds": list(args.seeds),
            "bondedAtomPairs": None,
            "userCCD": None,
        }
        input_sha256 = write_json(input_path, fold_input)
        records.append(
            {
                "index": index,
                "batch_index": batch_index,
                "name": name,
                "sequence_sha256": item["sequence_sha256"],
                "manifest_vh_seq_hashes": sorted(set(item["manifest_vh_seq_hashes"])),
                "sequence_length": len(item["sequence"]),
                "instance_count": len(item["instance_ids"]),
                "instance_ids": sorted(item["instance_ids"]),
                "input_path": str(input_path),
                "input_sha256": input_sha256,
            }
        )

    plan = {
        "format_version": "sabdab2_sdh_af3_input_plan_v1",
        "task_labels_loaded": False,
        "test_access": False,
        "temporal_holdout_access": False,
        "source_manifest_path": str(args.manifest_csv),
        "source_manifest_sha256": sha256_file(args.manifest_csv),
        "source_split_path": str(args.split_csv),
        "source_split_sha256": sha256_file(args.split_csv),
        "allowed_roles": sorted(allowed_roles),
        "selection": "unique train/validation SD-H VH sequences, sorted by sequence SHA-256",
        "prediction_input": {
            "predictor": "AlphaFold 3",
            "templates": "disabled",
            "unpaired_msa": "empty",
            "paired_msa": "empty",
            "model_seeds": list(args.seeds),
        },
        "batch_size": args.batch_size,
        "unique_sequence_count": len(records),
        "covered_instance_count": sum(record["instance_count"] for record in records),
        "covered_role_counts": {
            role: sum(
                1
                for record in records
                for instance_id in record["instance_ids"]
                if role_by_id[instance_id] == role
            )
            for role in sorted(allowed_roles)
        },
        "batch_count": (len(records) + args.batch_size - 1) // args.batch_size,
        "records": records,
    }
    plan_sha256 = write_json(args.output_root / "AF3_SDH_INPUT_PLAN.json", plan)
    receipt = {
        "format_version": "sabdab2_sdh_af3_input_plan_receipt_v1",
        "plan_path": str(args.output_root / "AF3_SDH_INPUT_PLAN.json"),
        "plan_sha256": plan_sha256,
        "input_count": len(records),
        "all_input_sha256_verified": all(
            sha256_file(Path(record["input_path"])) == record["input_sha256"]
            for record in records
        ),
        "task_labels_loaded": False,
        "test_access": False,
        "temporal_holdout_access": False,
    }
    write_json(args.output_root / "AF3_SDH_INPUT_PLAN_RECEIPT.json", receipt)

    print(
        json.dumps(
            {
                "status": "PASS_INPUT_PLAN",
                "unique_sequence_count": len(records),
                "covered_instance_count": plan["covered_instance_count"],
                "batch_count": plan["batch_count"],
                "plan_sha256": plan_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
