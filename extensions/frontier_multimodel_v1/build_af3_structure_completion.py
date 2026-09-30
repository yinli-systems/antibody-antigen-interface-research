#!/usr/bin/env python3
"""Freeze the missing-chain set and build one AF3 input per sequence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SEEDS = [42, 100, 18, 37, 79]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def canonical_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"INSTANCE", "vh_seq", "vl_seq", "ag_seq"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"canonical manifest lacks {sorted(required)}")
        for row in reader:
            instance = str(row["INSTANCE"])
            if instance in rows:
                raise ValueError(f"duplicate instance {instance}")
            rows[instance] = {
                "heavy": (row.get("vh_seq") or "").strip(),
                "light": (row.get("vl_seq") or "").strip(),
                "antigen": (row.get("ag_seq") or "").strip(),
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--base-structure-manifest", type=Path, required=True)
    parser.add_argument("--af3-runner", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    canonical_path = args.canonical_manifest.resolve()
    split_path = args.split_manifest.resolve()
    base_path = args.base_structure_manifest.resolve()
    runner_path = args.af3_runner.resolve()
    for path in (canonical_path, split_path, base_path, runner_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation = list(map(str, split["splits"]["validation"]))
    if len(validation) != 217:
        raise ValueError("validation authority must contain 217 complexes")
    forbidden = set(map(str, split["splits"]["train"])) | set(map(str, split["splits"]["test"]))
    if set(validation) & forbidden:
        raise ValueError("split firewall overlap")
    canonical = canonical_rows(canonical_path)
    if not set(validation) <= set(canonical):
        raise ValueError("canonical manifest does not cover validation")

    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("contact_labels_loaded") is not False or base.get("test_loaded") is not False:
        raise ValueError("base structure authority violates firewall")
    present = {str(record["sequence_sha256"]) for record in base["records"]}

    missing: dict[str, dict[str, Any]] = {}
    for instance in validation:
        for role, sequence in canonical[instance].items():
            if not sequence:
                continue
            key = sequence_sha(sequence)
            if key in present:
                continue
            record = missing.setdefault(
                key,
                {
                    "sequence_sha256": key,
                    "sequence": sequence,
                    "length": len(sequence),
                    "roles": set(),
                    "instances": set(),
                },
            )
            if record["sequence"] != sequence:
                raise ValueError("sequence digest collision")
            record["roles"].add(role)
            record["instances"].add(instance)

    tasks = []
    output_root = args.output_root.resolve()
    input_root = output_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=False)
    for task_id, record in enumerate(sorted(missing.values(), key=lambda item: (item["length"], item["sequence_sha256"]))):
        name = f"af3_missing_{record['sequence_sha256'][:24]}"
        batch = input_root / f"batch-{task_id:03d}"
        batch.mkdir()
        payload = {
            "name": name,
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": record["sequence"],
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            ],
            "modelSeeds": SEEDS,
            "dialect": "alphafold3",
            "version": 1,
        }
        input_path = batch / f"{name}.json"
        write_json(input_path, payload)
        tasks.append({
            "task_id": task_id,
            "name": name,
            "sequence_sha256": record["sequence_sha256"],
            "sequence": record["sequence"],
            "length": record["length"],
            "roles": sorted(record["roles"]),
            "instances": sorted(record["instances"]),
            "input_path": str(input_path),
            "input_sha256": sha256(input_path),
            "batch_dir": str(batch),
        })

    if len(tasks) != 40:
        raise ValueError(f"frozen missing set changed: expected 40, got {len(tasks)}")
    role_counts = defaultdict(int)
    for task in tasks:
        for role in task["roles"]:
            role_counts[role] += 1
    payload = {
        "format_version": "af3_structure_authority_completion_plan_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": args.source_commit,
        "canonical_manifest": str(canonical_path),
        "canonical_manifest_sha256": sha256(canonical_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256(split_path),
        "base_structure_manifest": str(base_path),
        "base_structure_manifest_sha256": sha256(base_path),
        "af3_runner": str(runner_path),
        "af3_runner_sha256": sha256(runner_path),
        "model_seeds": SEEDS,
        "input_policy": {
            "single_protein_chain": True,
            "chain_id": "A",
            "unpaired_msa": "empty",
            "paired_msa": "empty",
            "templates": [],
            "one_sequence_per_task": True,
        },
        "validation_complex_count": len(validation),
        "base_complete_complex_count": 187,
        "base_incomplete_complex_count": 30,
        "task_count": len(tasks),
        "unique_missing_residues": sum(task["length"] for task in tasks),
        "missing_role_counts": dict(role_counts),
        "tasks": tasks,
        "contact_labels_read": False,
        "test_labels_read": False,
        "ood_labels_read": False,
        "blind_claim_authorized": False,
        "sota_claim_authorized": False,
    }
    plan_path = output_root / "PLAN.json"
    write_json(plan_path, payload)
    print(json.dumps({
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "tasks": len(tasks),
        "shortest": {key: tasks[0][key] for key in ("task_id", "length", "sequence_sha256", "roles")},
        "longest": {key: tasks[-1][key] for key in ("task_id", "length", "sequence_sha256", "roles")},
        "role_counts": dict(role_counts),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
