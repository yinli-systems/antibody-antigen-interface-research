#!/usr/bin/env python3
"""Build frozen label-free AF3 inputs for the eligible SABLE external cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCREEN_RESULT_SHA256 = (
    "2c6f6ec9f9b8ff6fd11112a9131fdd9fa024591bc99df36e42caecd8819eca00"
)
SCREEN_RECEIPT_SHA256 = (
    "5aee4ff1c98a9936674d68aae74d28e90f6d9ba357464f7776f72c7d5cbd73d6"
)
MODEL_SEEDS = (42, 100, 18, 37, 79)
EXPECTED_ELIGIBLE_ENTRIES = 97
EXPECTED_ELIGIBLE_PDB_IDS = 88


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> str:
    path.write_bytes(canonical_bytes(payload))
    return sha256_file(path)


def entity_digest(entity_type: str, sequences: tuple[str, ...]) -> str:
    return hashlib.sha256(
        canonical_bytes({"entity_type": entity_type, "sequences": sequences})
    ).hexdigest()


def af3_input(name: str, sequences: tuple[str, ...], chain_ids: tuple[str, ...]) -> dict[str, Any]:
    if len(sequences) != len(chain_ids) or not sequences:
        raise ValueError("AF3 entity sequence/chain cardinality differs")
    return {
        "dialect": "alphafold3",
        "version": 1,
        "name": name,
        "sequences": [
            {
                "protein": {
                    "id": chain_id,
                    "sequence": sequence,
                    "modifications": [],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                }
            }
            for chain_id, sequence in zip(chain_ids, sequences, strict=True)
        ],
        "modelSeeds": list(MODEL_SEEDS),
        "bondedAtomPairs": None,
        "userCCD": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--screen-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if args.batch_size != 10:
        raise ValueError("the frozen external input protocol requires batch size 10")
    if sha256_file(args.screen_result) != SCREEN_RESULT_SHA256:
        raise ValueError("label-free screen result SHA differs")
    if sha256_file(args.screen_receipt) != SCREEN_RECEIPT_SHA256:
        raise ValueError("label-free screen receipt SHA differs")

    screen = json.loads(args.screen_result.read_text(encoding="utf-8"))
    receipt = json.loads(args.screen_receipt.read_text(encoding="utf-8"))
    eligible = [record for record in screen["records"] if record["eligible"]]
    eligible_pdb_ids = {record["pdb_id"] for record in eligible}
    if (
        screen.get("status") != "PASS_LABEL_FREE_GATE"
        or receipt.get("status") != "PASS_LABEL_FREE_GATE"
        or receipt.get("result_sha256") != SCREEN_RESULT_SHA256
        or len(eligible) != EXPECTED_ELIGIBLE_ENTRIES
        or len(eligible_pdb_ids) != EXPECTED_ELIGIBLE_PDB_IDS
        or screen.get("task_labels_read") is not False
        or screen.get("sable_bound_structures_downloaded") is not False
        or screen.get("old_sabdab2_test_loaded") is not False
        or screen.get("temporal_holdout_loaded") is not False
        or screen.get("selection_or_tuning") is not False
    ):
        raise ValueError("label-free screen authority differs")

    entities: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    entry_bindings: list[dict[str, Any]] = []
    for record in sorted(eligible, key=lambda item: item["entry_id"]):
        heavy = str(record["heavy_sequence"])
        light = str(record["light_sequence"])
        antibody_sequences = (heavy, light) if light else (heavy,)
        antibody_key = ("antibody", antibody_sequences)
        antibody_sha = entity_digest(*antibody_key)
        antibody_id = f"sable_ab_{antibody_sha[:24]}"
        antibody = entities.setdefault(
            antibody_key,
            {
                "entity_id": antibody_id,
                "entity_type": "antibody",
                "entity_sha256": antibody_sha,
                "sequences": list(antibody_sequences),
                "chain_ids": ["H", "L"] if light else ["H"],
                "entry_ids": [],
            },
        )
        if antibody["entity_id"] != antibody_id:
            raise ValueError("antibody entity hash collision")
        antibody["entry_ids"].append(record["entry_id"])

        antigen_entity_ids: list[str] = []
        for antigen_sequence in record["antigen_chains"]:
            antigen_sequences = (str(antigen_sequence),)
            antigen_key = ("antigen_chain", antigen_sequences)
            antigen_sha = entity_digest(*antigen_key)
            antigen_id = f"sable_ag_{antigen_sha[:24]}"
            antigen = entities.setdefault(
                antigen_key,
                {
                    "entity_id": antigen_id,
                    "entity_type": "antigen_chain",
                    "entity_sha256": antigen_sha,
                    "sequences": list(antigen_sequences),
                    "chain_ids": ["A"],
                    "entry_ids": [],
                },
            )
            if antigen["entity_id"] != antigen_id:
                raise ValueError("antigen entity hash collision")
            antigen["entry_ids"].append(record["entry_id"])
            antigen_entity_ids.append(antigen_id)

        entry_bindings.append(
            {
                "entry_id": record["entry_id"],
                "pdb_id": record["pdb_id"],
                "antibody_entity_id": antibody_id,
                "antigen_entity_ids_in_declared_order": antigen_entity_ids,
                "task_labels_read": False,
                "bound_structure_read": False,
            }
        )

    entity_rows = sorted(entities.values(), key=lambda item: item["entity_id"])
    if len({row["entity_id"] for row in entity_rows}) != len(entity_rows):
        raise ValueError("entity ID collision")
    args.output_root.mkdir(parents=True)
    for index, row in enumerate(entity_rows):
        batch_index = index // args.batch_size
        batch_root = args.output_root / "batches" / f"batch-{batch_index:03d}"
        input_root = batch_root / "inputs"
        input_root.mkdir(parents=True, exist_ok=True)
        input_path = input_root / f"{row['entity_id']}.json"
        payload = af3_input(
            row["entity_id"], tuple(row["sequences"]), tuple(row["chain_ids"])
        )
        row["index"] = index
        row["batch_index"] = batch_index
        row["entry_ids"] = sorted(set(row["entry_ids"]))
        row["input_path"] = str(input_path)
        row["input_sha256"] = write_json(input_path, payload)

    antibody_count = sum(row["entity_type"] == "antibody" for row in entity_rows)
    antigen_count = sum(row["entity_type"] == "antigen_chain" for row in entity_rows)
    batch_count = (len(entity_rows) + args.batch_size - 1) // args.batch_size
    plan = {
        "format_version": "sable_v0_2_0_af3_label_free_input_plan_v1",
        "status": "PASS_FROZEN_LABEL_FREE_AF3_INPUT_PLAN",
        "authority": {
            "screen_result_sha256": SCREEN_RESULT_SHA256,
            "screen_receipt_sha256": SCREEN_RECEIPT_SHA256,
        },
        "prediction_protocol": {
            "predictor": "AlphaFold 3",
            "antibody_prediction": "H/L jointly when paired; H alone otherwise",
            "antigen_prediction": "each declared antigen chain independently",
            "antibody_antigen_joint_prediction": False,
            "templates": "disabled",
            "unpaired_msa": "empty",
            "paired_msa": "empty",
            "model_seeds": list(MODEL_SEEDS),
            "selected_model": "highest AF3 ranking score across fixed seeds/samples",
        },
        "counts": {
            "eligible_entries": len(eligible),
            "eligible_unique_pdb_ids": len(eligible_pdb_ids),
            "unique_antibody_entities": antibody_count,
            "unique_antigen_chain_entities": antigen_count,
            "prediction_entities": len(entity_rows),
            "batch_size": args.batch_size,
            "batch_count": batch_count,
        },
        "entities": entity_rows,
        "entry_bindings": entry_bindings,
        "task_labels_read": False,
        "sable_bound_structures_downloaded": False,
        "experimental_bound_geometry_used_as_prediction_input": False,
        "cross_partner_geometry_used": False,
        "selection_or_tuning": False,
    }
    plan_path = args.output_root / "AF3_LABEL_FREE_INPUT_PLAN.json"
    plan_sha256 = write_json(plan_path, plan)
    all_inputs_verified = all(
        sha256_file(Path(row["input_path"])) == row["input_sha256"]
        for row in entity_rows
    )
    if not all_inputs_verified:
        raise ValueError("AF3 input SHA verification failed")
    receipt_payload = {
        "format_version": "sable_v0_2_0_af3_label_free_input_plan_receipt_v1",
        "status": plan["status"],
        "plan_sha256": plan_sha256,
        "prediction_entities": len(entity_rows),
        "batch_count": batch_count,
        "all_input_sha256_verified": True,
        "task_labels_read": False,
        "sable_bound_structures_downloaded": False,
    }
    receipt_path = args.output_root / "AF3_LABEL_FREE_INPUT_PLAN_RECEIPT.json"
    receipt_sha256 = write_json(receipt_path, receipt_payload)
    print(
        json.dumps(
            {**receipt_payload, "receipt_sha256": receipt_sha256}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
