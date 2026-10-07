#!/usr/bin/env python3
"""Build a label-free ANARCII chain map for SABLE antibody entities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from anarcii import Anarcii


INPUT_PLAN_SHA256 = "c82c12f08c87b42d21aee31cc555373ffbc4211415d9aff302444baefc971ad6"
STRUCTURE_MANIFEST_SHA256 = "795223b2ebd02d6ce37cebf7dadafe5b1f50b9b212976ba15e36af3e62f23259"
TYPE_MAP = {"H": "heavy", "K": "kappa", "L": "lambda"}
ALLOWED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


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


def require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    require_sha(args.protocol, args.protocol_sha256)
    require_sha(args.input_plan, INPUT_PLAN_SHA256)
    require_sha(args.structure_manifest, STRUCTURE_MANIFEST_SHA256)
    protocol = json.loads(args.protocol.read_text())
    plan = json.loads(args.input_plan.read_text())
    structures = json.loads(args.structure_manifest.read_text())
    if (
        protocol.get("status")
        != "FROZEN_BEFORE_SABLE_LABEL_FREE_CHAIN_TYPING_AND_MIXED_EMBEDDINGS"
        or protocol.get("task_labels_read") is not False
        or plan.get("status") != "PASS_FROZEN_LABEL_FREE_AF3_INPUT_PLAN"
        or plan.get("task_labels_read") is not False
        or structures.get("status") != "PASS_LABEL_FREE_SEQUENCE_PREDICTED_STRUCTURE_CACHE"
        or structures.get("authority", {}).get("task_labels_read") is not False
    ):
        raise ValueError("SABLE label-free authority differs")

    unique_chains: dict[str, str] = {}
    antibodies: dict[str, dict[str, Any]] = {}
    entry_to_antibody: dict[str, str] = {}
    for entity in plan.get("entities", []):
        if entity.get("entity_type") != "antibody":
            continue
        sequences = [str(value).strip().upper() for value in entity["sequences"]]
        if not 1 <= len(sequences) <= 2:
            raise ValueError("antibody entity chain count differs")
        chain_rows: list[dict[str, Any]] = []
        for axis, sequence in enumerate(sequences):
            if not 80 <= len(sequence) <= 192 or set(sequence) - ALLOWED_RESIDUES:
                raise ValueError("antibody chain is outside ABF tokenizer authority")
            digest = sequence_sha256(sequence)
            previous = unique_chains.setdefault(digest, sequence)
            if previous != sequence:
                raise AssertionError("sequence SHA collision")
            chain_rows.append(
                {
                    "axis_index": axis,
                    "sequence_sha256": digest,
                    "sequence_length": len(sequence),
                }
            )
        combined = "".join(sequences)
        combined_sha = sequence_sha256(combined)
        if combined_sha in antibodies:
            raise ValueError("duplicate antibody entity sequence")
        antibodies[combined_sha] = {
            "entity_id": entity["entity_id"],
            "antibody_sequence_sha256": combined_sha,
            "antibody_sequence_length": len(combined),
            "chains": chain_rows,
            "entry_ids": sorted(str(value) for value in entity["entry_ids"]),
        }
        for entry_id in entity["entry_ids"]:
            if str(entry_id) in entry_to_antibody:
                raise ValueError("SABLE entry has repeated antibody entity")
            entry_to_antibody[str(entry_id)] = combined_sha
    if len(antibodies) != 96 or len(entry_to_antibody) != 97:
        raise ValueError("SABLE antibody cohort count differs")

    structure_by_entry = {str(row["entry_id"]): row for row in structures["records"]}
    if set(structure_by_entry) != set(entry_to_antibody):
        raise ValueError("SABLE plan and structure entry cohorts differ")
    for entry_id, antibody_sha in entry_to_antibody.items():
        row = structure_by_entry[entry_id]
        if (
            row["antibody_sequence_sha256"] != antibody_sha
            or int(row["antibody_length"])
            != int(antibodies[antibody_sha]["antibody_sequence_length"])
        ):
            raise ValueError(f"SABLE antibody axis differs: {entry_id}")

    classifier = Anarcii(
        seq_type="antibody",
        mode="accuracy",
        batch_size=args.batch_size,
        cpu=False,
        verbose=False,
        low_score_warn=False,
    )
    numbered = classifier.number(unique_chains)
    if set(numbered) != set(unique_chains):
        raise ValueError("ANARCII output cohort differs")
    chain_records: dict[str, dict[str, Any]] = {}
    for digest in sorted(unique_chains):
        result = numbered[digest]
        chain_type = result.get("chain_type")
        if result.get("error") is not None or chain_type not in TYPE_MAP:
            raise ValueError(f"ANARCII chain classification failed: {digest}")
        chain_records[digest] = {
            "sequence_sha256": digest,
            "sequence": unique_chains[digest],
            "sequence_length": len(unique_chains[digest]),
            "anarcii_chain_type": chain_type,
            "abf_chain_type": TYPE_MAP[chain_type],
            "score": float(result["score"]),
            "query_start": int(result["query_start"]),
            "query_end": int(result["query_end"]),
            "error": None,
        }

    type_counts = {"H": 0, "K": 0, "L": 0}
    paired = 0
    records: list[dict[str, Any]] = []
    for antibody_sha in sorted(antibodies):
        row = antibodies[antibody_sha]
        types = [
            chain_records[chain["sequence_sha256"]]["anarcii_chain_type"]
            for chain in row["chains"]
        ]
        if len(types) == 2:
            paired += 1
            if types[0] != "H" or types[1] not in {"K", "L"}:
                raise ValueError(f"paired SABLE chain order differs: {row['entity_id']} {types}")
        elif types[0] not in TYPE_MAP:
            raise ValueError("single SABLE chain type differs")
        for chain_type in types:
            type_counts[chain_type] += 1
        records.append(
            {
                **row,
                "chains": [
                    {
                        **chain,
                        "anarcii_chain_type": chain_records[chain["sequence_sha256"]][
                            "anarcii_chain_type"
                        ],
                        "abf_chain_type": chain_records[chain["sequence_sha256"]][
                            "abf_chain_type"
                        ],
                    }
                    for chain in row["chains"]
                ],
            }
        )
    payload = {
        "format_version": "abf30b_sable_chain_map_v1",
        "status": "PASS_LABEL_FREE_SABLE_ANARCII_CHAIN_TYPING",
        "protocol_sha256": args.protocol_sha256,
        "input_plan_sha256": INPUT_PLAN_SHA256,
        "structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
        "anarcii": {"version": "2.0.8", "mode": "accuracy", "cpu": False},
        "counts": {
            "entries": len(entry_to_antibody),
            "unique_antibodies": len(records),
            "unique_chains": len(chain_records),
            "paired_antibodies": paired,
            "single_chain_antibodies": len(records) - paired,
            "chain_occurrences_by_type": type_counts,
        },
        "unique_chains": [chain_records[key] for key in sorted(chain_records)],
        "records": records,
        "task_labels_read": False,
        "bound_label_artifacts_read": False,
        "performance_read": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    result_sha = publish(args.output, payload)
    receipt = {
        "format_version": "abf30b_sable_chain_map_receipt_v1",
        "status": payload["status"],
        "chain_map_sha256": result_sha,
        "counts": payload["counts"],
        "task_labels_read": False,
        "test_loaded": False,
    }
    receipt_sha = publish(args.receipt_output, receipt)
    print(json.dumps({"chain_map_sha256": result_sha, "receipt_sha256": receipt_sha}))


if __name__ == "__main__":
    main()
