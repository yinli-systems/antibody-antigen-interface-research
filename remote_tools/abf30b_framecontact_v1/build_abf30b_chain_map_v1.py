#!/usr/bin/env python3
"""Build a label-free, ANARCII-authenticated antibody chain/locus map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from anarcii import Anarcii


SEQUENCE_VIEW_SHA256 = "27e0beb755c3d8e4b9910d4ad2ca979161668a1e1fe7b4d09bf6a60251e07468"
SEQUENCE_RECEIPT_SHA256 = "36a9249d5d1cec4065292e79d1efae8623987a4643675557f32252bcde7752ee"
STRUCTURE_MANIFEST_SHA256 = "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
EXPECTED_RECORDS = 3114
EXPECTED_TRAIN = 2565
EXPECTED_VALIDATION = 549
ALLOWED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX")
TYPE_MAP = {"H": "heavy", "K": "kappa", "L": "lambda"}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def contiguous_chain_lengths(chain_ids: torch.Tensor, expected_length: int) -> list[int]:
    if chain_ids.ndim != 1 or int(chain_ids.numel()) != expected_length:
        raise ValueError("antibody chain-id axis differs")
    identifiers, counts = torch.unique_consecutive(chain_ids.to(torch.int64), return_counts=True)
    observed = [int(value) for value in identifiers.tolist()]
    if observed != list(range(len(observed))) or not 1 <= len(observed) <= 2:
        raise ValueError(f"antibody chain IDs are not one or two contiguous canonical axes: {observed}")
    return [int(value) for value in counts.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--sequence-view", type=Path, required=True)
    parser.add_argument("--sequence-receipt", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    require_sha(args.protocol, args.protocol_sha256)
    require_sha(args.sequence_view, SEQUENCE_VIEW_SHA256)
    require_sha(args.sequence_receipt, SEQUENCE_RECEIPT_SHA256)
    structure_manifest_path = args.structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    require_sha(structure_manifest_path, STRUCTURE_MANIFEST_SHA256)
    protocol = json.loads(args.protocol.read_text())
    sequence_receipt = json.loads(args.sequence_receipt.read_text())
    if (
        protocol.get("status")
        != "FROZEN_BEFORE_CHAIN_TYPING_DCP_RESTORE_AND_MODEL_SPECIFIC_MEASUREMENT"
        or sequence_receipt.get("task_labels_read") is not False
        or sequence_receipt.get("test_loaded") is not False
        or sequence_receipt.get("test_rows_emitted") is not False
    ):
        raise ValueError("label-free sequence authority differs")

    sequence_rows: dict[str, dict[str, Any]] = {}
    role_counts: dict[str, int] = {"train": 0, "validation": 0}
    with args.sequence_view.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identifier = str(row["instance_id"])
            role = str(row["role"])
            sequence = str(row["ab_seq"]).strip().upper()
            if (
                identifier in sequence_rows
                or role not in role_counts
                or not 80 <= len(sequence) <= 384
                or set(sequence) - ALLOWED_RESIDUES
                or sequence_sha256(sequence) != row["ab_sequence_sha256"]
                or int(row["ab_length"]) != len(sequence)
            ):
                raise ValueError(f"invalid development sequence row: {identifier}")
            sequence_rows[identifier] = row
            role_counts[role] += 1
    if (
        len(sequence_rows) != EXPECTED_RECORDS
        or role_counts != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}
    ):
        raise ValueError("development sequence counts differ")

    structure_manifest = json.loads(structure_manifest_path.read_text())
    structure_records = structure_manifest.get("records", [])
    if len(structure_records) != EXPECTED_RECORDS:
        raise ValueError("structure record count differs")
    record_by_id = {row["instance_id"]: row for row in structure_records}
    if len(record_by_id) != EXPECTED_RECORDS or set(record_by_id) != set(sequence_rows):
        raise ValueError("sequence and structure cohorts differ")

    unique_chains: dict[str, str] = {}
    instance_chains: dict[str, tuple[tuple[str, str], ...]] = {}
    antibody_segmentation: dict[str, tuple[tuple[str, str], ...]] = {}
    for index, identifier in enumerate(sorted(sequence_rows), start=1):
        row = sequence_rows[identifier]
        structure_record = record_by_id[identifier]
        artifact_path = args.structure_root / structure_record["artifact_path"]
        if artifact_path.is_symlink() or sha256_file(artifact_path) != structure_record["artifact_sha256"]:
            raise ValueError(f"structure artifact differs: {identifier}")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        sequence = str(row["ab_seq"]).strip().upper()
        if (
            artifact.get("instance_id") != identifier
            or artifact.get("antibody_sequence_sha256") != row["ab_sequence_sha256"]
        ):
            raise ValueError(f"structure binding differs: {identifier}")
        lengths = contiguous_chain_lengths(artifact["antibody_chain_ids"], len(sequence))
        chains: list[tuple[str, str]] = []
        cursor = 0
        for length in lengths:
            chain = sequence[cursor : cursor + length]
            cursor += length
            if not 80 <= len(chain) <= 192:
                raise ValueError(f"chain length outside ABF tokenizer boundary: {identifier}")
            digest = sequence_sha256(chain)
            previous = unique_chains.setdefault(digest, chain)
            if previous != chain:
                raise AssertionError("chain SHA256 collision")
            chains.append((digest, chain))
        if cursor != len(sequence):
            raise AssertionError("chain lengths do not cover antibody axis")
        segmentation = tuple(chains)
        previous_segmentation = antibody_segmentation.setdefault(
            row["ab_sequence_sha256"], segmentation
        )
        if previous_segmentation != segmentation:
            raise ValueError("repeated antibody sequence has inconsistent chain segmentation")
        instance_chains[identifier] = segmentation
        if index % 250 == 0:
            print(f"segmented {index}/{EXPECTED_RECORDS}", flush=True)

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
        raise ValueError("ANARCII result cohort differs")
    chain_records: dict[str, dict[str, Any]] = {}
    for digest in sorted(unique_chains):
        result = numbered[digest]
        chain_type = result.get("chain_type")
        if result.get("error") is not None or chain_type not in TYPE_MAP:
            raise ValueError(f"ANARCII classification failed for {digest}: {result}")
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

    output_records: list[dict[str, Any]] = []
    type_counts = {"H": 0, "K": 0, "L": 0}
    paired_count = 0
    for identifier in sorted(instance_chains):
        row = sequence_rows[identifier]
        chains = instance_chains[identifier]
        types = [chain_records[digest]["anarcii_chain_type"] for digest, _ in chains]
        if len(types) == 2:
            paired_count += 1
            if types[0] != "H" or types[1] not in {"K", "L"}:
                raise ValueError(f"paired chain order/type differs: {identifier} {types}")
        elif types[0] not in TYPE_MAP:
            raise ValueError(f"single chain type differs: {identifier} {types}")
        for chain_type in types:
            type_counts[chain_type] += 1
        output_records.append(
            {
                "instance_id": identifier,
                "role": row["role"],
                "antibody_sequence_sha256": row["ab_sequence_sha256"],
                "antibody_sequence_length": int(row["ab_length"]),
                "chains": [
                    {
                        "axis_index": axis,
                        "sequence_sha256": digest,
                        "sequence_length": len(sequence),
                        "anarcii_chain_type": chain_records[digest]["anarcii_chain_type"],
                        "abf_chain_type": chain_records[digest]["abf_chain_type"],
                    }
                    for axis, (digest, sequence) in enumerate(chains)
                ],
            }
        )

    payload = {
        "format_version": "abf30b_sabdab2_development_chain_map_v1",
        "status": "PASS_LABEL_FREE_ANARCII_CHAIN_TYPING",
        "protocol_sha256": args.protocol_sha256,
        "sequence_view_sha256": SEQUENCE_VIEW_SHA256,
        "sequence_receipt_sha256": SEQUENCE_RECEIPT_SHA256,
        "structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
        "anarcii": {
            "version": "2.0.8",
            "seq_type": "antibody",
            "mode": "accuracy",
            "cpu": False,
        },
        "counts": {
            "records": len(output_records),
            "train": role_counts["train"],
            "validation": role_counts["validation"],
            "paired_instances": paired_count,
            "single_chain_instances": len(output_records) - paired_count,
            "unique_chains": len(chain_records),
            "chain_occurrences_by_type": type_counts,
        },
        "unique_chains": [chain_records[key] for key in sorted(chain_records)],
        "records": output_records,
        "task_labels_read": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    output_sha = publish(args.output, payload)
    receipt = {
        "format_version": "abf30b_sabdab2_development_chain_map_receipt_v1",
        "status": payload["status"],
        "chain_map_sha256": output_sha,
        "counts": payload["counts"],
        "task_labels_read": False,
        "test_loaded": False,
    }
    receipt_sha = publish(args.receipt_output, receipt)
    print(f"chain_map_sha256={output_sha}")
    print(f"receipt_sha256={receipt_sha}")


if __name__ == "__main__":
    main()
