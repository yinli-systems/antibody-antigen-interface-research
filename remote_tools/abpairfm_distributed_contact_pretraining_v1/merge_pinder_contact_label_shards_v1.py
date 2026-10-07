#!/usr/bin/env python3
"""Merge authenticated B32K contact-label shards in curriculum order."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

import materialize_pinder_contact_labels_v1 as base
from materialize_pinder_contact_label_shard_v1 import SHARDS, STAGE, SYSTEMS


def _curriculum_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    identifiers = [row["id"] for row in rows]
    if (
        len(identifiers) != SYSTEMS
        or len(set(identifiers)) != SYSTEMS
        or any(row["split"] != "train" for row in rows)
    ):
        raise RuntimeError("B32K curriculum authority differs")
    return identifiers


def round_robin_records(
    handles: list[object], expected_ids: list[str]
):
    for index, expected in enumerate(expected_ids):
        line = handles[index % len(handles)].readline()
        if not line:
            raise RuntimeError("B32K shard ended early")
        record = json.loads(line)
        if record.get("example_id") != expected:
            raise RuntimeError("B32K shard order differs from curriculum")
        yield record
    if any(handle.readline() for handle in handles):
        raise RuntimeError("B32K shard contains trailing records")


def _publish_file_create_only(temporary: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, output)
        output.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite output: {output}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--expected-curriculum-sha256", required=True)
    parser.add_argument("--curriculum-receipt", type=Path, required=True)
    parser.add_argument("--expected-curriculum-receipt-sha256", required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=SHARDS)
    parser.add_argument("--expected-shard-source-sha256", required=True)
    parser.add_argument("--expected-base-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.shard_count != SHARDS:
        raise ValueError("B32K shard count differs")
    for path, expected in (
        (arguments.curriculum, arguments.expected_curriculum_sha256),
        (arguments.curriculum_receipt, arguments.expected_curriculum_receipt_sha256),
    ):
        if path.is_symlink() or not path.is_file() or base.sha256_file(path) != expected:
            raise RuntimeError(f"Authenticated input differs: {path}")
    curriculum_receipt = json.loads(
        arguments.curriculum_receipt.read_text(encoding="utf-8")
    )
    if (
        curriculum_receipt.get("status")
        != "PASS_TRAIN_ONLY_NESTED_CLUSTER_BALANCED_CURRICULUM"
        or curriculum_receipt.get("checks", {}).get("test_rows_read") is not False
        or curriculum_receipt.get("stages", {}).get(STAGE, {}).get("sha256")
        != arguments.expected_curriculum_sha256
    ):
        raise RuntimeError("B32K curriculum receipt differs")
    if arguments.output.exists() or arguments.receipt.exists():
        raise FileExistsError("Refusing to overwrite merged B32K labels")
    expected_ids = _curriculum_ids(arguments.curriculum)

    receipts: list[dict[str, object]] = []
    inputs = []
    handles = []
    try:
        for index in range(SHARDS):
            label = arguments.shard_root / f"part-{index:02d}.jsonl.gz"
            receipt_path = arguments.shard_root / f"part-{index:02d}.receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            expected_shard_ids = expected_ids[index::SHARDS]
            ids_sha = hashlib.sha256(
                "".join(f"{identifier}\n" for identifier in expected_shard_ids).encode("ascii")
            ).hexdigest()
            if (
                receipt.get("status") != "PASS_B32K_PAIR_LABEL_SHARD"
                or receipt.get("shard_index") != index
                or receipt.get("shard_count") != SHARDS
                or receipt.get("ids_sha256") != ids_sha
                or receipt.get("test_access") is not False
                or receipt.get("output", {}).get("sha256") != base.sha256_file(label)
                or receipt.get("counts", {}).get("systems") != SYSTEMS // SHARDS
                or receipt.get("source_sha256", {}).get("shard_materializer")
                != arguments.expected_shard_source_sha256
                or receipt.get("source_sha256", {}).get("base_materializer")
                != arguments.expected_base_source_sha256
            ):
                raise RuntimeError(f"B32K shard receipt differs: {index}")
            receipts.append(receipt)
            inputs.append(
                {
                    "index": index,
                    "label_sha256": receipt["output"]["sha256"],
                    "receipt_sha256": base.sha256_file(receipt_path),
                }
            )
            handles.append(gzip.open(label, "rt", encoding="ascii"))

        primary = auxiliary = residues = cropped = 0
        endpoint_r: set[str] = set()
        endpoint_l: set[str] = set()
        pdbs: set[str] = set()
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".b32k-labels.", dir=arguments.output.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as zipped:
                    for record in round_robin_records(handles, expected_ids):
                        payload = (
                            json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
                            + "\n"
                        ).encode("ascii")
                        zipped.write(payload)
                        primary += len(record["contact_pairs"])
                        auxiliary += len(record["auxiliary_contact_pairs_6A"])
                        residues += len(record["sequence1"]) + len(record["sequence2"])
                        crop = record["crop"]
                        cropped += int(
                            crop["left_resolved_length"] + crop["right_resolved_length"]
                            > len(record["sequence1"]) + len(record["sequence2"])
                        )
                        endpoint_r.add(record["endpoint_cluster1"])
                        endpoint_l.add(record["endpoint_cluster2"])
                        pdbs.add(record["pdb_id"])
                raw.flush()
                os.fsync(raw.fileno())
            _publish_file_create_only(temporary, arguments.output)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        for handle in handles:
            handle.close()

    counts = {
        "auxiliary_6A_contact_pairs": auxiliary,
        "cropped_systems": cropped,
        "primary_4p5A_contact_pairs": primary,
        "published_residues": residues,
        "systems": SYSTEMS,
        "unique_endpoint_cluster_R": len(endpoint_r),
        "unique_endpoint_cluster_L": len(endpoint_l),
        "unique_pdbs": len(pdbs),
    }
    receipt = {
        "authorization": {"B32K_training": True, "B500K_training": False},
        "checks": {
            "all_curriculum_ids_accounted_once_in_order": True,
            "all_rows_train": True,
            "bound_coordinates_published_as_model_input": False,
            "contact_indices_match_published_resolved_sequence_axes": True,
            "pinder_test_rows_read": False,
            "pinder_test_structures_read": False,
        },
        "claim_boundary": (
            "B32K PINDER-Train contact supervision for train-only biological pretraining. "
            "It does not authorize PINDER validation/test, antibody adaptation, or external-test claims."
        ),
        "counts": counts,
        "format_version": "abpairfm_pinder_b32k_contact_labels_v1",
        "inputs": {
            "curriculum_receipt_sha256": arguments.expected_curriculum_receipt_sha256,
            "curriculum_sha256": arguments.expected_curriculum_sha256,
            "shards": inputs,
        },
        "output": {
            "bytes": arguments.output.stat().st_size,
            "path": arguments.output.name,
            "sha256": base.sha256_file(arguments.output),
        },
        "status": "PASS_B32K_PAIR_LABELS_MODEL_INPUT_SEQUENCE_ONLY",
        "test_access": False,
    }
    serialized = (
        json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")
    base._publish_create_only(arguments.receipt, serialized)
    print(serialized.decode("ascii"), end="")


if __name__ == "__main__":
    main()
