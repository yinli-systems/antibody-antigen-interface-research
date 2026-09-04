#!/usr/bin/env python3
"""Add a verified split/index layer to an existing test-455 embedding cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--prepared-instances", type=Path, required=True)
    parser.add_argument("--expected-instance-count", type=int, default=455)
    parser.add_argument("--expected-id-set-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handoff_root = args.handoff_root.resolve()
    split_source = args.split_json.resolve()
    prepared_source = args.prepared_instances.resolve()
    required = (
        handoff_root / "REQUESTED_INSTANCE_IDS.txt",
        handoff_root / "INSTANCE_EMBEDDING_MAP.csv",
        handoff_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json",
        handoff_root / "EXPORT_RECEIPT.json",
        split_source,
        prepared_source,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required handoff inputs: {missing}")

    outputs = {
        "split": handoff_root / "SABDAB2_455_SPLIT.json",
        "index": handoff_root / "SABDAB2_TEST455_INSTANCE_INDEX.parquet",
        "loader": handoff_root / "LOADER_PATHS.json",
        "readme": handoff_root / "README.md",
        "receipt": handoff_root / "CACHE_BUNDLE_RECEIPT.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing bundle outputs: {existing}")

    split_payload = json.loads(split_source.read_text(encoding="utf-8"))
    split_test = sorted(map(str, split_payload.get("splits", {}).get("test", [])))
    requested = sorted(
        line.strip()
        for line in (handoff_root / "REQUESTED_INSTANCE_IDS.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )
    if len(split_test) != args.expected_instance_count or len(set(split_test)) != len(split_test):
        raise ValueError("Split test membership is not the expected unique test-455 set")
    if split_test != requested:
        raise ValueError("Split test membership differs from the exported embedding instances")
    id_set_sha256 = hashlib.sha256(("\n".join(split_test) + "\n").encode()).hexdigest()
    if id_set_sha256 != args.expected_id_set_sha256:
        raise ValueError("Test-455 ID-set SHA-256 differs from the frozen authority")

    with (handoff_root / "INSTANCE_EMBEDDING_MAP.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        mapping = list(csv.DictReader(handle))
    mapping_ids = sorted(row["instance_id"] for row in mapping)
    if mapping_ids != split_test:
        raise ValueError("Instance-to-embedding map differs from the test split")

    required_columns = [
        "instance_id",
        "ab_sequence_sha256",
        "ag_sequence_sha256",
        "ab_length",
        "ag_length",
        "frozen_split",
    ]
    prepared = pd.read_parquet(prepared_source, columns=required_columns)
    selected = prepared.loc[prepared["instance_id"].astype(str).isin(split_test), required_columns]
    selected = selected.sort_values("instance_id").reset_index(drop=True)
    if selected["instance_id"].astype(str).tolist() != split_test:
        raise ValueError("Prepared instance authority does not exactly cover test-455")
    if set(selected["frozen_split"].astype(str)) != {"test"}:
        raise ValueError("Prepared instance authority contains a non-test role")

    mapping_by_id = {row["instance_id"]: row for row in mapping}
    for record in selected.to_dict("records"):
        mapped = mapping_by_id[str(record["instance_id"])]
        for short, role in (("ab", "antibody"), ("ag", "antigen")):
            key = str(record[f"{short}_sequence_sha256"])
            relative = f"{key[:2]}/{key}.pt"
            if (mapped[f"{role}_sequence_sha256"] != key
                    or int(mapped[f"{role}_length"]) != int(record[f"{short}_length"])
                    or mapped[f"{role}_artifact_path"] != relative
                    or not (handoff_root / relative).is_file()):
                raise ValueError(f"Prepared index and embedding map differ: {record['instance_id']}")

    export_receipt = json.loads(
        (handoff_root / "EXPORT_RECEIPT.json").read_text(encoding="utf-8")
    )
    if (export_receipt.get("status") != "PASS_MODEL_READY_ESM2_SUBSET_EXPORT"
            or export_receipt.get("requested_instance_ids_sha256") != id_set_sha256):
        raise ValueError("Original export receipt does not bind the same passing test-455 export")
    for filename, field in (
        ("INSTANCE_EMBEDDING_MAP.csv", "instance_embedding_map_sha256"),
        ("SABDAB2_ESM_EMBEDDING_MANIFEST.json", "subset_embedding_manifest_sha256"),
    ):
        if sha256_file(handoff_root / filename) != export_receipt[field]:
            raise ValueError(f"Export authority changed: {filename}")

    temporary_index = outputs["index"].with_name(f".{outputs['index'].name}.{os.getpid()}.tmp")
    selected.to_parquet(temporary_index, index=False, compression="zstd")
    os.replace(temporary_index, outputs["index"])
    atomic_write(outputs["split"], canonical_json_bytes(split_payload))

    loader = {
        "format_version": "sabdab2_test455_loader_paths_v1",
        "status": "PASS_MODEL_READY_TEST455_LOADER_BUNDLE",
        "instance_count": len(split_test),
        "test_id_set_sha256": id_set_sha256,
        "embedding_root": ".",
        "embedding_manifest": "SABDAB2_ESM_EMBEDDING_MANIFEST.json",
        "instance_embedding_map": "INSTANCE_EMBEDDING_MAP.csv",
        "instance_index": outputs["index"].name,
        "split_manifest": outputs["split"].name,
        "split_role": "test",
        "note": (
            "Paths are relative to this directory. The embedding root follows the "
            "sequence-SHA-addressed layout consumed by EmbeddingStore and "
            "Sabdab2StructureEmbeddingDataset."
        ),
    }
    atomic_write(outputs["loader"], canonical_json_bytes(loader))

    readme = f"""# SAbDab2 frozen test-455 ESM-2 loader bundle

This directory is a directly loadable, sequence-SHA-addressed ESM-2 cache for
the canonical 455-instance SAbDab2 test split.

- embedding root: this directory
- embedding manifest: `SABDAB2_ESM_EMBEDDING_MANIFEST.json`
- instance map: `INSTANCE_EMBEDDING_MAP.csv`
- label-free instance index: `{outputs['index'].name}`
- split manifest: `{outputs['split'].name}` (`test`, 455 instances)
- machine-readable paths: `{outputs['loader'].name}`

The sorted test ID-set SHA-256 is `{id_set_sha256}`. The bundle contains
embeddings and label-free identity/length metadata only; it does not contain
contact labels, experimental coordinates, predictions, or metrics.
"""
    atomic_write(outputs["readme"], readme.encode())

    receipt = {
        "format_version": "sabdab2_test455_loader_bundle_receipt_v1",
        "status": "PASS_MODEL_READY_TEST455_LOADER_BUNDLE",
        "counts": {
            "instances": len(split_test),
            "mapping_rows": len(mapping),
            "unique_sequences": int(export_receipt["counts"]["unique_sequences"]),
        },
        "test_id_set_sha256": id_set_sha256,
        "source": {
            "split_json_sha256": sha256_file(split_source),
            "prepared_instances_sha256": sha256_file(prepared_source),
            "export_receipt_sha256": sha256_file(handoff_root / "EXPORT_RECEIPT.json"),
            "embedding_manifest_sha256": sha256_file(
                handoff_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
            ),
        },
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "receipt"
        },
        "checks": {
            "split_equals_requested_ids": True,
            "mapping_equals_split": True,
            "prepared_index_equals_split": True,
            "all_roles_test": True,
            "task_labels_included": False,
            "experimental_coordinates_included": False,
        },
    }
    atomic_write(outputs["receipt"], canonical_json_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
