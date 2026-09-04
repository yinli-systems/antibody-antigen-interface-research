#!/usr/bin/env python3
"""Verify the portable embedding subset through the actual EmbeddingStore."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--expected-id-set-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite a verification receipt")
    require(digest(args.runtime) == args.runtime_sha256, "Runtime SHA mismatch")
    spec = importlib.util.spec_from_file_location("handoff_embedding_runtime", args.runtime)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    torch.set_num_threads(1)
    root = args.bundle.resolve()
    receipt = json.loads((root / "CACHE_BUNDLE_RECEIPT.json").read_text())
    export = json.loads((root / "EXPORT_RECEIPT.json").read_text())
    for name, field in (
        ("SABDAB2_ESM_EMBEDDING_MANIFEST.json", "subset_embedding_manifest_sha256"),
        ("INSTANCE_EMBEDDING_MAP.csv", "instance_embedding_map_sha256"),
        ("REQUESTED_INSTANCE_IDS.txt", "requested_instance_ids_sha256"),
    ):
        require(digest(root / name) == export[field], f"Export hash mismatch: {name}")
    require(digest(root / "EXPORT_RECEIPT.json") == receipt["source"]["export_receipt_sha256"],
            "Original export receipt changed")
    for item in receipt["outputs"].values():
        path = root / item["path"]
        require(path.resolve().is_relative_to(root), "Non-local bundle artifact")
        require(digest(path) == item["sha256"], f"Bundle hash mismatch: {path.name}")
    mapping = pd.read_csv(root / "INSTANCE_EMBEDDING_MAP.csv").set_index("instance_id")
    index = pd.read_parquet(root / "SABDAB2_TEST455_INSTANCE_INDEX.parquet").set_index("instance_id")
    splits = json.loads((root / "SABDAB2_455_SPLIT.json").read_text())["splits"]
    ids = sorted(index.index.astype(str))
    require(len(ids) == len(set(ids)) == 455, "Test index is not 455 unique IDs")
    require(sorted(mapping.index.astype(str)) == sorted(splits["test"]) == ids,
            "Split, map, and index disagree")
    require(set(index.frozen_split) == {"test"}, "Non-test row in index")
    id_hash = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    require(id_hash == args.expected_id_set_sha256 == receipt["test_id_set_sha256"],
            "Frozen test ID set mismatch")
    manifest = json.loads((root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json").read_text())
    records = {r["sequence_sha256"]: r for r in manifest["records"]}
    require(len(records) == len(manifest["records"]) == 514, "Duplicate/missing embedding records")
    required_keys = set(index.ab_sequence_sha256) | set(index.ag_sequence_sha256)
    require(set(records) == required_keys, "Manifest and instance sequence keys disagree")
    # Hash each unique tensor once; load every Ab/Ag instance through production code.
    for key, record in records.items():
        relative = f"{key[:2]}/{key}.pt"
        require(record["artifact_path"] == relative, "Noncanonical embedding path")
        require(digest(root / relative) == record["artifact_sha256"], f"Embedding hash mismatch: {key}")
    store = runtime.EmbeddingStore(root, max_cache_gib=0.75)
    for instance_id in ids:
        row, mapped = index.loc[instance_id], mapping.loc[instance_id]
        for short, role in (("ab", "antibody"), ("ag", "antigen")):
            key = str(row[f"{short}_sequence_sha256"])
            length = int(row[f"{short}_length"])
            require(mapped[f"{role}_sequence_sha256"] == key
                    and int(mapped[f"{role}_length"]) == length
                    and mapped[f"{role}_artifact_path"] == f"{key[:2]}/{key}.pt",
                    f"Map/index axis mismatch: {instance_id} {role}")
            require(int(records[key]["sequence_length"]) == length, "Manifest length mismatch")
            chunks, starts, loaded_length = store.load(key, torch.device("cpu"))
            require(loaded_length == length and len(chunks) == len(starts) > 0,
                    f"EmbeddingStore length mismatch: {instance_id} {role}")
    result = {
        "status": "PASS_TEST455_EMBEDDING_LOADER_SMOKE",
        "instances": len(ids), "unique_embeddings": len(records), "role_loads": 2 * len(ids),
        "test_id_set_sha256": id_hash,
        "runtime_sha256": args.runtime_sha256,
        "verifier_sha256": digest(Path(__file__)),
        "cache_bundle_receipt_sha256": digest(root / "CACHE_BUNDLE_RECEIPT.json"),
        "embedding_manifest_sha256": digest(root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"),
        "torch_version": str(torch.__version__), "loader": "interaction_first.runtime.EmbeddingStore",
        "checks": {"all_artifact_hashes": True, "map_index_lengths_and_sequence_keys": True,
                   "all_role_loads_via_production_loader": True},
        "scope": "ESM-2 embeddings and test index only",
        "full_model_inference_executed": False,
        "requester_full_cache_schema_verified": False,
        "structure_cache_included": False, "task_labels_included": False,
        "store": store.summary(),
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
