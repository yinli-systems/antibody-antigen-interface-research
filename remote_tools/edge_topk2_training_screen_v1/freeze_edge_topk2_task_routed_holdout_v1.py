#!/usr/bin/env python3
"""Freeze an untouched cluster-disjoint holdout for deterministic task routing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


FULL_SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
OLD_PARTITION_SHA256 = "a1a21789c1d1890ff957b92f966720fd169b2b4004fe4acd17c693c37658d73a"
PARTNER_METADATA_SHA256 = "08fa39689be267769172c69a5424c2be4764d34318b75faa5481c8e2d1de16a8"
CHECKPOINT_BINDING_SHA256 = "accf813011342102fa2572aa27c2bde6796636cde3ac872756472c32b66f4ae1"
PREDECESSOR_ANALYSIS_SHA256 = "705980b12b6022f463450f1a6bc86b3e4c95ace7d9bfc24ac7e5392a3f11c317"
PANEL_BUILDER_SHA256 = "0d6b3cfc066f109978ff84a90cef3f9eeafe83043f3b5f393462d4f6e4b6f58b"


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--full-split", type=Path, required=True)
    parser.add_argument("--old-partition", type=Path, required=True)
    parser.add_argument("--partner-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint-binding", type=Path, required=True)
    parser.add_argument("--predecessor-analysis", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--measurement-sbatch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path, expected, label in (
        (args.full_split, FULL_SPLIT_SHA256, "full development split"),
        (args.old_partition, OLD_PARTITION_SHA256, "old development partition"),
        (args.partner_metadata, PARTNER_METADATA_SHA256, "partner metadata"),
        (args.checkpoint_binding, CHECKPOINT_BINDING_SHA256, "checkpoint binding"),
        (args.predecessor_analysis, PREDECESSOR_ANALYSIS_SHA256, "predecessor analysis"),
        (
            args.repository_root / "src/interaction_first/reciprocal_partner_panel.py",
            PANEL_BUILDER_SHA256,
            "reciprocal panel builder",
        ),
    ):
        require_sha(path, expected, label)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    sys.path.insert(0, str(args.repository_root / "src"))
    from interaction_first.reciprocal_partner_panel import (  # noqa: PLC0415
        build_reciprocal_partner_panel,
    )

    with args.full_split.open(newline="") as handle:
        reader = csv.DictReader(handle)
        full_fieldnames = list(reader.fieldnames or ())
        full_rows = list(reader)
    if len(full_rows) != 3663 or {row["role"] for row in full_rows} != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("full split differs")
    validation_by_id = {
        str(row["instance_id"]): row
        for row in full_rows
        if row["role"] == "validation"
    }
    if len(validation_by_id) != 549:
        raise ValueError("full validation count differs")
    old_partition = json.loads(args.old_partition.read_text())
    if (
        old_partition.get("task_labels_read") is not False
        or old_partition.get("test_features_or_labels_read") is not False
    ):
        raise ValueError("old partition authority differs")
    used_ids = {
        str(row["instance_id"])
        for rows in old_partition["roles"].values()
        for row in rows
    }
    if len(used_ids) != 128 or not used_ids.issubset(validation_by_id):
        raise ValueError("old development IDs differ")
    used_clusters = {
        str(validation_by_id[identifier]["antigen_cluster_id"])
        for identifier in used_ids
    }
    fresh_rows = [
        row
        for identifier, row in validation_by_id.items()
        if identifier not in used_ids
        and str(row["antigen_cluster_id"]) not in used_clusters
    ]
    fresh_ids = {str(row["instance_id"]) for row in fresh_rows}
    fresh_clusters = {str(row["antigen_cluster_id"]) for row in fresh_rows}
    if len(fresh_ids) != 156 or len(fresh_clusters) != 49:
        raise ValueError("fresh holdout cardinality differs")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        authority = temporary / "authority"
        authority.mkdir(parents=True)
        split_path = authority / "split-fresh156-measurement-only.csv"
        split_fieldnames = full_fieldnames + ["source_role", "fresh_holdout_included"]
        with split_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=split_fieldnames, lineterminator="\n")
            writer.writeheader()
            for original in full_rows:
                row = dict(original)
                row["source_role"] = row["role"]
                included = str(row["instance_id"]) in fresh_ids
                row["role"] = "validation" if included else "test"
                row["fresh_holdout_included"] = str(included).lower()
                writer.writerow(row)
        split_sha256 = sha256_file(split_path)

        with args.partner_metadata.open(newline="") as handle:
            reader = csv.DictReader(handle)
            metadata_fieldnames = list(reader.fieldnames or ())
            metadata_rows = list(reader)
        metadata_ids = {str(row["instance_id"]) for row in metadata_rows}
        if not fresh_ids.issubset(metadata_ids) or "role" not in metadata_fieldnames:
            raise ValueError("partner metadata coverage differs")
        bound_metadata = authority / "FRESH156_PARTNER_METADATA.csv"
        with bound_metadata.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=metadata_fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            for original in metadata_rows:
                row = dict(original)
                row["role"] = (
                    "validation" if str(row["instance_id"]) in fresh_ids else "excluded"
                )
                writer.writerow(row)
        metadata_sha256 = sha256_file(bound_metadata)
        panel_path = authority / "validation-fresh156-reciprocal-7.json"
        build_reciprocal_partner_panel(
            metadata_path=bound_metadata,
            metadata_sha256=metadata_sha256,
            output_path=panel_path,
            role="validation",
            panel_size_per_direction=7,
            protocol_seed=20260826,
        )
        panel_sha256 = sha256_file(panel_path)

        protocol = {
            "format_version": "sabdab2_edge_topk2_task_routed_holdout_protocol_v1",
            "status": "FROZEN_BEFORE_FRESH_HOLDOUT_MEASUREMENT",
            "candidate": {
                "name": "task_routed_baseline_contact_paratope_topk2_epitope",
                "contact_source": "baseline_current",
                "paratope_source": "baseline_current",
                "partner_retrieval_source": "baseline_current",
                "epitope_source": "edge_topk2_supervised",
                "routing_is_fixed_per_task": True,
                "mixture_weights_tuned": False,
                "additional_training": False,
                "inference_model_count": 2,
            },
            "seeds": [17, 29, 43],
            "holdout": {
                "complexes": len(fresh_ids),
                "antigen_clusters": len(fresh_clusters),
                "excluded_predecessor_complexes": len(used_ids),
                "excluded_predecessor_antigen_clusters": len(used_clusters),
                "antigen_cluster_overlap_with_predecessor": 0,
                "split_sha256": split_sha256,
                "partner_metadata_sha256": metadata_sha256,
                "partner_panel_sha256": panel_sha256,
                "labels_read_before_freeze": False,
            },
            "promotion_gate": {
                "median_epitope_delta_minimum": 0.005,
                "epitope_positive_seed_count_minimum": 2,
                "minimum_seed_epitope_delta": -0.01,
                "maximum_absolute_contact_delta": 1e-12,
                "maximum_absolute_paratope_delta": 1e-12,
            },
            "checkpoint_binding_sha256": CHECKPOINT_BINDING_SHA256,
            "predecessor_analysis_sha256": PREDECESSOR_ANALYSIS_SHA256,
            "execution_sources": {
                "reciprocal_panel_builder_sha256": PANEL_BUILDER_SHA256,
                "evaluator_sha256": sha256_file(args.evaluator),
                "analyzer_sha256": sha256_file(args.analyzer),
                "measurement_sbatch_sha256": sha256_file(args.measurement_sbatch),
            },
            "predecessor_confirmation_used_for_candidate_selection": False,
            "fresh_holdout_selection_performed": False,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        protocol_path = authority / "PROTOCOL_TASK_ROUTED_FRESH156_V1.json"
        protocol_path.write_bytes(canonical_bytes(protocol))
        protocol_sha256 = sha256_file(protocol_path)
        receipt = {
            "format_version": "sabdab2_edge_topk2_task_routed_holdout_receipt_v1",
            "status": protocol["status"],
            "protocol_sha256": protocol_sha256,
            "split_sha256": split_sha256,
            "partner_metadata_sha256": metadata_sha256,
            "partner_panel_sha256": panel_sha256,
            "fresh_holdout_complexes": len(fresh_ids),
            "fresh_holdout_antigen_clusters": len(fresh_clusters),
            "task_labels_read": False,
            "test_features_or_labels_read": False,
            "temporal_holdout_loaded": False,
        }
        (authority / "TASK_ROUTED_FRESH156_FREEZE_RECEIPT_V1.json").write_bytes(
            canonical_bytes(receipt)
        )
        temporary.replace(args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "protocol_sha256": protocol_sha256,
                "split_sha256": split_sha256,
                "partner_panel_sha256": panel_sha256,
                "fresh_holdout_complexes": len(fresh_ids),
                "fresh_holdout_antigen_clusters": len(fresh_clusters),
                "labels_read_before_freeze": False,
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
