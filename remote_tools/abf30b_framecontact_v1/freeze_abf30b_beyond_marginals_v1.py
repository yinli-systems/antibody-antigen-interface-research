#!/usr/bin/env python3
"""Freeze a read-only ABF-30B development audit of contact-map components."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SEEDS = (17, 29, 43)
FINAL_ANALYSIS_SHA256 = "c83a025c0f895ceecf61d0a079095dd182f3a8610cc419b1482604176af3b68d"
FINAL_ANALYSIS_RECEIPT_SHA256 = "2ce0e8586703dd1fd5166f35cba9b3a3eb416bc3bc5bf27c485f94d483205126"
EXPECTED_STRUCTURE_SHA256 = "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
EXPECTED_CONTACT_SHA256 = "d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88"
EXPECTED_SPLIT_SHA256 = "ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700"
EXPECTED_EMBEDDING_SHA256 = "5cd4bec4e7e168c52fc0e479cb71e521b5e62adab479c8097a6d8f418faea498"
EXPECTED_SOURCE_MANIFEST_SHA256 = "900b3d7646bebcbfd0f22b93108b791038e1a804aaa073f200fda2febeb06c88"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    analysis_path = args.campaign_root / "ABF30B_FRAMECONTACT_DEVELOPMENT_ANALYSIS_V1.json"
    analysis_receipt_path = (
        args.campaign_root / "ABF30B_FRAMECONTACT_DEVELOPMENT_ANALYSIS_V1_RECEIPT.json"
    )
    require_sha(analysis_path, FINAL_ANALYSIS_SHA256)
    require_sha(analysis_receipt_path, FINAL_ANALYSIS_RECEIPT_SHA256)
    analysis = json.loads(analysis_path.read_text())
    analysis_receipt = json.loads(analysis_receipt_path.read_text())
    if (
        analysis.get("status") != "PASS_ABF30B_ANTIBODY_REPRESENTATION_DEVELOPMENT"
        or analysis.get("development_only") is not True
        or analysis.get("independent_test") is not False
        or analysis.get("old_sabdab2_test_loaded") is not False
        or analysis.get("temporal_holdout_loaded") is not False
        or analysis.get("distance_loss_weight") != 0.0
        or analysis.get("distance_label_cache_supplied") is not False
        or analysis_receipt.get("analysis_sha256") != FINAL_ANALYSIS_SHA256
    ):
        raise ValueError("final ABF-30B development authority differs")

    run_bindings: dict[str, Any] = {}
    split_path: Path | None = None
    structure_root: Path | None = None
    contact_root: Path | None = None
    embedding_root: Path | None = None
    source_manifest_path: Path | None = None
    for seed in SEEDS:
        binding = analysis["run_bindings"]["baseline"][str(seed)]
        run_root = Path(binding["run_root"])
        result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
        receipt_path = run_root / "RUN_RECEIPT.json"
        for path, expected in (
            (result_path, binding["result_sha256"]),
            (checkpoint_path, binding["checkpoint_sha256"]),
            (receipt_path, binding["receipt_sha256"]),
        ):
            require_sha(path, expected)
        result = json.loads(result_path.read_text())
        configuration = result["configuration"]
        authority = result["authority"]
        if (
            result.get("experiment") != "epiformer_style_same_protocol"
            or configuration.get("seed") != seed
            or configuration.get("distance_loss_weight") != 0.0
            or configuration.get("specificity_loss_weight") != 0.0
            or authority.get("test_loaded") is not False
            or authority.get("external_test_loaded") is not False
            or authority.get("distance_label_manifest_sha256") is not None
            or authority.get("structure_manifest_sha256") != EXPECTED_STRUCTURE_SHA256
            or authority.get("contact_manifest_sha256") != EXPECTED_CONTACT_SHA256
            or authority.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
            or authority.get("embedding_manifest_sha256") != EXPECTED_EMBEDDING_SHA256
        ):
            raise ValueError(f"baseline run authority differs for seed {seed}")
        local_split = Path(configuration["split_manifest"])
        local_source_manifest = Path(configuration["source_manifest"])
        if split_path is None:
            split_path = local_split
            source_manifest_path = local_source_manifest
            structure_root = local_split.parent
            contact_root = Path(
                "/ssd/scxi253/model_deployment/results/development/"
                "sabdab2_sequence_predicted_geometry_development_v1/data/"
                "contact_cache_predicted_geometry_authority_v1"
            )
            embedding_root = (
                args.campaign_root
                / "embeddings/sabdab2_abf30b_antibody_esm2_antigen_v1"
            )
        elif local_split != split_path or local_source_manifest != source_manifest_path:
            raise ValueError("run data/source paths differ across seeds")
        run_bindings[str(seed)] = {
            "seed": seed,
            "run_root": str(run_root),
            "result_path": str(result_path),
            "result_sha256": binding["result_sha256"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": binding["receipt_sha256"],
            "selected_epoch": binding["selected_epoch"],
        }

    assert split_path is not None
    assert structure_root is not None
    assert contact_root is not None
    assert embedding_root is not None
    assert source_manifest_path is not None
    require_sha(split_path, EXPECTED_SPLIT_SHA256)
    require_sha(structure_root / "STRUCTURE_CACHE_MANIFEST.json", EXPECTED_STRUCTURE_SHA256)
    require_sha(contact_root / "CONTACT_CACHE_MANIFEST.json", EXPECTED_CONTACT_SHA256)
    require_sha(
        embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json",
        EXPECTED_EMBEDDING_SHA256,
    )
    require_sha(source_manifest_path, EXPECTED_SOURCE_MANIFEST_SHA256)
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest_sha = EXPECTED_SOURCE_MANIFEST_SHA256
    source_root = Path(source_manifest["source_root"])

    with split_path.open(newline="") as handle:
        validation = [row for row in csv.DictReader(handle) if row["role"] == "validation"]
    if (
        len(validation) != 549
        or len({row["instance_id"] for row in validation}) != 549
        or len({row["antigen_cluster_id"] for row in validation}) != 108
        or len({row["pdb_assembly_id"] for row in validation}) != 305
    ):
        raise ValueError("validation cohort cardinality differs")

    evaluator_sha = sha256_file(args.evaluator)
    analyzer_sha = sha256_file(args.analyzer)
    protocol = {
        "format_version": "abf30b_beyond_marginals_protocol_v1",
        "status": "FROZEN_BEFORE_READ_ONLY_DEVELOPMENT_COMPONENT_AUDIT",
        "frozen_checkpoint_arm": "baseline",
        "seeds": list(SEEDS),
        "run_bindings": run_bindings,
        "data": {
            "role": "validation",
            "instance_count": 549,
            "antigen_cluster_count": 108,
            "pdb_assembly_count": 305,
            "structure_root": str(structure_root),
            "structure_manifest_sha256": EXPECTED_STRUCTURE_SHA256,
            "contact_root": str(contact_root),
            "contact_manifest_sha256": EXPECTED_CONTACT_SHA256,
            "embedding_root": str(embedding_root),
            "embedding_manifest_sha256": EXPECTED_EMBEDDING_SHA256,
            "split_path": str(split_path),
            "split_sha256": EXPECTED_SPLIT_SHA256,
        },
        "source": {
            "source_root": str(source_root),
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": source_manifest_sha,
            "source_tree_sha256": source_manifest["tree_sha256"],
        },
        "scores": {
            "full_contact": "final contact logits",
            "unary_only": "full_contact minus raw_interaction; exact two unary terms plus contact bias",
            "residue_head_outer_product": "sigmoid(paratope_logit) outer-product sigmoid(epitope_logit) from the same checkpoint",
            "raw_interaction": "decoder contact_interaction_logits before unary terms",
            "gauge_fixed_residual": "two-way centered final contact logits on the complete valid rectangular grid",
        },
        "metrics": {
            "primary": "per-complex average precision",
            "pooled": "average precision over every valid pair",
            "ranking": [
                "reciprocal_rank_first_positive",
                "hits_at_1",
                "recall_at_P",
                "recall_at_ceil_P_over_2",
                "recall_at_ceil_P_over_5",
            ],
            "P": "number of positive contact pairs in that complex",
            "zero_positive_complexes": "excluded from per-complex/ranking estimands but retained in pooled AP",
        },
        "estimand": {
            "operation_order": [
                "compute each score and metric per complex and seed",
                "average the three fixed seeds within each complex",
                "compute score differences full_contact minus comparator",
                "average complex differences within each antigen cluster",
                "average antigen-cluster means with equal cluster weight",
            ],
            "primary_comparator": "residue_head_outer_product",
            "comparators": [
                "unary_only",
                "residue_head_outer_product",
                "raw_interaction",
                "gauge_fixed_residual",
            ],
            "primary_grouping": "antigen_cluster_id",
            "secondary_grouping": "pdb_assembly_id",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260831,
            "confidence_interval": [0.025, 0.975],
            "conditional_on_fixed_trained_seeds": True,
        },
        "tools": {
            "evaluator_sha256": evaluator_sha,
            "analyzer_sha256": analyzer_sha,
        },
        "upstream": {
            "final_development_analysis_sha256": FINAL_ANALYSIS_SHA256,
            "final_development_analysis_receipt_sha256": FINAL_ANALYSIS_RECEIPT_SHA256,
        },
        "evidence_boundary": {
            "development_only": True,
            "independent_test": False,
            "model_training": False,
            "checkpoint_selection": False,
            "calibration_or_threshold_fitting": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
            "sable_external_labels_loaded": False,
            "raw_pair_predictions_persisted": False,
            "experimental_bound_coordinates_used_as_model_input": False,
            "experimental_bound_structure_used_to_define_development_contact_labels": True,
        },
    }
    args.output_root.mkdir(parents=True)
    protocol_path = args.output_root / "ABF30B_BEYOND_MARGINALS_PROTOCOL_V1.json"
    protocol_sha = publish(protocol_path, protocol)
    receipt = {
        "format_version": "abf30b_beyond_marginals_protocol_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": protocol_sha,
        "evaluator_sha256": evaluator_sha,
        "analyzer_sha256": analyzer_sha,
        "development_only": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
        "sable_external_labels_loaded": False,
    }
    receipt_sha = publish(
        args.output_root / "ABF30B_BEYOND_MARGINALS_PROTOCOL_V1_RECEIPT.json",
        receipt,
    )
    print(json.dumps({"protocol_sha256": protocol_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
