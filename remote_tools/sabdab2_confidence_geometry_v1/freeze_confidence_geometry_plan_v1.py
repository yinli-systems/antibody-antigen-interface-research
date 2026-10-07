#!/usr/bin/env python3
"""Freeze the pLDDT-70 geometry sensitivity before reading cohort confidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_MANIFEST_SHA256 = (
    "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
)
CONTROL_ANALYSIS_SHA256 = (
    "1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c"
)
THRESHOLD = 70.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--training-launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    args = parser.parse_args()

    authority = args.campaign_root / "authority" / "confidence_geometry_plddt70_v1"
    if authority.exists() or authority.is_symlink():
        raise FileExistsError(authority)
    source_manifest = args.source_cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    control_analysis = (
        args.campaign_root / "PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json"
    )
    if sha256_file(source_manifest) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source predicted-geometry manifest differs")
    if sha256_file(control_analysis) != CONTROL_ANALYSIS_SHA256:
        raise ValueError("frozen no-distance control analysis differs")
    control = json.loads(control_analysis.read_text(encoding="utf-8"))
    if (
        control.get("status")
        != "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT"
        or control.get("old_sabdab2_test_loaded") is not False
        or control.get("temporal_holdout_loaded") is not False
        or control.get("distance_loss_weight") != 0.0
    ):
        raise ValueError("control analysis content differs")

    authority.mkdir(parents=True)
    plan = {
        "format_version": "sabdab2_confidence_geometry_plddt70_plan_v1",
        "status": "FROZEN_BEFORE_CONFIDENCE_EXTRACTION_AND_TRAINING",
        "source_structure_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "control_analysis_sha256": CONTROL_ANALYSIS_SHA256,
        "threshold": THRESHOLD,
        "threshold_rule": "mean_N_CA_C_plddt_greater_than_or_equal",
        "threshold_rationale": (
            "AlphaFold confidence convention: pLDDT >=70 is generally confident; "
            "the threshold is frozen before cohort-wide confidence extraction."
        ),
        "single_factor_change": (
            "suppress predicted intra-chain geometry at residues with mean N/CA/C "
            "pLDDT below 70 while retaining sequence embeddings"
        ),
        "arms": ["baseline", "edge_topk2_supervised"],
        "seeds": [17, 29, 43],
        "control_reused_not_retrained": True,
        "training_data_split_labels_optimization_selection_unchanged": True,
        "distance_loss_weight": 0.0,
        "distance_label_cache_supplied": False,
        "promotion_gate": {
            "median_paratope_delta_gt": 0.0,
            "minimum_seed_paratope_delta_gte": -0.005,
            "median_contact_delta_gte": -0.002,
            "median_epitope_delta_gte": -0.005,
        },
        "execution_sha256": {
            "freezer": sha256_file(Path(__file__)),
            "materializer": sha256_file(args.materializer),
            "finalizer": sha256_file(args.finalizer),
            "training_launcher": sha256_file(args.training_launcher),
            "analyzer": sha256_file(args.analyzer),
        },
        "evidence_boundary": {
            "development_only": True,
            "confidence_extraction_reads_task_labels": False,
            "experimental_bound_geometry_used_as_model_input": False,
            "experimental_bound_geometry_used_for_auxiliary_training_supervision": False,
            "experimental_bound_structure_used_to_define_task_contact_labels": True,
            "selection_or_tuning_from_this_study": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
        },
        "references": {
            "alphafold_database_confidence_interpretation": (
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC8728224/"
            ),
            "deepmind_plddt_and_pae_explanation": (
                "https://deepmind.google/blog/enabling-high-accuracy-protein-structure-"
                "prediction-at-the-proteome-scale/"
            ),
        },
    }
    plan_path = authority / "PLAN_CONFIDENCE_GEOMETRY_PLDDT70_V1.json"
    plan_path.write_bytes(canonical_bytes(plan))
    receipt = {
        "format_version": "sabdab2_confidence_geometry_plddt70_plan_receipt_v1",
        "status": plan["status"],
        "plan_sha256": sha256_file(plan_path),
        "threshold": THRESHOLD,
        "frozen_before_cohort_confidence_extraction": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_path = authority / "PLAN_CONFIDENCE_GEOMETRY_PLDDT70_V1_RECEIPT.json"
    receipt_path.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                **receipt,
                "receipt_sha256": sha256_file(receipt_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
