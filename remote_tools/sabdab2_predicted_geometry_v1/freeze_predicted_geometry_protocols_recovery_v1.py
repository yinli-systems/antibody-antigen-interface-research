#!/usr/bin/env python3
"""Freeze baseline/expert protocols for the predicted-geometry sensitivity study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--baseline-template", type=Path, required=True)
    parser.add_argument("--expert-template", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest-sha256", required=True)
    parser.add_argument("--expert-source-manifest", type=Path, required=True)
    parser.add_argument("--expert-source-manifest-sha256", required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--training-launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--recovery-manifest-sha256", required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--recovery-receipt-sha256", required=True)
    return parser.parse_args()


def protocol_receipt(
    path: Path,
    protocol_sha256: str,
    *,
    arm: str,
    structure_sha256: str,
    split_sha256: str,
) -> str:
    return write_json(
        path,
        {
            "format_version": "sabdab2_predicted_geometry_protocol_receipt_v1",
            "arm": arm,
            "protocol_file_sha256": protocol_sha256,
            "structure_manifest_sha256": structure_sha256,
            "development_split_view_sha256": split_sha256,
            "frozen_before_training": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    )


def common_claim_boundary() -> dict[str, Any]:
    return {
        "development_only": True,
        "geometry_sensitivity_analysis": True,
        "full_validation_is_checkpoint_selection_data": True,
        "full_validation_is_independent_confirmation": False,
        "model_architecture_loss_and_optimization_unchanged": True,
        "test_loaded": False,
        "old_sabdab2_test_materialized": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning_from_this_study": False,
        "claim": (
            "Sensitivity of the frozen FrameContact training recipe to independent "
            "sequence-predicted chain geometry on train/validation only."
        ),
    }


def update_data(
    protocol: dict[str, Any], structure_sha256: str, split_sha256: str
) -> None:
    data = protocol["data"]
    data["structure_manifest_sha256"] = structure_sha256
    data["full_split_sha256"] = split_sha256
    data["full_development"] = {
        "split_sha256": split_sha256,
        "train_count": 2565,
        "validation_count": 549,
        "test_count_not_materialized": 549,
        "test_features_or_labels_loaded": False,
    }
    data["geometry_input"] = {
        "source": "independent_sequence_predicted_af3",
        "antibody_and_antigen_predicted_independently": True,
        "experimental_bound_complex_coordinates_used": False,
        "cross_partner_coordinates_or_distances_used": False,
        "templates": "disabled",
        "historical_paired_and_antigen_unpaired_msa": "homologs",
        "new_sdh_unpaired_msa": "query_only",
    }


def main() -> None:
    args = parse_args()
    authority = args.campaign_root / "authority"
    if authority.exists() or authority.is_symlink():
        raise FileExistsError(authority)
    manifest_path = args.structure_cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    receipt_path = args.structure_cache_root / "STRUCTURE_CACHE_RECEIPT.json"
    split_path = args.structure_cache_root / "DEVELOPMENT_SPLIT_VIEW.csv"
    structure_sha256 = sha256_file(manifest_path)
    split_sha256 = sha256_file(split_path)
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        cache_receipt.get("status")
        != "PASS_DEVELOPMENT_ONLY_PREDICTED_STRUCTURE_CACHE"
        or cache_receipt.get("manifest_sha256") != structure_sha256
        or cache_receipt.get("development_split_view_sha256") != split_sha256
        or cache_receipt.get("record_count") != 3114
        or cache_receipt.get("role_counts") != {"train": 2565, "validation": 549}
        or cache_receipt.get("all_artifact_sha256_verified") is not True
        or cache_receipt.get("test_structures_resolved") is not False
        or cache_receipt.get("temporal_holdout_access") is not False
    ):
        raise ValueError("predicted structure cache receipt differs")
    manifest_authority = cache_manifest.get("authority", {})
    if (
        cache_manifest.get("format_version")
        != "sabdab2_intra_chain_structure_cohort_cache_v2"
        or cache_manifest.get("counts", {}).get("records") != 3114
        or cache_manifest.get("counts", {}).get("roles")
        != {"train": 2565, "validation": 549}
        or len(cache_manifest.get("records", [])) != 3114
        or {record.get("role") for record in cache_manifest.get("records", [])}
        != {"train", "validation"}
        or manifest_authority.get("geometry_source")
        != "independent_sequence_predicted_af3"
        or manifest_authority.get("experimental_bound_complex_coordinates_read")
        is not False
        or manifest_authority.get("cross_partner_coordinates_or_distances_used")
        is not False
        or manifest_authority.get("task_labels_read") is not False
        or manifest_authority.get("test_structures_resolved") is not False
        or manifest_authority.get("temporal_holdout_access") is not False
    ):
        raise ValueError("predicted structure cache manifest differs")
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    split_role_counts = {
        role: sum(row.get("role") == role for row in split_rows)
        for role in ("train", "validation")
    }
    if (
        len(split_rows) != 3114
        or split_role_counts != {"train": 2565, "validation": 549}
        or any(row.get("role") not in {"train", "validation"} for row in split_rows)
        or len({row.get("instance_id") for row in split_rows}) != 3114
        or {row.get("instance_id") for row in split_rows}
        != {
            record.get("instance_id")
            for record in cache_manifest.get("records", [])
        }
    ):
        raise ValueError("development split view differs")
    if (
        sha256_file(args.baseline_source_manifest)
        != args.baseline_source_manifest_sha256
        or sha256_file(args.expert_source_manifest)
        != args.expert_source_manifest_sha256
    ):
        raise ValueError("training source manifest differs")
    if (
        sha256_file(args.recovery_manifest) != args.recovery_manifest_sha256
        or sha256_file(args.recovery_receipt) != args.recovery_receipt_sha256
    ):
        raise ValueError("canonical recovery authority hash differs")
    recovery_manifest = json.loads(args.recovery_manifest.read_text(encoding="utf-8"))
    recovery_receipt = json.loads(args.recovery_receipt.read_text(encoding="utf-8"))
    if (
        recovery_manifest.get("status") != "PASS_CANONICAL_SDH_RECOVERY_VIEW"
        or recovery_manifest.get("record_count") != 332
        or recovery_manifest.get("exact_canonical_name_count") != 331
        or recovery_manifest.get("timestamp_suffixed_recovery_count") != 1
        or recovery_manifest.get("test_access") is not False
        or recovery_manifest.get("temporal_holdout_access") is not False
        or recovery_manifest.get("experimental_bound_complex_coordinates_read")
        is not False
        or recovery_receipt.get("status")
        != "PASS_CANONICAL_SDH_RECOVERY_VIEW"
        or recovery_receipt.get("manifest_sha256")
        != args.recovery_manifest_sha256
        or recovery_receipt.get("source_predictions_modified") is not False
        or recovery_receipt.get("test_access") is not False
        or recovery_receipt.get("temporal_holdout_access") is not False
    ):
        raise ValueError("canonical recovery authority differs")

    authority.mkdir(parents=True)
    baseline_manifest_copy = authority / "SOURCE_MANIFEST_BASELINE_V3.json"
    expert_manifest_copy = authority / "SOURCE_MANIFEST_EDGE_TOPK2_V1.json"
    recovery_manifest_copy = authority / "SDH_CANONICAL_RECOVERY_VIEW_MANIFEST.json"
    recovery_receipt_copy = authority / "SDH_CANONICAL_RECOVERY_VIEW_RECEIPT.json"
    shutil.copyfile(args.baseline_source_manifest, baseline_manifest_copy)
    shutil.copyfile(args.expert_source_manifest, expert_manifest_copy)
    shutil.copyfile(args.recovery_manifest, recovery_manifest_copy)
    shutil.copyfile(args.recovery_receipt, recovery_receipt_copy)

    baseline = json.loads(args.baseline_template.read_text(encoding="utf-8"))
    baseline["schema_version"] = "sabdab2_predicted_geometry_baseline_development_v1"
    baseline["protocol_status"] = "frozen_before_full_development_training"
    baseline["arm"] = "baseline"
    baseline["claim_boundary"] = common_claim_boundary()
    update_data(baseline, structure_sha256, split_sha256)
    baseline.pop("full_development_gate", None)
    baseline["source_authority"] = {
        "source_manifest_sha256": args.baseline_source_manifest_sha256,
        "protocol_freezer_sha256": sha256_file(Path(__file__)),
        "materializer_sha256": sha256_file(args.materializer),
        "training_launcher_sha256": sha256_file(args.training_launcher),
        "analyzer_sha256": sha256_file(args.analyzer),
        "sdh_canonical_recovery_manifest_sha256": args.recovery_manifest_sha256,
        "sdh_canonical_recovery_receipt_sha256": args.recovery_receipt_sha256,
    }

    expert = json.loads(args.expert_template.read_text(encoding="utf-8"))
    expert["schema_version"] = "sabdab2_predicted_geometry_topk2_development_v1"
    expert["protocol_status"] = "frozen_before_full_development_training"
    expert["arm"] = "edge_topk2_supervised"
    expert["claim_boundary"] = common_claim_boundary()
    update_data(expert, structure_sha256, split_sha256)
    expert.pop("baseline_binding", None)
    expert.pop("full_development_gate", None)
    expert["task_routing"] = {
        "name": "seed_matched_baseline_contact_paratope_topk2_epitope",
        "contact_source": "predicted_geometry_baseline_same_seed",
        "paratope_source": "predicted_geometry_baseline_same_seed",
        "epitope_source": "predicted_geometry_topk2_same_seed",
        "routing_is_fixed_per_task": True,
        "mixture_weights_tuned": False,
        "inference_model_count": 2,
    }
    expert["execution_sources"] = {
        "source_manifest_sha256": args.expert_source_manifest_sha256,
        "protocol_freezer_sha256": sha256_file(Path(__file__)),
        "materializer_sha256": sha256_file(args.materializer),
        "training_launcher_sha256": sha256_file(args.training_launcher),
        "analyzer_sha256": sha256_file(args.analyzer),
        "sdh_canonical_recovery_manifest_sha256": args.recovery_manifest_sha256,
        "sdh_canonical_recovery_receipt_sha256": args.recovery_receipt_sha256,
    }

    baseline_path = authority / "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_V1.json"
    expert_path = authority / "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_V1.json"
    baseline_sha256 = write_json(baseline_path, baseline)
    expert_sha256 = write_json(expert_path, expert)
    baseline_receipt_path = authority / "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_V1_RECEIPT.json"
    expert_receipt_path = authority / "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_V1_RECEIPT.json"
    baseline_receipt_sha256 = protocol_receipt(
        baseline_receipt_path,
        baseline_sha256,
        arm="baseline",
        structure_sha256=structure_sha256,
        split_sha256=split_sha256,
    )
    expert_receipt_sha256 = protocol_receipt(
        expert_receipt_path,
        expert_sha256,
        arm="edge_topk2_supervised",
        structure_sha256=structure_sha256,
        split_sha256=split_sha256,
    )
    campaign = {
        "format_version": "sabdab2_predicted_geometry_development_campaign_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "campaign_root": str(args.campaign_root),
        "structure_cache_root": str(args.structure_cache_root),
        "structure_manifest_sha256": structure_sha256,
        "development_split_view_sha256": split_sha256,
        "arms": {
            "baseline": {
                "protocol_sha256": baseline_sha256,
                "receipt_sha256": baseline_receipt_sha256,
                "source_manifest_sha256": args.baseline_source_manifest_sha256,
            },
            "edge_topk2_supervised": {
                "protocol_sha256": expert_sha256,
                "receipt_sha256": expert_receipt_sha256,
                "source_manifest_sha256": args.expert_source_manifest_sha256,
            },
        },
        "seeds": [17, 29, 43],
        "test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning": False,
        "sdh_canonical_recovery_manifest_sha256": args.recovery_manifest_sha256,
        "sdh_canonical_recovery_receipt_sha256": args.recovery_receipt_sha256,
    }
    campaign_path = authority / "CAMPAIGN_PREDICTED_GEOMETRY_DEVELOPMENT_V1.json"
    campaign_sha256 = write_json(campaign_path, campaign)
    lines = [
        f"{structure_sha256}  {manifest_path}",
        f"{split_sha256}  {split_path}",
        f"{baseline_sha256}  {baseline_path}",
        f"{baseline_receipt_sha256}  {baseline_receipt_path}",
        f"{expert_sha256}  {expert_path}",
        f"{expert_receipt_sha256}  {expert_receipt_path}",
        f"{args.baseline_source_manifest_sha256}  {baseline_manifest_copy}",
        f"{args.expert_source_manifest_sha256}  {expert_manifest_copy}",
        f"{args.recovery_manifest_sha256}  {recovery_manifest_copy}",
        f"{args.recovery_receipt_sha256}  {recovery_receipt_copy}",
        f"{campaign_sha256}  {campaign_path}",
    ]
    (authority / "AUTHORITY_SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_TRAINING",
                "structure_manifest_sha256": structure_sha256,
                "split_sha256": split_sha256,
                "baseline_protocol_sha256": baseline_sha256,
                "expert_protocol_sha256": expert_sha256,
                "campaign_sha256": campaign_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
