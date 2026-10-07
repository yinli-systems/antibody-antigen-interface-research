#!/usr/bin/env python3
"""Bind a label-free pLDDT-masked cache into frozen training protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


SOURCE_MANIFEST_SHA256 = (
    "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
)
CONTROL_ANALYSIS_SHA256 = (
    "1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c"
)
SPLIT_SHA256 = "ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700"
SOURCE_CONTACT_MANIFEST_SHA256 = (
    "d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88"
)
SOURCE_CONTACT_RECEIPT_SHA256 = (
    "426975feebdb5294b0fdbc45d31422fb3adbadc4d97b032958a38d19b050f1d9"
)
SOURCE_SHA = {
    "baseline": "a320a5f60794ea386d355e8ab5ec895130d3b714596bc817e303eee589dbc7e8",
    "edge_topk2_supervised": (
        "885449525b7f41b3b00511245ca6c0b0859336db959becc7680ad9cb6e872332"
    ),
}
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
    parser.add_argument("--masked-cache-root", type=Path, required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    args = parser.parse_args()
    authority = args.campaign_root / "authority" / "confidence_geometry_plddt70_v1"
    plan_path = authority / "PLAN_CONFIDENCE_GEOMETRY_PLDDT70_V1.json"
    plan_receipt_path = authority / "PLAN_CONFIDENCE_GEOMETRY_PLDDT70_V1_RECEIPT.json"
    for path in (plan_path, plan_receipt_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing frozen pre-materialization authority: {path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_receipt = json.loads(plan_receipt_path.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "FROZEN_BEFORE_CONFIDENCE_EXTRACTION_AND_TRAINING"
        or plan.get("threshold") != THRESHOLD
        or plan_receipt.get("plan_sha256") != sha256_file(plan_path)
        or plan_receipt.get("frozen_before_cohort_confidence_extraction") is not True
    ):
        raise ValueError("pre-materialization confidence plan differs")

    masked_manifest_path = args.masked_cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    masked_receipt_path = args.masked_cache_root / "STRUCTURE_CACHE_RECEIPT.json"
    manifest = json.loads(masked_manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(masked_receipt_path.read_text(encoding="utf-8"))
    masked_sha = sha256_file(masked_manifest_path)
    masked_receipt_sha = sha256_file(masked_receipt_path)
    if (
        manifest.get("counts", {}).get("roles")
        != {"train": 2565, "validation": 549}
        or len(manifest.get("records", [])) != 3114
        or manifest.get("authority", {}).get("source_structure_manifest_sha256")
        != SOURCE_MANIFEST_SHA256
        or manifest.get("authority", {}).get("confidence_masking", {}).get("threshold")
        != THRESHOLD
        or manifest.get("authority", {}).get("task_labels_read") is not False
        or manifest.get("authority", {}).get("test_structures_resolved") is not False
        or receipt.get("manifest_sha256") != masked_sha
        or receipt.get("task_labels_read") is not False
        or receipt.get("test_or_temporal_loaded") is not False
    ):
        raise ValueError("confidence-masked cache authority differs")

    contact_manifest_path = args.contact_cache_root / "CONTACT_CACHE_MANIFEST.json"
    contact_receipt_path = args.contact_cache_root / "CONTACT_CACHE_RECEIPT.json"
    contact_manifest = json.loads(contact_manifest_path.read_text(encoding="utf-8"))
    contact_receipt = json.loads(contact_receipt_path.read_text(encoding="utf-8"))
    contact_sha = sha256_file(contact_manifest_path)
    contact_receipt_sha = sha256_file(contact_receipt_path)
    contact_authority = contact_manifest.get("authority", {})
    if (
        contact_manifest.get("counts", {}).get("by_role")
        != {"train": 2565, "validation": 549}
        or contact_manifest.get("counts", {}).get("records") != 3114
        or len(contact_manifest.get("records", [])) != 3114
        or contact_authority.get("source_contact_manifest_sha256")
        != SOURCE_CONTACT_MANIFEST_SHA256
        or contact_authority.get("source_contact_receipt_sha256")
        != SOURCE_CONTACT_RECEIPT_SHA256
        or contact_authority.get("structure_manifest_sha256") != masked_sha
        or contact_authority.get("pair_label_artifacts_hard_linked") is not True
        or contact_authority.get("pair_label_artifacts_rehashed") is not True
        or contact_authority.get("pair_label_artifacts_reused_byte_identically")
        is not True
        or contact_authority.get("residue_axis_lengths_verified") is not True
        or contact_authority.get("selection_or_tuning") is not False
        or contact_authority.get("test_pair_artifacts_read") is not False
        or contact_authority.get("temporal_holdout_access") is not False
        or contact_receipt.get("manifest_sha256") != contact_sha
        or contact_receipt.get("predicted_structure_manifest_sha256") != masked_sha
        or contact_receipt.get("record_count") != 3114
        or contact_receipt.get("role_counts")
        != {"train": 2565, "validation": 549}
        or contact_receipt.get("all_pair_artifact_sha256_verified") is not True
        or contact_receipt.get("all_pair_artifacts_hard_linked") is not True
        or contact_receipt.get("all_residue_axis_lengths_verified") is not True
        or contact_receipt.get("pair_label_artifacts_modified") is not False
        or contact_receipt.get("pair_labels_regenerated") is not False
        or contact_receipt.get("test_pair_artifacts_read") is not False
        or contact_receipt.get("temporal_holdout_access") is not False
    ):
        raise ValueError("confidence contact-cache authority differs")

    control_authority = args.campaign_root / "authority" / "no_privileged_distance_v1"
    protocol_templates = {
        "baseline": control_authority
        / "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_NO_PRIVILEGED_DISTANCE_V1.json",
        "edge_topk2_supervised": control_authority
        / "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_NO_PRIVILEGED_DISTANCE_V1.json",
    }
    source_templates = {
        "baseline": control_authority / "SOURCE_MANIFEST_BASELINE_NO_DISTANCE_V1.json",
        "edge_topk2_supervised": control_authority
        / "SOURCE_MANIFEST_TOPK2_NO_DISTANCE_V1.json",
    }
    arms: dict[str, Any] = {}
    checksums: list[tuple[str, Path]] = []
    for arm in ("baseline", "edge_topk2_supervised"):
        if sha256_file(source_templates[arm]) != SOURCE_SHA[arm]:
            raise ValueError(f"source manifest differs for {arm}")
        source_copy = authority / f"SOURCE_MANIFEST_{arm.upper()}_PLDDT70_V1.json"
        shutil.copyfile(source_templates[arm], source_copy)
        protocol = json.loads(protocol_templates[arm].read_text(encoding="utf-8"))
        if protocol["data"].get("structure_manifest_sha256") != SOURCE_MANIFEST_SHA256:
            raise ValueError(f"control protocol structure authority differs for {arm}")
        protocol["schema_version"] = f"sabdab2_confidence_geometry_plddt70_v1_{arm}"
        protocol["protocol_status"] = "frozen_before_plddt70_training_v1"
        protocol["data"]["structure_manifest_sha256"] = masked_sha
        protocol["data"]["contact_manifest_sha256"] = contact_sha
        protocol["data"]["contact_authority_adapter"].update(
            {
                "type": "development_only_pair_label_rebind_without_artifact_change_v2",
                "receipt_sha256": contact_receipt_sha,
                "source_contact_manifest_sha256": SOURCE_CONTACT_MANIFEST_SHA256,
                "pair_label_artifacts_modified": False,
                "pair_labels_regenerated": False,
                "residue_axis_lengths_verified": True,
            }
        )
        protocol["data"]["geometry_input"].update(
            {
                "source": "independent_sequence_predicted_af3_plddt70_masked",
                "confidence_source": "AF3 model CIF N/CA/C pLDDT",
                "confidence_threshold": THRESHOLD,
                "low_confidence_geometry_suppressed": True,
                "sequence_embeddings_retained_for_masked_residues": True,
            }
        )
        protocol["claim_boundary"].update(
            {
                "claim": (
                    "Development-only single-factor sensitivity to suppressing "
                    "low-confidence sequence-predicted intra-chain geometry."
                ),
                "selection_or_tuning_from_this_study": False,
                "old_sabdab2_test_loaded": False,
                "temporal_holdout_loaded": False,
            }
        )
        protocol["confidence_geometry_plddt70_v1"] = {
            "plan_sha256": sha256_file(plan_path),
            "source_structure_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "masked_structure_manifest_sha256": masked_sha,
            "masked_structure_receipt_sha256": masked_receipt_sha,
            "control_analysis_sha256": CONTROL_ANALYSIS_SHA256,
            "threshold": THRESHOLD,
            "single_factor_change": plan["single_factor_change"],
            "training_data_split_labels_optimization_selection_unchanged": True,
            "distance_loss_weight": 0.0,
            "distance_label_cache_supplied": False,
        }
        protocol_path = authority / f"PROTOCOL_CONFIDENCE_GEOMETRY_{arm.upper()}_V1.json"
        protocol_path.write_bytes(canonical_bytes(protocol))
        protocol_sha = sha256_file(protocol_path)
        protocol_receipt = {
            "format_version": "sabdab2_confidence_geometry_protocol_receipt_v1",
            "arm": arm,
            "protocol_file_sha256": protocol_sha,
            "source_manifest_sha256": SOURCE_SHA[arm],
            "structure_manifest_sha256": masked_sha,
            "contact_manifest_sha256": contact_sha,
            "contact_receipt_sha256": contact_receipt_sha,
            "development_split_sha256": SPLIT_SHA256,
            "distance_loss_weight": 0.0,
            "distance_label_authority_supplied": False,
            "frozen_before_training": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        protocol_receipt_path = authority / (
            f"PROTOCOL_CONFIDENCE_GEOMETRY_{arm.upper()}_V1_RECEIPT.json"
        )
        protocol_receipt_path.write_bytes(canonical_bytes(protocol_receipt))
        arms[arm] = {
            "protocol_sha256": protocol_sha,
            "receipt_sha256": sha256_file(protocol_receipt_path),
            "source_manifest_sha256": SOURCE_SHA[arm],
        }
        checksums.extend(
            (
                (sha256_file(source_copy), source_copy),
                (protocol_sha, protocol_path),
                (sha256_file(protocol_receipt_path), protocol_receipt_path),
            )
        )

    campaign = {
        "format_version": "sabdab2_confidence_geometry_plddt70_campaign_v1",
        "status": "FROZEN_BEFORE_PLDDT70_DEVELOPMENT_TRAINING",
        "plan_sha256": sha256_file(plan_path),
        "source_structure_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "model_input_structure_manifest_sha256": masked_sha,
        "model_input_structure_receipt_sha256": masked_receipt_sha,
        "control_analysis_sha256": CONTROL_ANALYSIS_SHA256,
        "contact_manifest_sha256": contact_sha,
        "contact_receipt_sha256": contact_receipt_sha,
        "source_contact_manifest_sha256": SOURCE_CONTACT_MANIFEST_SHA256,
        "contact_authority_recovery": {
            "failed_array_job_id": "1495054",
            "failure_type": "contact_structure_authority_mismatch",
            "failed_before_training_updates": True,
            "pair_label_artifacts_modified": False,
            "pair_labels_regenerated": False,
            "model_loss_split_optimizer_selection_changed": False,
        },
        "development_split_sha256": SPLIT_SHA256,
        "threshold": THRESHOLD,
        "seeds": [17, 29, 43],
        "arms": arms,
        "promotion_gate": plan["promotion_gate"],
        "single_factor_change": plan["single_factor_change"],
        "distance_loss_weight": 0.0,
        "distance_label_cache_supplied": False,
        "selection_or_tuning": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_path = authority / "CAMPAIGN_CONFIDENCE_GEOMETRY_PLDDT70_V1.json"
    campaign_path.write_bytes(canonical_bytes(campaign))
    checksums.extend(
        (
            (sha256_file(plan_path), plan_path),
            (sha256_file(plan_receipt_path), plan_receipt_path),
            (masked_sha, masked_manifest_path),
            (masked_receipt_sha, masked_receipt_path),
            (contact_sha, contact_manifest_path),
            (contact_receipt_sha, contact_receipt_path),
            (sha256_file(campaign_path), campaign_path),
        )
    )
    checksums_path = authority / "AUTHORITY_SHA256SUMS.txt"
    checksums_path.write_text(
        "".join(f"{digest}  {path}\n" for digest, path in checksums),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "campaign_sha256": sha256_file(campaign_path),
                "masked_structure_manifest_sha256": masked_sha,
                "masked_structure_receipt_sha256": masked_receipt_sha,
                "authority_checksums_sha256": sha256_file(checksums_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
