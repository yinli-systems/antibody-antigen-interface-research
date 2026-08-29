#!/usr/bin/env python3
"""Freeze a metadata-only contact-authority recovery before retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


SOURCE_CONTACT_SHA256 = (
    "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
)
ADAPTER_MANIFEST_SHA256 = (
    "d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88"
)
ADAPTER_RECEIPT_SHA256 = (
    "426975feebdb5294b0fdbc45d31422fb3adbadc4d97b032958a38d19b050f1d9"
)
FAILURE_RECEIPT_SHA256 = (
    "87c2baf1deccb2c1eb6050027906c8f6f4f5c5e949f2c9b995820dbf88cca54e"
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--training-launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--adapter-builder", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest", type=Path, required=True)
    parser.add_argument("--expert-source-manifest", type=Path, required=True)
    return parser.parse_args()


def protocol_receipt(
    path: Path,
    protocol_sha256: str,
    *,
    arm: str,
    structure_sha256: str,
    contact_sha256: str,
    split_sha256: str,
) -> str:
    return write_json(
        path,
        {
            "format_version": (
                "sabdab2_predicted_geometry_contact_rebind_protocol_receipt_v1"
            ),
            "arm": arm,
            "protocol_file_sha256": protocol_sha256,
            "structure_manifest_sha256": structure_sha256,
            "contact_adapter_manifest_sha256": contact_sha256,
            "development_split_view_sha256": split_sha256,
            "frozen_before_recovery_training": True,
            "scientific_protocol_changed": False,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    )


def main() -> None:
    args = parse_args()
    original_authority = args.campaign_root / "authority"
    authority = original_authority / "contact_rebind_recovery_v1"
    if authority.exists() or authority.is_symlink():
        raise FileExistsError(authority)

    structure_root = args.campaign_root / "data" / "structure_cache_v1"
    contact_root = (
        args.campaign_root / "data" / "contact_cache_predicted_geometry_authority_v1"
    )
    structure_manifest = structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    split_path = structure_root / "DEVELOPMENT_SPLIT_VIEW.csv"
    contact_manifest = contact_root / "CONTACT_CACHE_MANIFEST.json"
    contact_receipt_path = contact_root / "CONTACT_CACHE_RECEIPT.json"
    failure_receipt = args.campaign_root / "FAIL_CLOSED_TRAINING_1483576.json"
    original_campaign_path = (
        original_authority / "CAMPAIGN_PREDICTED_GEOMETRY_DEVELOPMENT_V1.json"
    )
    baseline_template_path = (
        original_authority / "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_V1.json"
    )
    expert_template_path = (
        original_authority / "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_V1.json"
    )

    structure_sha256 = sha256_file(structure_manifest)
    split_sha256 = sha256_file(split_path)
    if (
        sha256_file(contact_manifest) != ADAPTER_MANIFEST_SHA256
        or sha256_file(contact_receipt_path) != ADAPTER_RECEIPT_SHA256
        or sha256_file(failure_receipt) != FAILURE_RECEIPT_SHA256
    ):
        raise ValueError("contact recovery evidence hash differs")
    contact = json.loads(contact_manifest.read_text(encoding="utf-8"))
    contact_receipt = json.loads(contact_receipt_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_receipt.read_text(encoding="utf-8"))
    if (
        contact.get("authority", {}).get("source_contact_manifest_sha256")
        != SOURCE_CONTACT_SHA256
        or contact.get("authority", {}).get("structure_manifest_sha256")
        != structure_sha256
        or contact.get("counts", {}).get("records") != 3114
        or contact.get("counts", {}).get("by_role")
        != {"train": 2565, "validation": 549}
        or len(contact.get("records", [])) != 3114
        or contact_receipt.get("status")
        != "PASS_PROVENANCE_PRESERVING_CONTACT_AUTHORITY_ADAPTER"
        or contact_receipt.get("all_pair_artifact_sha256_verified") is not True
        or contact_receipt.get("all_pair_artifacts_hard_linked") is not True
        or contact_receipt.get("pair_labels_regenerated") is not False
        or contact_receipt.get("pair_label_artifacts_modified") is not False
        or contact_receipt.get("test_pair_artifacts_read") is not False
        or contact_receipt.get("temporal_holdout_access") is not False
        or failure.get("status") != "FAIL_CLOSED_NEW_FAILURE_TYPE"
        or failure.get("training_updates_completed") != 0
        or failure.get("test_access") is not False
        or failure.get("temporal_holdout_access") is not False
    ):
        raise ValueError("contact recovery evidence content differs")

    baseline_source_sha256 = sha256_file(args.baseline_source_manifest)
    expert_source_sha256 = sha256_file(args.expert_source_manifest)
    if (
        baseline_source_sha256
        != "ca31451b4be85ee1e07045609a41ed3c90789971368191383c28cbf4251b581e"
        or expert_source_sha256
        != "f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55"
    ):
        raise ValueError("training source manifest differs")

    authority.mkdir(parents=True)
    baseline_source_copy = authority / "SOURCE_MANIFEST_BASELINE_V3.json"
    expert_source_copy = authority / "SOURCE_MANIFEST_EDGE_TOPK2_V1.json"
    shutil.copyfile(args.baseline_source_manifest, baseline_source_copy)
    shutil.copyfile(args.expert_source_manifest, expert_source_copy)

    execution = {
        "recovery_freezer_sha256": sha256_file(Path(__file__)),
        "training_launcher_sha256": sha256_file(args.training_launcher),
        "analyzer_sha256": sha256_file(args.analyzer),
        "base_analyzer_sha256": sha256_file(
            args.analyzer.with_name("analyze_predicted_geometry_development_v1.py")
        ),
        "adapter_builder_sha256": sha256_file(args.adapter_builder),
        "contact_adapter_manifest_sha256": ADAPTER_MANIFEST_SHA256,
        "contact_adapter_receipt_sha256": ADAPTER_RECEIPT_SHA256,
        "failure_receipt_sha256": FAILURE_RECEIPT_SHA256,
    }
    baseline = json.loads(baseline_template_path.read_text(encoding="utf-8"))
    expert = json.loads(expert_template_path.read_text(encoding="utf-8"))
    for protocol, arm in (
        (baseline, "baseline"),
        (expert, "edge_topk2_supervised"),
    ):
        protocol["schema_version"] += "_contact_rebind_recovery_v1"
        protocol["protocol_status"] = (
            "frozen_before_contact_rebind_recovery_training"
        )
        protocol["data"]["source_contact_manifest_sha256"] = SOURCE_CONTACT_SHA256
        protocol["data"]["contact_manifest_sha256"] = ADAPTER_MANIFEST_SHA256
        protocol["data"]["contact_authority_adapter"] = {
            "type": "development_only_pair_label_rebind_without_artifact_change",
            "receipt_sha256": ADAPTER_RECEIPT_SHA256,
            "pair_labels_regenerated": False,
            "pair_label_artifacts_modified": False,
            "residue_axis_lengths_verified": True,
        }
        protocol["claim_boundary"]["contact_pair_labels_unchanged"] = True
        protocol["claim_boundary"]["scientific_protocol_changed_by_recovery"] = False
        protocol["contact_authority_recovery"] = {
            **execution,
            "failed_array_job_id": "1483576",
            "failure_class": "CONTACT_STRUCTURE_AUTHORITY_MISMATCH",
            "training_updates_before_recovery": 0,
        }
        key = "source_authority" if arm == "baseline" else "execution_sources"
        protocol[key] = {**protocol.get(key, {}), **execution}

    baseline_path = (
        authority
        / "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_CONTACT_REBIND_RECOVERY_V1.json"
    )
    expert_path = (
        authority
        / "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_CONTACT_REBIND_RECOVERY_V1.json"
    )
    baseline_sha256 = write_json(baseline_path, baseline)
    expert_sha256 = write_json(expert_path, expert)
    baseline_receipt_path = baseline_path.with_name(
        baseline_path.stem + "_RECEIPT.json"
    )
    expert_receipt_path = expert_path.with_name(expert_path.stem + "_RECEIPT.json")
    baseline_receipt_sha256 = protocol_receipt(
        baseline_receipt_path,
        baseline_sha256,
        arm="baseline",
        structure_sha256=structure_sha256,
        contact_sha256=ADAPTER_MANIFEST_SHA256,
        split_sha256=split_sha256,
    )
    expert_receipt_sha256 = protocol_receipt(
        expert_receipt_path,
        expert_sha256,
        arm="edge_topk2_supervised",
        structure_sha256=structure_sha256,
        contact_sha256=ADAPTER_MANIFEST_SHA256,
        split_sha256=split_sha256,
    )

    campaign = {
        "format_version": (
            "sabdab2_predicted_geometry_contact_rebind_recovery_campaign_v1"
        ),
        "status": "FROZEN_BEFORE_CONTACT_REBIND_RECOVERY_TRAINING",
        "campaign_root": str(args.campaign_root),
        "base_campaign_sha256": sha256_file(original_campaign_path),
        "failed_array_job_id": "1483576",
        "failure_receipt_sha256": FAILURE_RECEIPT_SHA256,
        "structure_cache_root": str(structure_root),
        "structure_manifest_sha256": structure_sha256,
        "contact_adapter_root": str(contact_root),
        "source_contact_manifest_sha256": SOURCE_CONTACT_SHA256,
        "contact_adapter_manifest_sha256": ADAPTER_MANIFEST_SHA256,
        "contact_adapter_receipt_sha256": ADAPTER_RECEIPT_SHA256,
        "development_split_view_sha256": split_sha256,
        "arms": {
            "baseline": {
                "protocol_sha256": baseline_sha256,
                "receipt_sha256": baseline_receipt_sha256,
                "source_manifest_sha256": baseline_source_sha256,
            },
            "edge_topk2_supervised": {
                "protocol_sha256": expert_sha256,
                "receipt_sha256": expert_receipt_sha256,
                "source_manifest_sha256": expert_source_sha256,
            },
        },
        "seeds": [17, 29, 43],
        "scientific_protocol_changed": False,
        "model_architecture_changed": False,
        "loss_or_optimization_changed": False,
        "pair_labels_changed": False,
        "split_changed": False,
        "selection_or_tuning": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
        "execution_sources": execution,
    }
    campaign_path = (
        authority / "CAMPAIGN_PREDICTED_GEOMETRY_CONTACT_REBIND_RECOVERY_V1.json"
    )
    campaign_sha256 = write_json(campaign_path, campaign)
    lines = [
        f"{structure_sha256}  {structure_manifest}",
        f"{split_sha256}  {split_path}",
        f"{ADAPTER_MANIFEST_SHA256}  {contact_manifest}",
        f"{ADAPTER_RECEIPT_SHA256}  {contact_receipt_path}",
        f"{FAILURE_RECEIPT_SHA256}  {failure_receipt}",
        f"{baseline_source_sha256}  {baseline_source_copy}",
        f"{expert_source_sha256}  {expert_source_copy}",
        f"{baseline_sha256}  {baseline_path}",
        f"{baseline_receipt_sha256}  {baseline_receipt_path}",
        f"{expert_sha256}  {expert_path}",
        f"{expert_receipt_sha256}  {expert_receipt_path}",
        f"{campaign_sha256}  {campaign_path}",
    ]
    (authority / "AUTHORITY_SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "campaign_sha256": campaign_sha256,
                "baseline_protocol_sha256": baseline_sha256,
                "baseline_receipt_sha256": baseline_receipt_sha256,
                "expert_protocol_sha256": expert_sha256,
                "expert_receipt_sha256": expert_receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
