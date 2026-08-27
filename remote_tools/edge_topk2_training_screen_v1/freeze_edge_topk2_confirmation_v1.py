#!/usr/bin/env python3
"""Freeze paired three-seed training and a sealed confirmation measurement."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys


SELECTION_RESULT_SHA256 = "3e74ba4fd4d2f4b6deffd68ae0f1420712ac8425ddba6b4bdcd3b0e1cbad416c"
PARTITION_SHA256 = "a1a21789c1d1890ff957b92f966720fd169b2b4004fe4acd17c693c37658d73a"
CONFIRMATION_SPLIT_SHA256 = "69350cfe30348c818fa9998bc5063df3c235954633fae46e20a2295f751efad4"
PARTNER_METADATA_SHA256 = "76d2683558622af8e616d1ba39d24486145be9816ed17294fbdb62c9d03ebe75"
SELECTION_SPLIT_SHA256 = "39a32540dd4a6d9abcc8414c23dfbe52d446e9154c06ce58921a0dcbd5039686"
TRAIN_PANEL_SHA256 = "c209b72cfe174c63fa986cb59e8dbf6a3831f128493b0c992a604ddf6a6cfca6"
SELECTION_PANEL_SHA256 = "86d613c1917d3a8d3fe6da49439cb0a13ab9f44affd2ea27e30043f3d71ae6e2"
BASE_PROTOCOL_SHA256 = "7cb1ab917137dfe1de821a1eea846fd22e656a12c1d3dd99a45121bf9f77e181"
TOPK2_PROTOCOL_SHA256 = "cac9957ec6c34d27f2c29cbde8d012ccfcb1ca779d4ea378376429d1a238ca94"
BASE_SOURCE_MANIFEST_SHA256 = "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
TOPK2_SOURCE_MANIFEST_SHA256 = "f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55"
BASE_FRAMECONTACT_SHA256 = "a85584f288b32e66e6ab34bcd93b0001ffcd2b173bb1b0fc9e3ab48bccd6171c"
TOPK2_FRAMECONTACT_SHA256 = "121b3e06e24db0eeb79db100a3870d19b299b44f77355b590c97ab45cc8f38fe"
STRUCTURE_MANIFEST_SHA256 = "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0"
CONTACT_MANIFEST_SHA256 = "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
EMBEDDING_MANIFEST_SHA256 = "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545"


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


def authenticate(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def write_new(path: Path, payload: object) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.write_bytes(canonical_bytes(payload))
    return sha256_file(path)


def copy_new(source: Path, destination: Path) -> str:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.write_bytes(source.read_bytes())
    return sha256_file(destination)


def write_csv_new(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--confirmation-split", type=Path, required=True)
    parser.add_argument("--partner-metadata", type=Path, required=True)
    parser.add_argument("--base-source-root", type=Path, required=True)
    parser.add_argument("--base-source-manifest", type=Path, required=True)
    parser.add_argument("--topk2-source-root", type=Path, required=True)
    parser.add_argument("--topk2-source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    selection_authority = args.selection_root / "authority"
    parent_authority = args.parent_root / "authority"
    selection_result = args.selection_root / "EDGE_TOPK2_TRAINING_SELECTION_RESULT_V1.json"
    base_protocol = parent_authority / "PROTOCOL_BASELINE_CURRENT_V1.json"
    topk2_protocol = selection_authority / "PROTOCOL_EDGE_TOPK2_SUPERVISED_V1.json"
    checks = (
        (selection_result, SELECTION_RESULT_SHA256, "selection result"),
        (args.partition, PARTITION_SHA256, "cluster partition"),
        (args.confirmation_split, CONFIRMATION_SPLIT_SHA256, "confirmation split"),
        (args.partner_metadata, PARTNER_METADATA_SHA256, "partner metadata"),
        (parent_authority / "split-train512-selection63.csv", SELECTION_SPLIT_SHA256, "selection split"),
        (parent_authority / "train512-reciprocal-6.json", TRAIN_PANEL_SHA256, "training panel"),
        (parent_authority / "validation-selection63-reciprocal-6.json", SELECTION_PANEL_SHA256, "selection panel"),
        (base_protocol, BASE_PROTOCOL_SHA256, "baseline protocol"),
        (topk2_protocol, TOPK2_PROTOCOL_SHA256, "top-k2 protocol"),
        (args.base_source_manifest, BASE_SOURCE_MANIFEST_SHA256, "baseline source manifest"),
        (args.topk2_source_manifest, TOPK2_SOURCE_MANIFEST_SHA256, "top-k2 source manifest"),
        (args.base_source_root / "src/interaction_first/framecontact.py", BASE_FRAMECONTACT_SHA256, "baseline FrameContact"),
        (args.topk2_source_root / "src/interaction_first/framecontact.py", TOPK2_FRAMECONTACT_SHA256, "top-k2 FrameContact"),
    )
    for path, expected, label in checks:
        authenticate(path, expected, label)

    selected = json.loads(selection_result.read_text())
    if (
        selected.get("status") != "PASS_FREEZE_THREE_SEED_CONFIRMATION"
        or selected.get("authorization", {}).get(
            "freeze_three_seed_confirmation_development_protocol"
        )
        is not True
        or selected.get("authorization", {}).get("three_seed_confirmation_started")
        is not False
        or selected.get("authority", {}).get("confirmation_panel_loaded") is not False
        or selected.get("authority", {}).get("test_loaded") is not False
        or selected.get("authority", {}).get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("selection gate does not authorize confirmation freeze")

    partition = json.loads(args.partition.read_text())
    selection_entries = partition.get("roles", {}).get("selection")
    confirmation_entries = partition.get("roles", {}).get("confirmation")
    if not isinstance(selection_entries, list) or not isinstance(confirmation_entries, list):
        raise ValueError("partition roles are missing")
    selection_ids = {str(row["instance_id"]) for row in selection_entries}
    confirmation_ids = {str(row["instance_id"]) for row in confirmation_entries}
    selection_clusters = {str(row["antigen_cluster_id"]) for row in selection_entries}
    confirmation_clusters = {str(row["antigen_cluster_id"]) for row in confirmation_entries}
    if (
        len(selection_ids) != 63
        or len(confirmation_ids) != 65
        or len(selection_clusters) != 32
        or len(confirmation_clusters) != 27
        or selection_ids & confirmation_ids
        or selection_clusters & confirmation_clusters
    ):
        raise ValueError("selection and confirmation partition differs")

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    authority = args.output_root / "authority"
    authority.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    (args.output_root / "runs").mkdir()
    (args.output_root / "confirmation-results").mkdir()

    selection_split_sha256 = copy_new(
        parent_authority / "split-train512-selection63.csv",
        authority / "split-train512-selection63.csv",
    )
    train_panel_sha256 = copy_new(
        parent_authority / "train512-reciprocal-6.json",
        authority / "train512-reciprocal-6.json",
    )
    selection_panel_sha256 = copy_new(
        parent_authority / "validation-selection63-reciprocal-6.json",
        authority / "validation-selection63-reciprocal-6.json",
    )
    confirmation_split_sha256 = copy_new(
        args.confirmation_split,
        authority / "split-confirmation65-measurement-only.csv",
    )

    with args.partner_metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        metadata_fieldnames = list(reader.fieldnames or [])
        metadata_rows = list(reader)
    bound_rows = []
    for source_row in metadata_rows:
        row = dict(source_row)
        if row["instance_id"] in confirmation_ids:
            row["role"] = "validation"
        elif row.get("role") == "train":
            row["role"] = "train"
        else:
            row["role"] = "excluded"
        bound_rows.append(row)
    bound_metadata = authority / "SCREEN_PARTNER_METADATA_CONFIRMATION65.csv"
    bound_metadata_sha256 = write_csv_new(
        bound_metadata, metadata_fieldnames, bound_rows
    )

    sys.path.insert(0, str(args.base_source_root.resolve() / "src"))
    from interaction_first.reciprocal_partner_panel import (  # noqa: PLC0415
        build_reciprocal_partner_panel,
    )

    confirmation_panel = authority / "validation-confirmation65-reciprocal-6.json"
    build_reciprocal_partner_panel(
        metadata_path=bound_metadata,
        metadata_sha256=bound_metadata_sha256,
        output_path=confirmation_panel,
        role="validation",
        panel_size_per_direction=6,
        protocol_seed=20260826,
    )
    confirmation_panel_payload = json.loads(confirmation_panel.read_text())
    if len(confirmation_panel_payload.get("entries", [])) != 65:
        raise ValueError("confirmation partner panel count differs")
    confirmation_panel_sha256 = sha256_file(confirmation_panel)

    arm_specs = {
        "baseline_current": {
            "template": base_protocol,
            "source_manifest_sha256": BASE_SOURCE_MANIFEST_SHA256,
            "framecontact_sha256": BASE_FRAMECONTACT_SHA256,
            "readout": "direct_head_plus_logmeanexp_contact",
        },
        "edge_topk2_supervised": {
            "template": topk2_protocol,
            "source_manifest_sha256": TOPK2_SOURCE_MANIFEST_SHA256,
            "framecontact_sha256": TOPK2_FRAMECONTACT_SHA256,
            "readout": "masked_top_k_mean_2_edge_probability_no_direct_head",
        },
    }
    protocol_hashes = {}
    receipt_hashes = {}
    for arm, specification in arm_specs.items():
        protocol = copy.deepcopy(json.loads(specification["template"].read_text()))
        protocol["schema_version"] = "sabdab2_edge_topk2_confirmation_training_protocol_v1"
        protocol["arm"] = arm
        protocol["authority"] = {
            "selection_result_sha256": SELECTION_RESULT_SHA256,
            "partition_sha256": PARTITION_SHA256,
            "source_manifest_sha256": specification["source_manifest_sha256"],
            "framecontact_sha256": specification["framecontact_sha256"],
        }
        protocol["model_selection"]["seeds"] = [29, 43]
        protocol["tasks"]["winner"]["name"] = arm
        protocol["epitope_readout"] = {
            "rule": specification["readout"],
            "single_factor_only": True,
            "joint_contact_paratope_epitope_training_preserved": True,
            "loss_weights_unchanged": True,
        }
        protocol["claim_boundary"].update(
            {
                "selection_panel_only": True,
                "confirmation_panel_loaded": False,
                "test_loaded": False,
                "temporal_holdout_loaded": False,
            }
        )
        protocol["data"]["screen_split_sha256"] = selection_split_sha256
        protocol["data"]["reciprocal_partner_panel"]["train_sha256"] = train_panel_sha256
        protocol["data"]["reciprocal_partner_panel"]["validation_sha256"] = selection_panel_sha256
        protocol_path = authority / f"PROTOCOL_{arm.upper()}_CONFIRMATION_TRAINING_V1.json"
        protocol_sha256 = write_new(protocol_path, protocol)
        receipt = {
            "format_version": "sabdab2_edge_topk2_confirmation_training_receipt_v1",
            "arm": arm,
            "protocol_file_sha256": protocol_sha256,
            "source_manifest_sha256": specification["source_manifest_sha256"],
            "training_split_sha256": selection_split_sha256,
            "training_partner_panel_sha256": train_panel_sha256,
            "selection_partner_panel_sha256": selection_panel_sha256,
            "frozen_before_training": True,
            "confirmation_panel_loaded": False,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        receipt_sha256 = write_new(
            authority / f"PROTOCOL_{arm.upper()}_CONFIRMATION_TRAINING_V1_RECEIPT.json",
            receipt,
        )
        protocol_hashes[arm] = protocol_sha256
        receipt_hashes[arm] = receipt_sha256

    measurement_contract = {
        "format_version": "sabdab2_edge_topk2_confirmation_measurement_contract_v1",
        "status": "FROZEN_BEFORE_CHECKPOINT_BINDING",
        "arms": list(arm_specs),
        "seeds": [17, 29, 43],
        "measurement_only": True,
        "model_parameters_updated_during_measurement": False,
        "checkpoint_binding": "required_after_training_before_measurement",
        "panel_role": "cluster_disjoint_confirmation65",
        "partition_sha256": PARTITION_SHA256,
        "confirmation_split_sha256": confirmation_split_sha256,
        "confirmation_partner_metadata_sha256": bound_metadata_sha256,
        "confirmation_partner_panel_sha256": confirmation_panel_sha256,
        "calibration": "reuse_each_checkpoint_selection_panel_temperatures_and_thresholds",
        "metrics": [
            "contact_mean_per_complex_auprc",
            "epitope_mean_per_complex_auprc",
            "paratope_mean_per_complex_auprc",
            "contact_brier",
            "epitope_brier",
            "paratope_brier",
            "contact_ece_15",
            "epitope_ece_15",
            "paratope_ece_15",
            "reciprocal_partner_metrics",
        ],
        "promotion_gate": {
            "median_epitope_delta_minimum": 0.005,
            "epitope_positive_seed_count_minimum": 2,
            "minimum_seed_epitope_delta": -0.01,
            "median_contact_delta_minimum": -0.01,
            "minimum_seed_contact_delta": -0.02,
            "median_paratope_delta_minimum": -0.01,
            "minimum_seed_paratope_delta": -0.02,
        },
        "authorization_after_pass": "freeze_full_development_protocol_only",
        "full_development_started": False,
        "old_test": False,
        "temporal_holdout": False,
    }
    measurement_contract_sha256 = write_new(
        authority / "PROTOCOL_CONFIRMATION_MEASUREMENT_CONTRACT_V1.json",
        measurement_contract,
    )
    campaign = {
        "format_version": "sabdab2_edge_topk2_three_seed_confirmation_campaign_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "selection_result_sha256": SELECTION_RESULT_SHA256,
        "partition_sha256": PARTITION_SHA256,
        "training_seeds": [29, 43],
        "confirmation_seeds": [17, 29, 43],
        "training_protocol_sha256": protocol_hashes,
        "training_protocol_receipt_sha256": receipt_hashes,
        "selection_split_sha256": selection_split_sha256,
        "training_partner_panel_sha256": train_panel_sha256,
        "selection_partner_panel_sha256": selection_panel_sha256,
        "confirmation_split_sha256": confirmation_split_sha256,
        "confirmation_partner_panel_sha256": confirmation_panel_sha256,
        "measurement_contract_sha256": measurement_contract_sha256,
        "confirmation_labels_read": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_sha256 = write_new(
        authority / "CAMPAIGN_EDGE_TOPK2_THREE_SEED_CONFIRMATION_V1.json",
        campaign,
    )
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "campaign_sha256": campaign_sha256,
                "measurement_contract_sha256": measurement_contract_sha256,
                "confirmation_complexes": len(confirmation_ids),
                "confirmation_clusters": len(confirmation_clusters),
                "confirmation_labels_read": False,
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
