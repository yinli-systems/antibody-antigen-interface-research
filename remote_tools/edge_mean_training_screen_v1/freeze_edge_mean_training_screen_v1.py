#!/usr/bin/env python3
"""Freeze a paired baseline versus edge-supervised epitope readout screen."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys


SOURCE_SPLIT_SHA256 = "4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd"
PARTITION_SHA256 = "a1a21789c1d1890ff957b92f966720fd169b2b4004fe4acd17c693c37658d73a"
READOUT_SELECTION_RESULT_SHA256 = "84c7cefc108b38018e885a5318659107ff01f915d24d9c77a78ff36ecacd3d1c"
TEMPLATE_PROTOCOL_SHA256 = "2678f09ff99639862acb75147b100deffb14046bdded4d9c06ca31181bfae335"
METADATA_SHA256 = "76d2683558622af8e616d1ba39d24486145be9816ed17294fbdb62c9d03ebe75"
TRAIN_PANEL_SHA256 = "2ba7b5a3d62b731e81583ff09acca30e8be56c0452166cbba822a811dcd20a24"
BASE_SOURCE_MANIFEST_SHA256 = "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
CANDIDATE_SOURCE_MANIFEST_SHA256 = "993205cff450c8585bf79757fa0e97f5bbc96a5cf48b8df39bbed822d624c6a3"
BASE_FRAMECONTACT_SHA256 = "a85584f288b32e66e6ab34bcd93b0001ffcd2b173bb1b0fc9e3ab48bccd6171c"
CANDIDATE_FRAMECONTACT_SHA256 = "4d2229ebbf4ea8729d99dd70e36d9f4ccc4d7d709c42b1f5a544d248194a9b5f"


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
    parser.add_argument("--base-source-root", type=Path, required=True)
    parser.add_argument("--candidate-source-root", type=Path, required=True)
    parser.add_argument("--base-source-manifest", type=Path, required=True)
    parser.add_argument("--candidate-source-manifest", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--readout-selection-result", type=Path, required=True)
    parser.add_argument("--template-protocol", type=Path, required=True)
    parser.add_argument("--partner-metadata", type=Path, required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for path, expected, label in (
        (args.source_split, SOURCE_SPLIT_SHA256, "source split"),
        (args.partition, PARTITION_SHA256, "cluster partition"),
        (
            args.readout_selection_result,
            READOUT_SELECTION_RESULT_SHA256,
            "readout selection result",
        ),
        (args.template_protocol, TEMPLATE_PROTOCOL_SHA256, "template protocol"),
        (args.partner_metadata, METADATA_SHA256, "partner metadata"),
        (args.train_panel, TRAIN_PANEL_SHA256, "training partner panel"),
        (
            args.base_source_manifest,
            BASE_SOURCE_MANIFEST_SHA256,
            "baseline source manifest",
        ),
        (
            args.candidate_source_manifest,
            CANDIDATE_SOURCE_MANIFEST_SHA256,
            "candidate source manifest",
        ),
        (
            args.base_source_root / "src/interaction_first/framecontact.py",
            BASE_FRAMECONTACT_SHA256,
            "baseline FrameContact source",
        ),
        (
            args.candidate_source_root / "src/interaction_first/framecontact.py",
            CANDIDATE_FRAMECONTACT_SHA256,
            "candidate FrameContact source",
        ),
    ):
        authenticate(path, expected, label)
    selection_result = json.loads(args.readout_selection_result.read_text())
    if (
        selection_result.get("status") != "FAIL_CLEAN"
        or selection_result.get("winner") is not None
        or selection_result.get("authorization", {}).get("confirmation_panel") is not False
        or selection_result.get("authorization", {}).get("old_test") is not False
        or selection_result.get("authorization", {}).get("temporal_holdout") is not False
    ):
        raise ValueError("readout-only prerequisite differs")
    partition = json.loads(args.partition.read_text())
    selection_entries = partition.get("roles", {}).get("selection")
    confirmation_entries = partition.get("roles", {}).get("confirmation")
    if not isinstance(selection_entries, list) or not isinstance(
        confirmation_entries, list
    ):
        raise ValueError("cluster partition roles are missing")
    selection_ids = {str(entry["instance_id"]) for entry in selection_entries}
    confirmation_ids = {str(entry["instance_id"]) for entry in confirmation_entries}
    selection_clusters = {
        str(entry["antigen_cluster_id"]) for entry in selection_entries
    }
    confirmation_clusters = {
        str(entry["antigen_cluster_id"]) for entry in confirmation_entries
    }
    if selection_ids & confirmation_ids or selection_clusters & confirmation_clusters:
        raise ValueError("selection and confirmation roles overlap")

    with args.source_split.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)
    training_rows = []
    original_train = 0
    for source_row in source_rows:
        row = dict(source_row)
        if row.get("role") == "train":
            original_train += 1
        elif row.get("role") == "validation":
            row["role"] = "validation" if row["instance_id"] in selection_ids else "test"
        training_rows.append(row)
    if original_train != 512:
        raise ValueError(f"expected 512 frozen training rows, observed {original_train}")

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    authority = args.output_root / "authority"
    authority.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    (args.output_root / "runs").mkdir()
    training_split = authority / "split-train512-selection63.csv"
    training_split_sha256 = write_csv_new(training_split, fieldnames, training_rows)

    source_root = args.base_source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))
    from interaction_first.reciprocal_partner_panel import (  # noqa: PLC0415
        build_reciprocal_partner_panel,
    )

    with args.partner_metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        metadata_fieldnames = list(reader.fieldnames or [])
        metadata_rows = list(reader)
    bound_rows = []
    for source_row in metadata_rows:
        row = dict(source_row)
        if row["instance_id"] in selection_ids:
            row["role"] = "validation"
        elif row.get("role") == "train":
            row["role"] = "train"
        else:
            row["role"] = "excluded"
        bound_rows.append(row)
    bound_metadata = authority / "SCREEN_PARTNER_METADATA_SELECTION63.csv"
    bound_metadata_sha256 = write_csv_new(
        bound_metadata, metadata_fieldnames, bound_rows
    )
    training_panel = authority / "train512-reciprocal-6.json"
    build_reciprocal_partner_panel(
        metadata_path=bound_metadata,
        metadata_sha256=bound_metadata_sha256,
        output_path=training_panel,
        role="train",
        panel_size_per_direction=6,
        protocol_seed=20260826,
    )
    training_panel_sha256 = sha256_file(training_panel)
    validation_panel = authority / "validation-selection63-reciprocal-6.json"
    build_reciprocal_partner_panel(
        metadata_path=bound_metadata,
        metadata_sha256=bound_metadata_sha256,
        output_path=validation_panel,
        role="validation",
        panel_size_per_direction=6,
        protocol_seed=20260826,
    )
    validation_panel_sha256 = sha256_file(validation_panel)
    validation_panel_payload = json.loads(validation_panel.read_text())
    if len(validation_panel_payload.get("entries", [])) != len(selection_ids):
        raise ValueError("selection validation partner panel count differs")

    template = json.loads(args.template_protocol.read_text())
    protocol_hashes = {}
    receipt_hashes = {}
    arms = {
        "baseline_current": {
            "readout": "direct_head_plus_logmeanexp_contact",
            "source_manifest_sha256": BASE_SOURCE_MANIFEST_SHA256,
            "framecontact_sha256": BASE_FRAMECONTACT_SHA256,
        },
        "edge_mean_supervised": {
            "readout": "paper_mean_edge_probability_no_direct_head",
            "source_manifest_sha256": CANDIDATE_SOURCE_MANIFEST_SHA256,
            "framecontact_sha256": CANDIDATE_FRAMECONTACT_SHA256,
        },
    }
    for arm, specification in arms.items():
        protocol = copy.deepcopy(template)
        protocol["schema_version"] = "sabdab2_edge_mean_training_screen_protocol_v1"
        protocol["arm"] = arm
        protocol["authority"] = {
            "cluster_partition_sha256": PARTITION_SHA256,
            "readout_only_selection_result_sha256": READOUT_SELECTION_RESULT_SHA256,
            "source_manifest_sha256": specification["source_manifest_sha256"],
            "framecontact_sha256": specification["framecontact_sha256"],
        }
        protocol["claim_boundary"].update(
            {
                "selection_panel_only": True,
                "confirmation_panel_loaded": False,
                "test_loaded": False,
                "temporal_holdout_loaded": False,
            }
        )
        protocol["data"]["screen_split_sha256"] = training_split_sha256
        protocol["data"]["development_screen"] = {
            "train_count": 512,
            "validation_count": len(selection_ids),
            "selection_antigen_cluster_count": len(selection_clusters),
            "confirmation_count_sealed": len(confirmation_ids),
            "confirmation_antigen_cluster_count_sealed": len(
                confirmation_clusters
            ),
            "test_features_or_labels_loaded": False,
        }
        protocol["data"]["reciprocal_partner_panel"][
            "validation_sha256"
        ] = validation_panel_sha256
        protocol["data"]["reciprocal_partner_panel"][
            "train_sha256"
        ] = training_panel_sha256
        protocol["data"]["reciprocal_partner_panel"]["size_per_query"] = 6
        protocol["model_selection"]["seeds"] = [17]
        protocol["model_selection"]["checkpoint_rule"] = (
            "maximize selection-panel epitope mean-per-complex AUPRC; break ties "
            "by contact mean-per-complex AUPRC and earlier epoch"
        )
        protocol["tasks"]["winner"]["name"] = arm
        protocol["epitope_readout"] = {
            "rule": specification["readout"],
            "single_factor_only": True,
            "joint_contact_paratope_epitope_training_preserved": True,
            "loss_weights_unchanged": True,
        }
        protocol_path = authority / f"PROTOCOL_{arm.upper()}_V1.json"
        protocol_sha256 = write_new(protocol_path, protocol)
        receipt = {
            "format_version": "sabdab2_edge_mean_training_protocol_receipt_v1",
            "arm": arm,
            "protocol_file_sha256": protocol_sha256,
            "source_manifest_sha256": specification["source_manifest_sha256"],
            "training_split_sha256": training_split_sha256,
            "training_partner_panel_sha256": training_panel_sha256,
            "validation_partner_panel_sha256": validation_panel_sha256,
            "frozen_before_training": True,
            "confirmation_panel_loaded": False,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        receipt_path = authority / f"PROTOCOL_{arm.upper()}_V1_RECEIPT.json"
        receipt_sha256 = write_new(receipt_path, receipt)
        protocol_hashes[arm] = protocol_sha256
        receipt_hashes[arm] = receipt_sha256
    campaign = {
        "format_version": "sabdab2_edge_mean_training_campaign_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "arms": arms,
        "protocol_sha256": protocol_hashes,
        "protocol_receipt_sha256": receipt_hashes,
        "training_split_sha256": training_split_sha256,
        "source_training_partner_panel_sha256": TRAIN_PANEL_SHA256,
        "training_partner_panel_sha256": training_panel_sha256,
        "validation_partner_panel_sha256": validation_panel_sha256,
        "cluster_partition_sha256": PARTITION_SHA256,
        "selection_seed": 17,
        "promotion_scope": "three_seed_confirmation_panel_only",
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    campaign_sha256 = write_new(
        authority / "CAMPAIGN_EDGE_MEAN_TRAINING_V1.json", campaign
    )
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "campaign_sha256": campaign_sha256,
                "training_split_sha256": training_split_sha256,
                "validation_partner_panel_sha256": validation_panel_sha256,
                "selection_complexes": len(selection_ids),
                "selection_clusters": len(selection_clusters),
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
