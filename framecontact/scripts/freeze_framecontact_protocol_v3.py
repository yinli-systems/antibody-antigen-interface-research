#!/usr/bin/env python3
"""Validate and seal the executable SAbDab2 FrameContact v3 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "framecontact_primary_protocol_v3"
RECEIPT_SCHEMA = "framecontact_protocol_freeze_receipt_v3"


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--split-receipt", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--usalign-receipt", type=Path, required=True)
    parser.add_argument("--usalign-table", type=Path, required=True)
    parser.add_argument("--cache-verification", type=Path, required=True)
    parser.add_argument("--esm-manifest", type=Path, required=True)
    parser.add_argument("--partner-metadata-receipt", type=Path, required=True)
    parser.add_argument("--training-panel", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--test-panel", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite frozen receipt: {arguments.receipt}")
    raw = arguments.protocol.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("Unknown v3 protocol schema")
    parent = protocol.get("parent_protocol", {})
    if sha256_file(arguments.parent_protocol) != parent.get("sha256"):
        raise ValueError("Parent protocol digest differs")
    if sha256_file(arguments.parent_receipt) != parent.get("receipt_sha256"):
        raise ValueError("Parent receipt digest differs")
    authority = protocol.get("data", {}).get("authorities", {})
    paths = {
        "split_receipt_sha256": arguments.split_receipt,
        "structural_audit_sha256": arguments.structural_audit,
        "usalign_receipt_sha256": arguments.usalign_receipt,
        "usalign_table_sha256": arguments.usalign_table,
        "cache_verification_receipt_sha256": arguments.cache_verification,
        "esm_embedding_manifest_sha256": arguments.esm_manifest,
        "partner_metadata_receipt_sha256": arguments.partner_metadata_receipt,
    }
    for field, path in paths.items():
        if sha256_file(path) != authority.get(field):
            raise ValueError(f"Protocol authority differs: {field}")

    split = json.loads(arguments.split_receipt.read_text(encoding="utf-8"))
    split_audit = split.get("audits", {}).get("antigen_fold", {})
    protocol_split = protocol["data"]["split"]
    if (
        split_audit.get("valid") is not True
        or split_audit.get("foldseek_score_rule") != "reciprocal_min"
        or int(split_audit.get("hard_edge_cross_role_count", -1)) != 0
        or int(split_audit.get("policy_component_count", -1))
        != int(protocol_split["policy_components"])
        or split.get("manifests", {}).get("antigen_fold", {}).get("sha256")
        != authority.get("split_manifest_sha256")
    ):
        raise ValueError("Split receipt does not prove the frozen antigen-fold policy")
    structural = json.loads(arguments.structural_audit.read_text(encoding="utf-8"))
    if (
        structural.get("valid") is not True
        or structural.get("thresholds", {}).get("foldseek_score_rule")
        != "reciprocal_min"
        or int(structural.get("violations", {}).get("cross_role_hard_edge_count", -1))
        != 0
    ):
        raise ValueError("Independent Foldseek structural audit did not pass")
    usalign = json.loads(arguments.usalign_receipt.read_text(encoding="utf-8"))
    if (
        usalign.get("valid") is not True
        or usalign.get("tm_score_rule") != "reciprocal_min"
        or int(usalign.get("sampled_cross_role_pair_count", -1)) != 103
        or int(usalign.get("cross_role_usalign_hard_edge_count", -1)) != 0
        or usalign.get("table_sha256") != authority.get("usalign_table_sha256")
    ):
        raise ValueError("US-align boundary audit did not pass")
    cache = json.loads(arguments.cache_verification.read_text(encoding="utf-8"))
    if cache.get("task_labels_read") is not False or not all(
        cache.get("checks", {}).values()
    ):
        raise ValueError("SAbDab2 cache verification did not pass")
    if cache.get("authority", {}).get("structure_manifest_sha256") != authority.get(
        "structure_manifest_sha256"
    ) or cache.get("authority", {}).get("contact_manifest_sha256") != authority.get(
        "contact_manifest_sha256"
    ):
        raise ValueError("Cache verification authorities differ")

    panel = protocol["data"]["reciprocal_partner_panel"]
    for role, path in (
        ("train", arguments.training_panel),
        ("validation", arguments.validation_panel),
        ("test", arguments.test_panel),
    ):
        if sha256_file(path) != panel.get(f"{role}_sha256"):
            raise ValueError(f"Frozen {role} panel digest differs")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("format_version") != panel.get("format_version")
            or payload.get("role") != role
            or int(payload.get("panel_size_per_direction", -1))
            != int(panel["size_per_query"])
            or payload.get("authority", {}).get("task_label_columns_read") is not False
        ):
            raise ValueError(f"Frozen {role} panel is ineligible")
    boundary = protocol.get("evidence_boundary_at_freeze", {})
    for key in (
        "sabdab2_antigen_fold_fitting_started",
        "sabdab2_antigen_fold_test_labels_accessed",
        "sabdab2_antigen_fold_test_predictions_accessed",
        "external_test_predictions_accessed",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"Protocol is not eligible for freeze: {key}")
    if protocol.get("training_objective", {}).get("specificity_score_source") != (
        "interaction_only"
    ):
        raise ValueError("Specificity selection must exclude unary propensities")
    if protocol.get("model_selection", {}).get("seeds") != [17, 42, 73, 101, 137]:
        raise ValueError("The exact five-seed set is not frozen")
    expected_matrix = [
        "sequence_dense",
        "spatial_topology",
        "distance",
        "distance_direction",
        "full_pose",
        "frozen_frame_ipa",
        "framecontact",
        "framecontact_interleaved",
    ]
    if protocol.get("development_matrix") != expected_matrix:
        raise ValueError("Development matrix differs from the executable matrix")

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(canonical_bytes(protocol)).hexdigest(),
        "parent_protocol_sha256": parent["sha256"],
        "parent_receipt_sha256": parent["receipt_sha256"],
        "authority_sha256": authority,
        "split_and_usalign_validated": True,
        "all_cache_artifacts_validated": True,
        "reciprocal_panels_validated": True,
        "interaction_only_specificity_validated": True,
        "sabdab2_pretraining_boundary_validated": True,
        "test_access_boundary_validated": True,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
