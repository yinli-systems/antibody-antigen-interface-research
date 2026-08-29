#!/usr/bin/env python3
"""Freeze corrected distance-label authorities before any recovery training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


BASELINE_SOURCE_SHA256 = "a320a5f60794ea386d355e8ab5ec895130d3b714596bc817e303eee589dbc7e8"
TOPK_SOURCE_SHA256 = "885449525b7f41b3b00511245ca6c0b0859336db959becc7680ad9cb6e872332"
INVALID_FAILURE_SHA256 = "2e181e95d80268ee79c734c6377e7f75aa56c987eb56e100bb34b39672278728"
INVALID_FAILURE_RECEIPT_SHA256 = "c45f01585f7eaa6d5e1220b7221aef4a2bd703ba7d7b3709ca3cd5487c11a1bc"
FREEZER_FAILURE_SHA256 = "1a50babe958def5a14adbe469df6a3330a2ab01536537a5e4ebc5a1af732ab46"
FREEZER_FAILURE_RECEIPT_SHA256 = "8890843d4199cab4357aa5e91f39d3159cc48a0e80080a07bcd67320b1f677c1"
BOUND_STRUCTURE_MANIFEST_SHA256 = "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0"
MODEL_INPUT_STRUCTURE_MANIFEST_SHA256 = "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
CONTACT_MANIFEST_SHA256 = "d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88"
CONTACT_RECEIPT_SHA256 = "426975feebdb5294b0fdbc45d31422fb3adbadc4d97b032958a38d19b050f1d9"
SPLIT_SHA256 = "ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: object) -> str:
    path.write_bytes(canonical_bytes(payload))
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--topk-source-root", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest", type=Path, required=True)
    parser.add_argument("--topk-source-manifest", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--training-launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    return parser.parse_args()


def protocol_receipt(
    path: Path,
    protocol_sha256: str,
    *,
    arm: str,
    distance_manifest_sha256: str,
) -> str:
    return write_json(
        path,
        {
            "format_version": "sabdab2_predicted_geometry_distance_label_protocol_receipt_v2",
            "arm": arm,
            "protocol_file_sha256": protocol_sha256,
            "structure_manifest_sha256": MODEL_INPUT_STRUCTURE_MANIFEST_SHA256,
            "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
            "distance_label_manifest_sha256": distance_manifest_sha256,
            "development_split_sha256": SPLIT_SHA256,
            "frozen_before_recovery_training": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    )


def main() -> None:
    args = parse_args()
    authority = args.campaign_root / "authority" / "distance_label_recovery_v2"
    if authority.exists() or authority.is_symlink():
        raise FileExistsError(authority)
    distance_root = args.campaign_root / "data" / "bound_distance_labels_v1"
    distance_manifest_path = distance_root / "DISTANCE_LABEL_CACHE_MANIFEST.json"
    distance_receipt_path = distance_root / "DISTANCE_LABEL_CACHE_RECEIPT.json"
    distance_manifest_sha256 = sha256_file(distance_manifest_path)
    distance_receipt_sha256 = sha256_file(distance_receipt_path)
    distance_manifest = json.loads(distance_manifest_path.read_text(encoding="utf-8"))
    distance_receipt = json.loads(distance_receipt_path.read_text(encoding="utf-8"))
    distance_authority = distance_manifest.get("authority", {})
    if (
        distance_manifest.get("format_version")
        != "sabdab2_cross_chain_distance_label_cache_v1"
        or distance_receipt.get("status")
        != "PASS_DEVELOPMENT_ONLY_BOUND_DISTANCE_LABEL_CACHE"
        or distance_receipt.get("manifest_sha256") != distance_manifest_sha256
        or distance_manifest.get("counts", {}).get("records") != 3114
        or distance_manifest.get("counts", {}).get("by_role")
        != {"train": 2565, "validation": 549}
        or distance_authority.get("source_bound_structure_manifest_sha256")
        != BOUND_STRUCTURE_MANIFEST_SHA256
        or distance_authority.get("model_input_structure_manifest_sha256")
        != MODEL_INPUT_STRUCTURE_MANIFEST_SHA256
        or distance_authority.get("development_split_sha256") != SPLIT_SHA256
        or distance_authority.get("distance_boundaries_angstrom")
        != [4.0, 8.0, 16.0, 32.0]
        or distance_authority.get(
            "bound_cross_partner_distances_used_as_auxiliary_training_labels"
        )
        is not True
        or distance_authority.get("bound_coordinates_exposed_to_model_input")
        is not False
        or distance_authority.get("cross_partner_geometry_exposed_to_model_input")
        is not False
        or distance_authority.get("task_labels_read") is not False
        or distance_authority.get("test_structure_artifacts_read") is not False
        or distance_authority.get("temporal_holdout_access") is not False
        or distance_receipt.get("all_source_artifact_sha256_verified") is not True
        or distance_receipt.get("all_sequence_and_length_axes_verified") is not True
        or distance_receipt.get("test_structure_artifacts_read") is not False
    ):
        raise ValueError("distance-label cache authority differs")
    for record in distance_manifest["records"]:
        artifact = distance_root / str(record["artifact_path"])
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or artifact.stat().st_size != int(record["artifact_size_bytes"])
            or sha256_file(artifact) != record["artifact_sha256"]
        ):
            raise ValueError(f"distance-label artifact differs: {record['instance_id']}")

    failure = args.campaign_root / "FAIL_CLOSED_INVALID_INDEPENDENT_COORDINATE_DISTANCE_SUPERVISION_1483894.json"
    failure_receipt = failure.with_name(failure.stem + "_RECEIPT.json")
    if (
        sha256_file(failure) != INVALID_FAILURE_SHA256
        or sha256_file(failure_receipt) != INVALID_FAILURE_RECEIPT_SHA256
    ):
        raise ValueError("invalid-run failure evidence differs")
    failure_payload = json.loads(failure.read_text(encoding="utf-8"))
    failure_boundary = failure_payload.get("claim_boundary", {})
    if (
        failure_payload.get("status")
        != "FAIL_CLOSED_INVALID_INDEPENDENT_COORDINATE_DISTANCE_SUPERVISION"
        or failure_boundary.get("test_loaded") is not False
        or failure_boundary.get("temporal_holdout_loaded") is not False
        or failure_boundary.get("scientific_result_authorized") is not False
    ):
        raise ValueError("invalid-run failure evidence content differs")

    freezer_failure = args.campaign_root / "FAIL_CLOSED_FREEZER_SCHEMA_DIFFERENCE_1484895.json"
    freezer_failure_receipt = freezer_failure.with_name(
        freezer_failure.stem + "_RECEIPT.json"
    )
    if (
        sha256_file(freezer_failure) != FREEZER_FAILURE_SHA256
        or sha256_file(freezer_failure_receipt) != FREEZER_FAILURE_RECEIPT_SHA256
    ):
        raise ValueError("freezer-failure evidence differs")
    freezer_failure_payload = json.loads(
        freezer_failure.read_text(encoding="utf-8")
    )
    if (
        freezer_failure_payload.get("status")
        != "FAIL_CLOSED_FREEZER_SCHEMA_DIFFERENCE"
        or freezer_failure_payload.get("training_started") is not False
        or freezer_failure_payload.get("test_loaded") is not False
        or freezer_failure_payload.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("freezer-failure evidence content differs")

    if sha256_file(args.baseline_source_manifest) != BASELINE_SOURCE_SHA256:
        raise ValueError("baseline corrected source manifest differs")
    if sha256_file(args.topk_source_manifest) != TOPK_SOURCE_SHA256:
        raise ValueError("TopK2 corrected source manifest differs")
    baseline_source = json.loads(args.baseline_source_manifest.read_text(encoding="utf-8"))
    topk_source = json.loads(args.topk_source_manifest.read_text(encoding="utf-8"))
    if Path(baseline_source["source_root"]).resolve() != args.baseline_source_root.resolve():
        raise ValueError("baseline corrected source root differs")
    if Path(topk_source["source_root"]).resolve() != args.topk_source_root.resolve():
        raise ValueError("TopK2 corrected source root differs")

    previous = args.campaign_root / "authority" / "contact_rebind_recovery_v1"
    baseline_template_path = previous / "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_CONTACT_REBIND_RECOVERY_V1.json"
    topk_template_path = previous / "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_CONTACT_REBIND_RECOVERY_V1.json"
    execution = {
        "distance_label_materializer_sha256": sha256_file(args.materializer),
        "distance_label_recovery_freezer_sha256": sha256_file(Path(__file__)),
        "training_launcher_sha256": sha256_file(args.training_launcher),
        "analyzer_sha256": sha256_file(args.analyzer),
        "invalid_array_job_id": "1483894",
        "invalid_failure_sha256": INVALID_FAILURE_SHA256,
        "invalid_failure_receipt_sha256": INVALID_FAILURE_RECEIPT_SHA256,
        "freezer_failure_job_id": "1484895",
        "freezer_failure_sha256": FREEZER_FAILURE_SHA256,
        "freezer_failure_receipt_sha256": FREEZER_FAILURE_RECEIPT_SHA256,
        "distance_label_manifest_sha256": distance_manifest_sha256,
        "distance_label_receipt_sha256": distance_receipt_sha256,
    }
    authority.mkdir(parents=True)
    baseline_source_copy = authority / "SOURCE_MANIFEST_BASELINE_DISTANCE_LABELS_V4.json"
    topk_source_copy = authority / "SOURCE_MANIFEST_EDGE_TOPK2_DISTANCE_LABELS_V2.json"
    shutil.copyfile(args.baseline_source_manifest, baseline_source_copy)
    shutil.copyfile(args.topk_source_manifest, topk_source_copy)

    protocols: dict[str, tuple[dict, Path, str, str]] = {}
    for arm, template_path, source_sha256, source_key in (
        ("baseline", baseline_template_path, BASELINE_SOURCE_SHA256, "source_authority"),
        ("edge_topk2_supervised", topk_template_path, TOPK_SOURCE_SHA256, "execution_sources"),
    ):
        protocol = json.loads(template_path.read_text(encoding="utf-8"))
        protocol["schema_version"] += "_distance_label_recovery_v2"
        protocol["protocol_status"] = "frozen_before_distance_label_recovery_training_v2"
        geometry_input = protocol["data"]["geometry_input"]
        geometry_input.pop("experimental_bound_complex_coordinates_used", None)
        geometry_input.pop("cross_partner_coordinates_or_distances_used", None)
        geometry_input["experimental_bound_complex_coordinates_used_as_model_input"] = False
        geometry_input["cross_partner_coordinates_or_distances_used_as_model_input"] = False
        protocol["data"]["distance_label_manifest_sha256"] = distance_manifest_sha256
        protocol["data"]["distance_label_receipt_sha256"] = distance_receipt_sha256
        protocol["data"]["distance_label_source_bound_structure_manifest_sha256"] = BOUND_STRUCTURE_MANIFEST_SHA256
        protocol["training_objective"]["loss_weights"]["distance"] = 0.1
        protocol["training_objective"]["loss_weights"]["contrastive"] = 0.0
        distance_supervision = protocol["training_objective"].setdefault(
            "distance_supervision", {}
        )
        distance_supervision.update(
            {
                "source": "authenticated experimental bound-complex C-alpha distances",
                "materialization": "development-only five-bin labels and frame-valid masks",
                "distance_label_manifest_sha256": distance_manifest_sha256,
                "bound_coordinates_available_to_model": False,
                "labels_available_only_for_auxiliary_training_loss": True,
            }
        )
        protocol["claim_boundary"].update(
            {
                "experimental_bound_geometry_used_for_auxiliary_training_supervision": True,
                "experimental_bound_geometry_used_as_model_input": False,
                "experimental_structure_required_at_inference": False,
                "implementation_corrected_after_invalid_distance_target_run": True,
                "invalid_run_reused": False,
                "scientific_protocol_changed_by_recovery": False,
            }
        )
        protocol["distance_label_recovery_v2"] = {
            **execution,
            "source_manifest_sha256": source_sha256,
            "model_architecture_changed": False,
            "loss_weights_changed": False,
            "distance_bin_definition_changed": False,
            "training_data_or_split_changed": False,
        }
        protocol[source_key] = {
            **protocol.get(source_key, {}),
            **execution,
            "source_manifest_sha256": source_sha256,
        }
        name = (
            "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_DISTANCE_LABEL_RECOVERY_V2.json"
            if arm == "baseline"
            else "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_DISTANCE_LABEL_RECOVERY_V2.json"
        )
        path = authority / name
        protocol_sha256 = write_json(path, protocol)
        receipt_path = path.with_name(path.stem + "_RECEIPT.json")
        receipt_sha256 = protocol_receipt(
            receipt_path,
            protocol_sha256,
            arm=arm,
            distance_manifest_sha256=distance_manifest_sha256,
        )
        protocols[arm] = (protocol, path, protocol_sha256, receipt_sha256)

    campaign = {
        "format_version": "sabdab2_predicted_geometry_distance_label_recovery_campaign_v2",
        "status": "FROZEN_BEFORE_DISTANCE_LABEL_RECOVERY_TRAINING_V2",
        "campaign_root": str(args.campaign_root),
        "invalid_array_job_id": "1483894",
        "invalid_failure_sha256": INVALID_FAILURE_SHA256,
        "invalid_failure_receipt_sha256": INVALID_FAILURE_RECEIPT_SHA256,
        "model_input_structure_manifest_sha256": MODEL_INPUT_STRUCTURE_MANIFEST_SHA256,
        "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
        "contact_receipt_sha256": CONTACT_RECEIPT_SHA256,
        "distance_label_cache_root": str(distance_root),
        "distance_label_manifest_sha256": distance_manifest_sha256,
        "distance_label_receipt_sha256": distance_receipt_sha256,
        "source_bound_structure_manifest_sha256": BOUND_STRUCTURE_MANIFEST_SHA256,
        "development_split_sha256": SPLIT_SHA256,
        "arms": {
            arm: {
                "protocol_sha256": values[2],
                "receipt_sha256": values[3],
                "source_manifest_sha256": (
                    BASELINE_SOURCE_SHA256 if arm == "baseline" else TOPK_SOURCE_SHA256
                ),
            }
            for arm, values in protocols.items()
        },
        "seeds": [17, 29, 43],
        "implementation_corrected": True,
        "scientific_protocol_changed": False,
        "model_architecture_changed": False,
        "loss_or_optimization_changed": False,
        "distance_bin_definition_changed": False,
        "pair_contact_labels_changed": False,
        "split_changed": False,
        "invalid_run_reused": False,
        "selection_or_tuning": False,
        "experimental_bound_geometry_used_as_model_input": False,
        "experimental_bound_geometry_used_for_auxiliary_training_supervision": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
        "execution_sources": execution,
    }
    campaign_path = authority / "CAMPAIGN_PREDICTED_GEOMETRY_DISTANCE_LABEL_RECOVERY_V2.json"
    campaign_sha256 = write_json(campaign_path, campaign)
    checksum_paths = [
        distance_manifest_path,
        distance_receipt_path,
        failure,
        failure_receipt,
        freezer_failure,
        freezer_failure_receipt,
        baseline_source_copy,
        topk_source_copy,
        protocols["baseline"][1],
        protocols["baseline"][1].with_name(protocols["baseline"][1].stem + "_RECEIPT.json"),
        protocols["edge_topk2_supervised"][1],
        protocols["edge_topk2_supervised"][1].with_name(protocols["edge_topk2_supervised"][1].stem + "_RECEIPT.json"),
        campaign_path,
    ]
    (authority / "AUTHORITY_SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "campaign_sha256": campaign_sha256,
                "distance_label_manifest_sha256": distance_manifest_sha256,
                "distance_label_receipt_sha256": distance_receipt_sha256,
                "baseline_protocol_sha256": protocols["baseline"][2],
                "topk_protocol_sha256": protocols["edge_topk2_supervised"][2],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
