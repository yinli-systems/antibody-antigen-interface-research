#!/usr/bin/env python3
"""Authorize one SABLE bound-label opening after all label-free gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_EXTERNAL_PLAN_SHA256 = (
    "d97f0b94863fe9227d80c5675e944091809245b8fab22a89cda81148a8de172a"
)
EXPECTED_SCREEN_RESULT_SHA256 = (
    "2c6f6ec9f9b8ff6fd11112a9131fdd9fa024591bc99df36e42caecd8819eca00"
)
EXPECTED_SCREEN_RECEIPT_SHA256 = (
    "5aee4ff1c98a9936674d68aae74d28e90f6d9ba357464f7776f72c7d5cbd73d6"
)
EXPECTED_ENTRIES = 97
EXPECTED_PDB_IDS = 88
EXPECTED_ENTITIES = 179
REQUIRED_MEASUREMENT_SOURCE_FILES = frozenset(
    {
        "SABLE_V0_2_0_EXTERNAL_EXECUTION_PLAN_V1.json",
        "download_sable_v0_2_0_bound_structures_once_v1.py",
        "materialize_sable_v0_2_0_bound_labels_once_v1.py",
        "materialize_sable_v0_2_0_bound_labels_once_v1.sbatch",
        "evaluate_sable_v0_2_0_component_once_v1.py",
        "evaluate_sable_v0_2_0_component_once_v1.sbatch",
        "analyze_sable_v0_2_0_external_once_v1.py",
        "analyze_sable_v0_2_0_external_once_v1.sbatch",
    }
)


class OpeningError(ValueError):
    """Raised when a required label-free authority differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> str:
    data = canonical_bytes(payload)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OpeningError(f"JSON root is not an object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-plan", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--screen-receipt", type=Path, required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--checkpoint-binding", type=Path, required=True)
    parser.add_argument("--checkpoint-binding-receipt", type=Path, required=True)
    parser.add_argument("--measurement-source-manifest", type=Path, required=True)
    parser.add_argument("--measurement-source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser.parse_args()


def verify_structure_cache(root: Path) -> tuple[str, str]:
    manifest_path = root / "STRUCTURE_CACHE_MANIFEST.json"
    receipt_path = root / "STRUCTURE_CACHE_RECEIPT.json"
    manifest = load_json(manifest_path)
    receipt = load_json(receipt_path)
    manifest_sha256 = sha256_file(manifest_path)
    if (
        manifest.get("status") != "PASS_LABEL_FREE_SEQUENCE_PREDICTED_STRUCTURE_CACHE"
        or receipt.get("status") != manifest.get("status")
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("record_count") != EXPECTED_ENTRIES
        or receipt.get("unique_pdb_id_count") != EXPECTED_PDB_IDS
        or receipt.get("prediction_entity_count") != EXPECTED_ENTITIES
        or receipt.get("all_batch_sha256_verified") is not True
        or receipt.get("all_artifact_sha256_verified") is not True
        or receipt.get("experimental_bound_complex_coordinates_read") is not False
        or receipt.get("cross_partner_coordinates_or_distances_used") is not False
        or receipt.get("task_labels_read") is not False
        or manifest.get("authority", {}).get(
            "experimental_bound_complex_coordinates_read"
        )
        is not False
        or manifest.get("authority", {}).get("task_labels_read") is not False
        or manifest.get("authority", {}).get("old_sabdab2_test_loaded") is not False
        or manifest.get("authority", {}).get("temporal_holdout_loaded") is not False
    ):
        raise OpeningError("predicted structure cache authority differs")
    records = manifest.get("records", [])
    if len(records) != EXPECTED_ENTRIES:
        raise OpeningError("predicted structure record count differs")
    for record in records:
        path = root / record["artifact_path"]
        if sha256_file(path) != record["artifact_sha256"]:
            raise OpeningError(f"predicted structure artifact SHA differs: {path}")
    return manifest_sha256, sha256_file(receipt_path)


def verify_embeddings(root: Path) -> str:
    manifest_path = root / "SABLE_ESM_EMBEDDING_MANIFEST.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "PASS_LABEL_FREE_ESM2_EMBEDDINGS"
        or manifest.get("entry_count") != EXPECTED_ENTRIES
        or manifest.get("unique_pdb_id_count") != EXPECTED_PDB_IDS
        or manifest.get("authority", {}).get("task_labels_read") is not False
        or manifest.get("authority", {}).get("bound_structures_read") is not False
        or manifest.get("authority", {}).get("old_sabdab2_test_loaded") is not False
        or manifest.get("authority", {}).get("temporal_holdout_loaded") is not False
        or manifest.get("authority", {}).get("selection_or_tuning") is not False
    ):
        raise OpeningError("ESM embedding authority differs")
    records = manifest.get("records", [])
    if len(records) != manifest.get("unique_sequence_count") or not records:
        raise OpeningError("ESM embedding record count differs")
    for record in records:
        path = root / record["artifact_path"]
        if sha256_file(path) != record["artifact_sha256"]:
            raise OpeningError(f"ESM embedding artifact SHA differs: {path}")
    return sha256_file(manifest_path)


def verify_checkpoint_binding(path: Path, receipt_path: Path) -> tuple[str, str]:
    binding = load_json(path)
    receipt = load_json(receipt_path)
    binding_sha256 = sha256_file(path)
    if (
        binding.get("status") != "PASS_EXACT_NO_DISTANCE_CHECKPOINT_BINDING"
        or receipt.get("status") != binding.get("status")
        or receipt.get("binding_sha256") != binding_sha256
        or receipt.get("all_six_checkpoint_sha256_verified") is not True
        or binding.get("distance_label_cache_supplied") is not False
        or binding.get("distance_labels_loaded") is not False
        or binding.get("experimental_bound_geometry_used_as_model_input") is not False
        or binding.get(
            "experimental_bound_geometry_used_for_auxiliary_training_supervision"
        )
        is not False
        or binding.get("experimental_structure_required_at_inference") is not False
        or binding.get("performance_threshold_applied") is not False
        or binding.get("task_labels_read") is not False
        or binding.get("sable_bound_structures_downloaded") is not False
        or binding.get("bound_label_opening_authorized") is not False
        or binding.get("old_sabdab2_test_loaded") is not False
        or binding.get("temporal_holdout_loaded") is not False
        or binding.get("selection_or_tuning") is not False
    ):
        raise OpeningError("checkpoint binding authority differs")
    return binding_sha256, sha256_file(receipt_path)


def verify_measurement_source(
    manifest_path: Path, receipt_path: Path
) -> tuple[str, str]:
    manifest = load_json(manifest_path)
    receipt = load_json(receipt_path)
    manifest_sha256 = sha256_file(manifest_path)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise OpeningError("measurement source records are malformed")
    relative_paths = {str(record.get("relative_path")) for record in records}
    if (
        manifest.get("status") != "FROZEN_MEASUREMENT_SOURCE_BEFORE_BOUND_LABEL_OPENING"
        or manifest.get("external_execution_plan_sha256")
        != EXPECTED_EXTERNAL_PLAN_SHA256
        or manifest.get("task_labels_read_at_freeze") is not False
        or manifest.get("bound_structures_downloaded_at_freeze") is not False
        or manifest.get("selection_or_tuning") is not False
        or manifest.get("old_sabdab2_test_loaded") is not False
        or manifest.get("temporal_holdout_loaded") is not False
        or relative_paths != REQUIRED_MEASUREMENT_SOURCE_FILES
        or receipt.get("status") != manifest.get("status")
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("all_source_sha256_verified_at_freeze") is not True
        or receipt.get("task_labels_read_at_freeze") is not False
    ):
        raise OpeningError("measurement source authority differs")
    source_root = Path(str(manifest.get("source_root", "")))
    if not source_root.is_absolute() or not source_root.is_dir():
        raise OpeningError("measurement source root differs")
    for record in records:
        relative_path = Path(str(record["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise OpeningError("measurement source path is not confined")
        path = source_root / relative_path
        if not path.is_file() or path.is_symlink():
            raise OpeningError(f"measurement source is missing or a symlink: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise OpeningError(f"measurement source SHA differs: {path}")
    return manifest_sha256, sha256_file(receipt_path)


def main() -> None:
    args = parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if sha256_file(args.external_plan) != EXPECTED_EXTERNAL_PLAN_SHA256:
        raise OpeningError("external execution plan SHA256 differs")
    if sha256_file(args.screen_result) != EXPECTED_SCREEN_RESULT_SHA256:
        raise OpeningError("label-free screen result SHA256 differs")
    if sha256_file(args.screen_receipt) != EXPECTED_SCREEN_RECEIPT_SHA256:
        raise OpeningError("label-free screen receipt SHA256 differs")

    plan = load_json(args.external_plan)
    screen = load_json(args.screen_result)
    screen_receipt = load_json(args.screen_receipt)
    if (
        plan.get("status") != "FROZEN_POST_GATE_BEFORE_BOUND_LABEL_OPENING"
        or plan.get("bound_label_opening_authorized") is not False
        or plan.get("task_labels_read_at_freeze") is not False
        or plan.get("opening", {}).get("maximum_label_openings") != 1
        or screen.get("status") != "PASS_LABEL_FREE_GATE"
        or screen_receipt.get("status") != screen.get("status")
        or screen_receipt.get("result_sha256") != EXPECTED_SCREEN_RESULT_SHA256
        or screen.get("task_labels_read") is not False
        or screen.get("sable_bound_structures_downloaded") is not False
        or screen.get("old_sabdab2_test_loaded") is not False
        or screen.get("temporal_holdout_loaded") is not False
        or screen.get("selection_or_tuning") is not False
    ):
        raise OpeningError("label-free cohort authority differs")

    structure_sha256, structure_receipt_sha256 = verify_structure_cache(
        args.structure_cache_root
    )
    embedding_sha256 = verify_embeddings(args.embedding_root)
    binding_sha256, binding_receipt_sha256 = verify_checkpoint_binding(
        args.checkpoint_binding, args.checkpoint_binding_receipt
    )
    measurement_source_sha256, measurement_source_receipt_sha256 = (
        verify_measurement_source(
            args.measurement_source_manifest, args.measurement_source_receipt
        )
    )
    payload = {
        "format_version": "sable_v0_2_0_bound_label_opening_authorization_v1",
        "status": "AUTHORIZED_ONE_TIME_BOUND_LABEL_OPENING",
        "external_execution_plan_sha256": EXPECTED_EXTERNAL_PLAN_SHA256,
        "label_free_screen_result_sha256": EXPECTED_SCREEN_RESULT_SHA256,
        "label_free_screen_receipt_sha256": EXPECTED_SCREEN_RECEIPT_SHA256,
        "predicted_structure_manifest_sha256": structure_sha256,
        "predicted_structure_receipt_sha256": structure_receipt_sha256,
        "esm_embedding_manifest_sha256": embedding_sha256,
        "checkpoint_binding_sha256": binding_sha256,
        "checkpoint_binding_receipt_sha256": binding_receipt_sha256,
        "measurement_source_manifest_sha256": measurement_source_sha256,
        "measurement_source_receipt_sha256": measurement_source_receipt_sha256,
        "eligible_entries": EXPECTED_ENTRIES,
        "eligible_unique_pdb_ids": EXPECTED_PDB_IDS,
        "maximum_label_openings": 1,
        "bound_structure_download_authorized": True,
        "bound_label_materialization_authorized": True,
        "one_time_evaluation_authorized": True,
        "feedback_to_model_or_protocol": False,
        "performance_threshold_applied": False,
        "task_labels_read_by_this_freezer": False,
        "sable_bound_structures_downloaded_by_this_freezer": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning": False,
    }
    output_sha256 = write_json(args.output, payload)
    receipt_payload = {
        "format_version": "sable_v0_2_0_bound_label_opening_authorization_receipt_v1",
        "status": payload["status"],
        "authorization_sha256": output_sha256,
        "all_label_free_gates_verified": True,
        "maximum_label_openings": 1,
        "task_labels_read_by_this_freezer": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha256 = write_json(args.receipt_output, receipt_payload)
    print(json.dumps({**receipt_payload, "receipt_sha256": receipt_sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
