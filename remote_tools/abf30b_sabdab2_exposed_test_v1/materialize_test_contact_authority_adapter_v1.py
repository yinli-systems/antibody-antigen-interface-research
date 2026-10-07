#!/usr/bin/env python3
"""Rebind unchanged exposed-test contact artifacts to predicted geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


SOURCE_CONTACT_MANIFEST_SHA256 = "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
STRUCTURE_MANIFEST_SHA256 = "b13ecb8a3b184f64a0ff509ddefea41a6aa7da226f710d3ddf79c55c84eef4cb"
STRUCTURE_RECEIPT_SHA256 = "e043be4d266ea72833b9c7fc7fa7d19d1f22f46a97555d5bb221416f145ecb69"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def safe_artifact(root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {raw_path!r}")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"contact artifact is missing or symlinked: {path}")
    return path


def scalar(array: np.ndarray) -> Any:
    if array.shape != ():
        raise ValueError("expected scalar NPZ field")
    return array.item()


def verify_artifact(path: Path, record: dict[str, Any], ab_length: int, ag_length: int) -> None:
    if sha256_file(path) != record["artifact_sha256"]:
        raise ValueError(f"contact artifact SHA differs: {path}")
    with np.load(path, allow_pickle=False) as payload:
        expected_fields = {
            "format_version", "instance_id", "antibody_length", "antigen_length",
            "candidate_pair_count", "positive_flat_indices",
        }
        if set(payload.files) != expected_fields:
            raise ValueError(f"contact artifact fields differ: {path}")
        instance_id = str(scalar(payload["instance_id"]))
        observed_ab = int(scalar(payload["antibody_length"]))
        observed_ag = int(scalar(payload["antigen_length"]))
        candidate_count = int(scalar(payload["candidate_pair_count"]))
        positive = payload["positive_flat_indices"]
    if (
        instance_id != record["instance_id"]
        or observed_ab != ab_length
        or observed_ag != ag_length
        or observed_ab != int(record["antibody_length"])
        or observed_ag != int(record["antigen_length"])
        or candidate_count != ab_length * ag_length
        or candidate_count != int(record["candidate_pair_count"])
        or positive.ndim != 1
        or positive.dtype.kind not in "iu"
        or int(positive.size) != int(record["positive_pair_count"])
        or (positive.size and (int(positive.min()) < 0 or int(positive.max()) >= candidate_count))
    ):
        raise ValueError(f"contact artifact content differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--recovery-protocol-sha256", required=True)
    parser.add_argument("--source-contact-root", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.recovery_protocol) != args.recovery_protocol_sha256:
        raise ValueError("recovery protocol SHA differs")
    recovery = json.loads(args.recovery_protocol.read_text(encoding="utf-8"))
    if (
        recovery.get("status") not in {
            "FROZEN_AFTER_CONTACT_AUTHORITY_FAILURE_FOR_SAME_OPENING",
            "FROZEN_AFTER_STRUCTURE_RECEIPT_FIELD_ALIAS_FAILURE_FOR_SAME_OPENING",
        }
        or recovery.get("same_opening_retry") is not True
        or recovery.get("scientific_protocol_changed") is not False
        or recovery.get("labels_regenerated") is not False
        or Path(recovery["adapter_output_root"]).resolve() != args.output_root.resolve()
    ):
        raise ValueError("adapter recovery authority differs")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    source_manifest_path = args.source_contact_root / "CONTACT_CACHE_MANIFEST.json"
    structure_manifest_path = args.structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    structure_receipt_path = args.structure_root / "STRUCTURE_CACHE_RECEIPT.json"
    for path, expected in (
        (source_manifest_path, SOURCE_CONTACT_MANIFEST_SHA256),
        (structure_manifest_path, STRUCTURE_MANIFEST_SHA256),
        (structure_receipt_path, STRUCTURE_RECEIPT_SHA256),
    ):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"input authority differs: {path}")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    structure = json.loads(structure_manifest_path.read_text(encoding="utf-8"))
    structure_receipt = json.loads(structure_receipt_path.read_text(encoding="utf-8"))
    if (
        source.get("format_version") != "sabdab2_framecontact_cohort_cache_v2"
        or source.get("counts", {}).get("records") != 3663
        or structure.get("format_version") != "sabdab2_intra_chain_structure_cohort_cache_v2"
        or structure.get("counts", {}).get("roles") != {"test": 549}
        or len(structure.get("records", [])) != 549
        or structure_receipt.get("manifest_sha256") != STRUCTURE_MANIFEST_SHA256
        or structure_receipt.get("task_labels_read") is not False
        or structure_receipt.get("temporal_holdout_access") is not False
        or structure.get("authority", {}).get("selection_or_tuning") is not False
    ):
        raise ValueError("input authority content differs")
    source_records = {row["instance_id"]: row for row in source["records"]}
    structure_records = structure["records"]
    if len(source_records) != 3663 or len({row["instance_id"] for row in structure_records}) != 549:
        raise ValueError("instance identity axis differs")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent))
    try:
        selected: list[dict[str, Any]] = []
        inodes: list[tuple[int, int]] = []
        for index, structure_row in enumerate(structure_records):
            identifier = structure_row["instance_id"]
            contact_row = source_records.get(identifier)
            if contact_row is None:
                raise ValueError(f"missing source contact record: {identifier}")
            ab_length = int(structure_row["antibody_length"])
            ag_length = int(structure_row["antigen_length"])
            source_artifact = safe_artifact(args.source_contact_root, contact_row["artifact_path"])
            verify_artifact(source_artifact, contact_row, ab_length, ag_length)
            target_artifact = temporary_root / contact_row["artifact_path"]
            target_artifact.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_artifact, target_artifact)
            source_stat = source_artifact.stat()
            target_stat = target_artifact.stat()
            if (
                (source_stat.st_dev, source_stat.st_ino) != (target_stat.st_dev, target_stat.st_ino)
                or sha256_file(target_artifact) != contact_row["artifact_sha256"]
            ):
                raise ValueError(f"hard-link verification failed: {identifier}")
            inodes.append((target_stat.st_dev, target_stat.st_ino))
            selected.append(dict(contact_row))
            if (index + 1) % 100 == 0:
                print(json.dumps({"verified_records": index + 1}, sort_keys=True), flush=True)
        if len(selected) != 549 or len(set(inodes)) != 549:
            raise ValueError("adapted artifact count or one-to-one mapping differs")

        original_authority = dict(source["authority"])
        manifest = {
            "format_version": source["format_version"],
            "authority": {
                **original_authority,
                "structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
                "adapter_type": "test_only_pair_label_rebind_without_artifact_change",
                "source_contact_manifest_sha256": SOURCE_CONTACT_MANIFEST_SHA256,
                "source_contact_structure_manifest_sha256": original_authority["structure_manifest_sha256"],
                "target_predicted_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
                "pair_label_artifacts_reused_byte_identically": True,
                "pair_label_artifacts_rehashed": True,
                "pair_label_artifacts_hard_linked": True,
                "residue_axis_lengths_verified": True,
                "test_membership_source": "authenticated predicted structure manifest test role",
                "labels_regenerated": False,
                "selection_or_tuning": False,
                "experimental_bound_pair_labels_used_for_measurement": True,
                "experimental_bound_coordinates_used_as_model_input": False,
                "same_opening_retry": True,
                "cumulative_test_access_ordinal": 3,
                "temporal_holdout_access": False,
            },
            "counts": {
                "records": 549,
                "candidate_pairs": sum(int(row["candidate_pair_count"]) for row in selected),
                "positive_pairs": sum(int(row["positive_pair_count"]) for row in selected),
                "by_role": {"test": 549},
                "skipped_non_test_records": 3663 - 549,
            },
            "records": selected,
        }
        manifest_path = temporary_root / "CONTACT_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_bytes(manifest))
        manifest_sha = sha256_file(manifest_path)
        receipt = {
            "format_version": "abf30b_sabdab2_exposed_test_contact_authority_adapter_receipt_v1",
            "status": "PASS_PROVENANCE_PRESERVING_TEST_CONTACT_AUTHORITY_ADAPTER",
            "manifest_sha256": manifest_sha,
            "recovery_protocol_sha256": args.recovery_protocol_sha256,
            "source_contact_manifest_sha256": SOURCE_CONTACT_MANIFEST_SHA256,
            "predicted_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
            "record_count": 549,
            "role_counts": {"test": 549},
            "all_pair_artifact_sha256_verified": True,
            "all_pair_artifacts_hard_linked": True,
            "all_residue_axis_lengths_verified": True,
            "pair_label_artifacts_modified": False,
            "pair_labels_regenerated": False,
            "selection_or_tuning": False,
            "same_opening_retry": True,
            "cumulative_test_access_ordinal": 3,
            "temporal_holdout_access": False,
            "claim_boundary": "Metadata-only test authority adapter for unchanged pair-label artifacts; no model, label, cohort, calibration, routing, or result change.",
        }
        receipt_path = temporary_root / "CONTACT_CACHE_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        os.replace(temporary_root, args.output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(json.dumps({
        "status": receipt["status"],
        "manifest_sha256": manifest_sha,
        "receipt_sha256": sha256_file(args.output_root / "CONTACT_CACHE_RECEIPT.json"),
        "record_count": 549,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
