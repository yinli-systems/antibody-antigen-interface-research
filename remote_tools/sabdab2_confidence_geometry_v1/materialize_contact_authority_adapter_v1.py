#!/usr/bin/env python3
"""Rebind unchanged development contact labels to a new structure authority."""

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


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88"
)
EXPECTED_SOURCE_RECEIPT_SHA256 = (
    "426975feebdb5294b0fdbc45d31422fb3adbadc4d97b032958a38d19b050f1d9"
)
EXPECTED_STRUCTURE_MANIFEST_SHA256 = (
    "1c49c5d49742bd7f1a2dba5c52539750cbd90ef03f0b4778492abe0955e496e5"
)
EXPECTED_STRUCTURE_RECEIPT_SHA256 = (
    "c2f3129d1b59a96937e7b2094dfde149211451b58da92748e7359d9d5ccf0ab3"
)
EXPECTED_ROLES = {"train": 2565, "validation": 549}


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


def verify_contact_artifact(
    path: Path,
    record: dict[str, Any],
    *,
    antibody_length: int,
    antigen_length: int,
) -> None:
    if sha256_file(path) != record["artifact_sha256"]:
        raise ValueError(f"contact artifact SHA-256 differs: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "format_version",
            "instance_id",
            "antibody_length",
            "antigen_length",
            "candidate_pair_count",
            "positive_flat_indices",
        }:
            raise ValueError(f"contact artifact fields differ: {path}")
        instance_id = str(scalar(payload["instance_id"]))
        ab_length = int(scalar(payload["antibody_length"]))
        ag_length = int(scalar(payload["antigen_length"]))
        candidate_count = int(scalar(payload["candidate_pair_count"]))
        positive = payload["positive_flat_indices"]
    if (
        instance_id != record["instance_id"]
        or ab_length != antibody_length
        or ag_length != antigen_length
        or ab_length != int(record["antibody_length"])
        or ag_length != int(record["antigen_length"])
        or candidate_count != antibody_length * antigen_length
        or candidate_count != int(record["candidate_pair_count"])
        or positive.ndim != 1
        or positive.dtype.kind not in "iu"
        or int(positive.size) != int(record["positive_pair_count"])
        or (positive.size and (int(positive.min()) < 0 or int(positive.max()) >= candidate_count))
    ):
        raise ValueError(f"contact artifact content differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contact-root", type=Path, required=True)
    parser.add_argument("--target-structure-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    source_manifest_path = args.source_contact_root / "CONTACT_CACHE_MANIFEST.json"
    source_receipt_path = args.source_contact_root / "CONTACT_CACHE_RECEIPT.json"
    structure_manifest_path = args.target_structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    structure_receipt_path = args.target_structure_root / "STRUCTURE_CACHE_RECEIPT.json"
    expected = {
        source_manifest_path: EXPECTED_SOURCE_MANIFEST_SHA256,
        source_receipt_path: EXPECTED_SOURCE_RECEIPT_SHA256,
        structure_manifest_path: EXPECTED_STRUCTURE_MANIFEST_SHA256,
        structure_receipt_path: EXPECTED_STRUCTURE_RECEIPT_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"input authority differs: {path}")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    structure_manifest = json.loads(structure_manifest_path.read_text(encoding="utf-8"))
    structure_receipt = json.loads(structure_receipt_path.read_text(encoding="utf-8"))
    if (
        source_receipt.get("manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
        or source_receipt.get("pair_labels_regenerated") is not False
        or source_receipt.get("pair_label_artifacts_modified") is not False
        or source_receipt.get("test_pair_artifacts_read") is not False
        or source_receipt.get("temporal_holdout_access") is not False
        or structure_manifest.get("counts", {}).get("roles") != EXPECTED_ROLES
        or len(structure_manifest.get("records", [])) != 3114
        or structure_receipt.get("manifest_sha256")
        != EXPECTED_STRUCTURE_MANIFEST_SHA256
        or structure_receipt.get("task_labels_read") is not False
        or structure_receipt.get("test_or_temporal_loaded") is not False
    ):
        raise ValueError("input authority content differs")

    source_records = source_manifest.get("records", [])
    structure_records = structure_manifest.get("records", [])
    if len(source_records) != 3114 or len(structure_records) != 3114:
        raise ValueError("development cohort count differs")
    structures = {record["instance_id"]: record for record in structure_records}
    if len(structures) != 3114 or {r["instance_id"] for r in source_records} != set(structures):
        raise ValueError("contact and structure instance identities differ")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        for index, record in enumerate(source_records):
            structure = structures[record["instance_id"]]
            antibody_length = int(structure["antibody_length"])
            antigen_length = int(structure["antigen_length"])
            source_artifact = safe_artifact(args.source_contact_root, record["artifact_path"])
            verify_contact_artifact(
                source_artifact,
                record,
                antibody_length=antibody_length,
                antigen_length=antigen_length,
            )
            target_artifact = temporary_root / record["artifact_path"]
            target_artifact.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_artifact, target_artifact)
            if (
                target_artifact.stat().st_ino != source_artifact.stat().st_ino
                or sha256_file(target_artifact) != record["artifact_sha256"]
            ):
                raise ValueError(f"hard-link verification failed: {target_artifact}")
            if (index + 1) % 250 == 0:
                print(json.dumps({"verified_records": index + 1}, sort_keys=True), flush=True)

        authority = dict(source_manifest["authority"])
        authority.update(
            {
                "adapter_type": (
                    "development_only_pair_label_rebind_without_artifact_change_v2"
                ),
                "source_contact_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
                "source_contact_receipt_sha256": EXPECTED_SOURCE_RECEIPT_SHA256,
                "source_contact_structure_manifest_sha256": source_manifest[
                    "authority"
                ]["structure_manifest_sha256"],
                "ultimate_source_contact_manifest_sha256": source_manifest[
                    "authority"
                ].get("source_contact_manifest_sha256"),
                "structure_manifest_sha256": EXPECTED_STRUCTURE_MANIFEST_SHA256,
                "target_predicted_structure_manifest_sha256": (
                    EXPECTED_STRUCTURE_MANIFEST_SHA256
                ),
                "pair_label_artifacts_hard_linked": True,
                "pair_label_artifacts_rehashed": True,
                "pair_label_artifacts_reused_byte_identically": True,
                "residue_axis_lengths_verified": True,
                "selection_or_tuning": False,
                "test_pair_artifacts_materialized": False,
                "test_pair_artifacts_read": False,
                "temporal_holdout_access": False,
            }
        )
        manifest = {
            "format_version": "sabdab2_framecontact_cohort_cache_v2",
            "authority": authority,
            "counts": source_manifest["counts"],
            "records": source_records,
        }
        manifest_path = temporary_root / "CONTACT_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_bytes(manifest))
        manifest_sha = sha256_file(manifest_path)
        receipt = {
            "format_version": "sabdab2_contact_authority_adapter_receipt_v2",
            "status": "PASS_PROVENANCE_PRESERVING_CONTACT_AUTHORITY_ADAPTER",
            "manifest_sha256": manifest_sha,
            "source_contact_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "source_contact_receipt_sha256": EXPECTED_SOURCE_RECEIPT_SHA256,
            "predicted_structure_manifest_sha256": EXPECTED_STRUCTURE_MANIFEST_SHA256,
            "record_count": 3114,
            "role_counts": EXPECTED_ROLES,
            "all_pair_artifact_sha256_verified": True,
            "all_pair_artifacts_hard_linked": True,
            "all_residue_axis_lengths_verified": True,
            "pair_label_artifacts_modified": False,
            "pair_labels_regenerated": False,
            "selection_or_tuning": False,
            "test_pair_artifacts_materialized": False,
            "test_pair_artifacts_read": False,
            "temporal_holdout_access": False,
            "claim_boundary": (
                "Metadata-only development authority adapter for unchanged pair-label "
                "artifacts; no model, label, split, optimization, or result change."
            ),
        }
        receipt_path = temporary_root / "CONTACT_CACHE_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        os.replace(temporary_root, args.output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": receipt["status"],
                "manifest_sha256": manifest_sha,
                "receipt_sha256": sha256_file(
                    args.output_root / "CONTACT_CACHE_RECEIPT.json"
                ),
                "record_count": 3114,
                "pair_labels_regenerated": False,
                "test_pair_artifacts_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
