#!/usr/bin/env python3
"""Bind unchanged development pair-label artifacts to predicted geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


SOURCE_CONTACT_MANIFEST_SHA256 = (
    "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contact-root", type=Path, required=True)
    parser.add_argument("--predicted-structure-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    source_manifest_path = args.source_contact_root / "CONTACT_CACHE_MANIFEST.json"
    structure_manifest_path = (
        args.predicted_structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    )
    if sha256_file(source_manifest_path) != SOURCE_CONTACT_MANIFEST_SHA256:
        raise ValueError("source contact manifest SHA-256 differs")

    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    structure = json.loads(structure_manifest_path.read_text(encoding="utf-8"))
    structure_sha256 = sha256_file(structure_manifest_path)
    if (
        source.get("format_version") != "sabdab2_framecontact_cohort_cache_v2"
        or structure.get("format_version")
        != "sabdab2_intra_chain_structure_cohort_cache_v2"
        or structure.get("counts", {}).get("records") != 3114
        or structure.get("counts", {}).get("roles")
        != {"train": 2565, "validation": 549}
        or structure.get("authority", {}).get("geometry_source")
        != "independent_sequence_predicted_af3"
        or structure.get("authority", {}).get(
            "experimental_bound_complex_coordinates_read"
        )
        is not False
        or structure.get("authority", {}).get("test_structures_resolved") is not False
        or structure.get("authority", {}).get("temporal_holdout_access") is not False
    ):
        raise ValueError("predicted structure authority differs")

    source_records = {record["instance_id"]: record for record in source["records"]}
    structure_records = structure["records"]
    structure_ids = [record["instance_id"] for record in structure_records]
    if len(structure_ids) != 3114 or len(set(structure_ids)) != 3114:
        raise ValueError("predicted structure instance axis differs")

    temporary_root = args.output_root.with_name(args.output_root.name + ".tmp")
    if temporary_root.exists() or temporary_root.is_symlink():
        raise FileExistsError(temporary_root)
    temporary_root.mkdir(parents=True)
    try:
        selected: list[dict[str, Any]] = []
        role_counts = {"train": 0, "validation": 0}
        artifact_inodes: list[tuple[int, int]] = []
        for structure_record in structure_records:
            identifier = structure_record["instance_id"]
            role = structure_record["role"]
            if role not in role_counts:
                raise ValueError(f"unexpected development role: {role}")
            contact_record = source_records.get(identifier)
            if contact_record is None:
                raise ValueError(f"missing contact record: {identifier}")
            if (
                int(contact_record["antibody_length"])
                != int(structure_record["antibody_length"])
                or int(contact_record["antigen_length"])
                != int(structure_record["antigen_length"])
            ):
                raise ValueError(f"residue-axis length mismatch: {identifier}")

            relative = Path(contact_record["artifact_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe contact artifact path: {relative}")
            source_artifact = args.source_contact_root / relative
            if sha256_file(source_artifact) != contact_record["artifact_sha256"]:
                raise ValueError(f"contact artifact SHA-256 differs: {identifier}")
            target_artifact = temporary_root / relative
            target_artifact.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_artifact, target_artifact)
            if sha256_file(target_artifact) != contact_record["artifact_sha256"]:
                raise ValueError(f"adapted contact artifact SHA-256 differs: {identifier}")
            source_stat = source_artifact.stat()
            target_stat = target_artifact.stat()
            if (source_stat.st_dev, source_stat.st_ino) != (
                target_stat.st_dev,
                target_stat.st_ino,
            ):
                raise ValueError(f"contact artifact was not hard-linked: {identifier}")
            artifact_inodes.append((target_stat.st_dev, target_stat.st_ino))
            selected.append(dict(contact_record))
            role_counts[role] += 1

        if role_counts != {"train": 2565, "validation": 549}:
            raise ValueError("development role counts differ")
        if len(set(artifact_inodes)) != 3114:
            raise ValueError("contact artifacts are not one-to-one")

        original_authority = dict(source.get("authority", {}))
        manifest = {
            "format_version": source["format_version"],
            "authority": {
                **original_authority,
                "structure_manifest_sha256": structure_sha256,
                "adapter_type": (
                    "development_only_pair_label_rebind_without_artifact_change"
                ),
                "source_contact_manifest_sha256": SOURCE_CONTACT_MANIFEST_SHA256,
                "source_contact_structure_manifest_sha256": original_authority.get(
                    "structure_manifest_sha256"
                ),
                "target_predicted_structure_manifest_sha256": structure_sha256,
                "pair_label_artifacts_reused_byte_identically": True,
                "pair_label_artifacts_rehashed": True,
                "pair_label_artifacts_hard_linked": True,
                "residue_axis_lengths_verified": True,
                "development_membership_source": (
                    "predicted structure manifest train/validation roles"
                ),
                "test_pair_artifacts_read": False,
                "test_pair_artifacts_materialized": False,
                "temporal_holdout_access": False,
                "selection_or_tuning": False,
                "experimental_bound_pair_labels_used_as_supervision": True,
                "experimental_bound_coordinates_used_as_model_input": False,
            },
            "counts": {
                "records": len(selected),
                "candidate_pairs": sum(
                    int(record["candidate_pair_count"]) for record in selected
                ),
                "positive_pairs": sum(
                    int(record["positive_pair_count"]) for record in selected
                ),
                "by_role": role_counts,
                "skipped_non_development_records": len(source["records"])
                - len(selected),
            },
            "records": selected,
        }
        manifest_path = temporary_root / "CONTACT_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_bytes(manifest))
        manifest_sha256 = sha256_file(manifest_path)
        receipt = {
            "format_version": "sabdab2_contact_authority_adapter_receipt_v1",
            "status": "PASS_PROVENANCE_PRESERVING_CONTACT_AUTHORITY_ADAPTER",
            "manifest_sha256": manifest_sha256,
            "source_contact_manifest_sha256": SOURCE_CONTACT_MANIFEST_SHA256,
            "predicted_structure_manifest_sha256": structure_sha256,
            "record_count": 3114,
            "role_counts": role_counts,
            "all_pair_artifact_sha256_verified": True,
            "all_pair_artifacts_hard_linked": True,
            "all_residue_axis_lengths_verified": True,
            "pair_labels_regenerated": False,
            "pair_label_artifacts_modified": False,
            "test_pair_artifacts_read": False,
            "test_pair_artifacts_materialized": False,
            "temporal_holdout_access": False,
            "selection_or_tuning": False,
            "claim_boundary": (
                "Metadata-only development authority adapter for unchanged pair-label "
                "artifacts; no model, label, split, optimization, or result claim."
            ),
        }
        (temporary_root / "CONTACT_CACHE_RECEIPT.json").write_bytes(
            canonical_bytes(receipt)
        )
        os.replace(temporary_root, args.output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": receipt["status"],
                "manifest_sha256": manifest_sha256,
                "receipt_sha256": sha256_file(
                    args.output_root / "CONTACT_CACHE_RECEIPT.json"
                ),
                "record_count": len(selected),
                "role_counts": role_counts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
