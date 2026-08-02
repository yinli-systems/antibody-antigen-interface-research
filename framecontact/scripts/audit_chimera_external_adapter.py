#!/usr/bin/env python3
"""Audit frozen CHIMERA materialization and residue-axis mapping before scoring."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    adapter_modes = parser.add_mutually_exclusive_group()
    adapter_modes.add_argument(
        "--native-axis",
        action="store_true",
        help=(
            "Use the post-access atom14-axis adapter without PDB residue mapping. "
            "This mode is coverage/secondary evidence only, never confirmatory."
        ),
    )
    adapter_modes.add_argument(
        "--pdb-anchored",
        action="store_true",
        help=(
            "Map official contact tuples to native feature axes through unique CA "
            "coordinate anchors. This post-access mode is secondary evidence only."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    adapter = _load_module(
        "chimera_external_adapter_audit",
        project_root / "src" / "interaction_first" / "chimera_external.py",
    )
    downloader = _load_module(
        "chimera_external_downloader_audit",
        project_root / "scripts" / "download_chimera_external_subset.py",
    )
    protocol = adapter.verify_frozen_authority(args.protocol, args.receipt)
    materialization = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        materialization.get("schema_version")
        != "chimera_external_materialization_manifest_v1"
        or materialization.get("dataset_id") != adapter.DATASET_ID
        or materialization.get("pinned_revision") != adapter.PINNED_REVISION
    ):
        raise SystemExit("materialization manifest authority differs")

    artifacts: dict[str, dict[str, Any]] = {}
    for record in materialization.get("artifacts", []):
        relative = record.get("path")
        if not isinstance(relative, str) or relative in artifacts:
            raise SystemExit("materialization manifest contains malformed/duplicate paths")
        path = args.root / relative
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"missing or unsafe materialized artifact: {relative}")
        observed = downloader.observed_digest(path, record["digest_kind"])
        if observed != record["digest"] or path.stat().st_size != int(record["size"]):
            raise SystemExit(f"materialized artifact verification failed: {relative}")
        artifacts[relative] = record

    split_payloads: dict[str, tuple[str, ...]] = {}
    split_hashes: dict[str, str] = {}
    for split in protocol["evaluation"]["primary_splits"]:
        split_name, subset = split.split(".", 1)
        relative = f"splits/{split_name}.json"
        if relative not in artifacts:
            raise SystemExit(f"split is absent from materialization manifest: {relative}")
        split_path = args.root / relative
        payload = adapter.load_split(split_path)
        split_payloads[split] = payload[subset]
        split_hashes[split] = adapter.sha256_file(split_path)

    all_ids = sorted(set().union(*(set(values) for values in split_payloads.values())))
    adapted: dict[str, dict[str, Any]] = {}
    exclusions: dict[str, str] = {}
    cache_records: list[dict[str, Any]] = []
    temporary_cache: Path | None = None
    if args.cache_root is not None:
        if args.cache_root.exists() or args.cache_root.is_symlink():
            raise SystemExit("cache-root already exists")
        args.cache_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = Path(
            tempfile.mkdtemp(
                prefix=f".{args.cache_root.name}.tmp-", dir=args.cache_root.parent
            )
        )
    for index, complex_id in enumerate(all_ids, start=1):
        feature_relative = f"complex_features/{complex_id}.pt"
        pdb_id = complex_id.split("_", 1)[0].lower()
        pdb_relative = f"structures/{pdb_id}.pdb"
        required_artifacts = [feature_relative]
        if not args.native_axis:
            required_artifacts.append(pdb_relative)
        if any(relative not in artifacts for relative in required_artifacts):
            exclusions[complex_id] = "missing_manifest_bound_artifact"
            continue
        try:
            feature_record = artifacts[feature_relative]
            expected_sha = (
                feature_record["digest"]
                if feature_record["digest_kind"] == "sha256"
                else None
            )
            feature = adapter.load_feature_archive(
                args.root / feature_relative, expected_sha256=expected_sha
            )
            if args.native_axis:
                row = adapter.adapt_complex_native_axis(feature)
            elif args.pdb_anchored:
                row = adapter.adapt_complex_pdb_anchored(
                    feature, args.root / pdb_relative
                )
            else:
                row = adapter.adapt_complex(feature, args.root / pdb_relative)
            cache_record: dict[str, Any] = {}
            if temporary_cache is not None:
                identifier_sha = hashlib.sha256(complex_id.encode("ascii")).hexdigest()
                relative_cache = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.npz"
                cache_path = temporary_cache / relative_cache
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_artifact = cache_path.with_suffix(".npz.tmp")
                with temporary_artifact.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        instance_id=np.asarray(complex_id, dtype=np.str_),
                        antibody_sequence=np.asarray(row["antibody_sequence"], dtype=np.str_),
                        antigen_sequence=np.asarray(row["antigen_sequence"], dtype=np.str_),
                        antibody_backbone=row["antibody_backbone"],
                        antigen_backbone=row["antigen_backbone"],
                        antibody_chain_ids=row["antibody_chain_ids"],
                        antigen_chain_ids=row["antigen_chain_ids"],
                        antibody_frame_valid_mask=row["antibody_frame_valid_mask"],
                        antigen_frame_valid_mask=row["antigen_frame_valid_mask"],
                        antibody_missing_frame_mask=row["antibody_missing_frame_mask"],
                        antigen_missing_frame_mask=row["antigen_missing_frame_mask"],
                        antibody_labels=row["antibody_labels"],
                        antigen_labels=row["antigen_labels"],
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_artifact, cache_path)
                cache_record = {
                    "artifact_path": relative_cache.as_posix(),
                    "artifact_sha256": adapter.sha256_file(cache_path),
                    "split_memberships": sorted(
                        split
                        for split, identifiers in split_payloads.items()
                        if complex_id in identifiers
                    ),
                    "source_feature_path": feature_relative,
                    "source_feature_digest": artifacts[feature_relative]["digest"],
                }
                if not args.native_axis:
                    cache_record.update(
                        {
                            "source_pdb_path": pdb_relative,
                            "source_pdb_digest": artifacts[pdb_relative]["digest"],
                        }
                    )
                cache_records.append({"instance_id": complex_id, **cache_record})
            adapted[complex_id] = {
                "antibody_residues": len(row["antibody_sequence"]),
                "antigen_residues": len(row["antigen_sequence"]),
                "paratope_positives": int(row["antibody_labels"].sum()),
                "epitope_positives": int(row["antigen_labels"].sum()),
                "antibody_valid_frames": int(row["antibody_frame_valid_mask"].sum()),
                "antigen_valid_frames": int(row["antigen_frame_valid_mask"].sum()),
                "antibody_missing_frames": int(row["antibody_missing_frame_mask"].sum()),
                "antigen_missing_frames": int(row["antigen_missing_frame_mask"].sum()),
                **(
                    {
                        "native_contact_pair_count": int(
                            row["native_contact_pair_count"]
                        ),
                        "provided_contact_pair_count": int(
                            row["provided_contact_pair_count"]
                        ),
                    }
                    if args.native_axis
                    else {}
                ),
                **(
                    {
                        "provided_contact_pair_count": int(
                            row["provided_contact_pair_count"]
                        ),
                        "coordinate_anchored_antibody_residues": int(
                            row["coordinate_anchored_antibody_residues"]
                        ),
                        "coordinate_anchored_antigen_residues": int(
                            row["coordinate_anchored_antigen_residues"]
                        ),
                        "feature_only_antibody_residues": int(
                            row["feature_only_antibody_residues"]
                        ),
                        "feature_only_antigen_residues": int(
                            row["feature_only_antigen_residues"]
                        ),
                        "distance_disambiguated_pair_count": int(
                            row["distance_disambiguated_pair_count"]
                        ),
                        "unique_pair_distance_mismatch_count": int(
                            row["unique_pair_distance_mismatch_count"]
                        ),
                        "maximum_unique_pair_distance_error": float(
                            row["maximum_unique_pair_distance_error"]
                        ),
                    }
                    if args.pdb_anchored
                    else {}
                ),
                **cache_record,
            }
        except adapter.ChimeraExternalError as error:
            exclusions[complex_id] = str(error)
        if index % 50 == 0 or index == len(all_ids):
            print(f"mapped {index}/{len(all_ids)} complexes", flush=True)

    split_results: dict[str, dict[str, Any]] = {}
    for split, identifiers in split_payloads.items():
        usable = [identifier for identifier in identifiers if identifier in adapted]
        excluded = [identifier for identifier in identifiers if identifier in exclusions]
        coverage = len(usable) / len(identifiers)
        split_results[split] = {
            "eligible": len(identifiers),
            "usable": len(usable),
            "excluded": len(excluded),
            "coverage": coverage,
            "coverage_gate_passed": coverage >= 0.95,
            "confirmatory_coverage_gate_passed": (
                coverage >= 0.95 and not (args.native_axis or args.pdb_anchored)
            ),
            "excluded_complexes": [
                {"complex_id": identifier, "reason": exclusions[identifier]}
                for identifier in excluded
            ],
        }

    output = {
        "schema_version": (
            "chimera_native_axis_coverage_audit_v2"
            if args.native_axis
            else (
                "chimera_pdb_anchored_coverage_audit_v3"
                if args.pdb_anchored
                else "chimera_external_adapter_audit_v1"
            )
        ),
        "adapter_mode": (
            "native_atom14_axis_post_access_v2"
            if args.native_axis
            else (
                "pdb_coordinate_anchored_official_contacts_post_access_v3"
                if args.pdb_anchored
                else "pdb_numbered_axis_frozen_v1"
            )
        ),
        "claim_boundary": (
            "Post-access coverage and secondary robustness analysis only; changing "
            "the mapping after test access cannot create a new confirmatory result."
            if args.native_axis or args.pdb_anchored
            else "Pre-access frozen strict adapter audit."
        ),
        "post_access_method_change": bool(args.native_axis or args.pdb_anchored),
        "confirmatory_interpretation_permitted": not (
            args.native_axis or args.pdb_anchored
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": adapter.sha256_file(args.protocol),
        "receipt_sha256": adapter.sha256_file(args.receipt),
        "materialization_manifest_sha256": adapter.sha256_file(args.manifest),
        "dataset_id": adapter.DATASET_ID,
        "pinned_revision": adapter.PINNED_REVISION,
        "split_sha256": split_hashes,
        "unique_eligible_complexes": len(all_ids),
        "unique_usable_complexes": len(adapted),
        "unique_excluded_complexes": len(exclusions),
        "exclusion_reason_counts": dict(sorted(Counter(exclusions.values()).items())),
        "splits": split_results,
        "adapted_complexes": adapted,
        "model_scores_read": False,
        "metrics_computed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(output, indent=2, sort_keys=True) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, args.output)
    if temporary_cache is not None and args.cache_root is not None:
        cache_manifest = {
            "schema_version": "chimera_strict_adapter_cache_v1",
            "claim_boundary": (
                (
                    "Post-access adapter cache for secondary robustness analysis only; "
                    "it cannot create a new confirmatory frozen-test result."
                )
                if args.native_axis or args.pdb_anchored
                else (
                    "Frozen-protocol strict adapter cache only; no model scores or "
                    "metrics are contained in this cache."
                )
            ),
            "adapter_mode": output["adapter_mode"],
            "dataset_id": adapter.DATASET_ID,
            "pinned_revision": adapter.PINNED_REVISION,
            "protocol_sha256": adapter.sha256_file(args.protocol),
            "remote_freeze_receipt_sha256": adapter.sha256_file(args.receipt),
            "materialization_manifest_sha256": adapter.sha256_file(args.manifest),
            "adapter_audit_sha256": adapter.sha256_file(args.output),
            "adapter_source_sha256": adapter.sha256_file(
                project_root / "src" / "interaction_first" / "chimera_external.py"
            ),
            "record_count": len(cache_records),
            "records": sorted(cache_records, key=lambda item: item["instance_id"]),
        }
        cache_manifest_path = temporary_cache / "ADAPTER_CACHE_MANIFEST.json"
        cache_manifest_path.write_text(
            json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_cache, args.cache_root)
    print(f"audit={args.output}")
    print(f"audit_sha256={adapter.sha256_file(args.output)}")
    if args.cache_root is not None:
        print(f"cache_root={args.cache_root}")
        print(
            "cache_manifest_sha256="
            f"{adapter.sha256_file(args.cache_root / 'ADAPTER_CACHE_MANIFEST.json')}"
        )


if __name__ == "__main__":
    main()
