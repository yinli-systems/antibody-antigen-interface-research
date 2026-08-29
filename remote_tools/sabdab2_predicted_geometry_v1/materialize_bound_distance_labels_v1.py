#!/usr/bin/env python3
"""Materialize development-only bound-derived distance bins as labels, never inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import torch


MANIFEST_FORMAT = "sabdab2_cross_chain_distance_label_cache_v1"
RECEIPT_FORMAT = "sabdab2_cross_chain_distance_label_cache_receipt_v1"
ARTIFACT_FORMAT = "sabdab2_cross_chain_distance_label_artifact_v1"
BOUNDARIES = (4.0, 8.0, 16.0, 32.0)


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


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def authenticated_json(path: Path, expected_sha256: str, role: str) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{role} SHA-256 differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"{role} lacks records")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound-structure-root", type=Path, required=True)
    parser.add_argument("--bound-structure-manifest-sha256", required=True)
    parser.add_argument("--model-input-structure-root", type=Path, required=True)
    parser.add_argument("--model-input-structure-manifest-sha256", required=True)
    parser.add_argument("--development-split", type=Path, required=True)
    parser.add_argument("--development-split-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    bound_manifest_path = args.bound_structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    model_manifest_path = (
        args.model_input_structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    )
    bound = authenticated_json(
        bound_manifest_path,
        args.bound_structure_manifest_sha256,
        "bound structure manifest",
    )
    model_input = authenticated_json(
        model_manifest_path,
        args.model_input_structure_manifest_sha256,
        "model-input structure manifest",
    )
    if sha256_file(args.development_split) != args.development_split_sha256:
        raise ValueError("development split SHA-256 differs")
    with args.development_split.open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    if len(split_rows) != 3114:
        raise ValueError("development split must contain exactly 3114 instances")
    split_by_id = {str(row["instance_id"]): row for row in split_rows}
    if len(split_by_id) != len(split_rows):
        raise ValueError("development split identifiers are not unique")
    role_counts = Counter(str(row["role"]) for row in split_rows)
    if role_counts != Counter({"train": 2565, "validation": 549}):
        raise ValueError("development split role counts differ")
    bound_by_id = {str(row["instance_id"]): row for row in bound["records"]}
    model_by_id = {
        str(row["instance_id"]): row for row in model_input["records"]
    }
    selected_ids = set(split_by_id)
    if len(bound_by_id) != len(bound["records"]) or not selected_ids.issubset(
        bound_by_id
    ):
        raise ValueError("bound structure cohort lacks unique development instances")
    if set(model_by_id) != selected_ids:
        raise ValueError("model-input structure cache differs from development split")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent
        )
    )
    records: list[dict] = []
    total_pairs = 0
    observed_pairs = 0
    try:
        boundaries = torch.tensor(BOUNDARIES, dtype=torch.float32)
        for ordinal, identifier in enumerate(sorted(selected_ids), start=1):
            role = str(split_by_id[identifier]["role"])
            bound_record = bound_by_id[identifier]
            model_record = model_by_id[identifier]
            for key in (
                "antibody_length",
                "antigen_length",
                "antibody_sequence_sha256",
                "antigen_sequence_sha256",
            ):
                if str(bound_record[key]) != str(model_record[key]):
                    raise ValueError(f"{identifier}: bound/model residue authority differs")
            source_path = args.bound_structure_root / str(
                bound_record["artifact_path"]
            )
            if (
                source_path.is_symlink()
                or not source_path.is_file()
                or sha256_file(source_path) != bound_record["artifact_sha256"]
            ):
                raise ValueError(f"{identifier}: bound structure artifact differs")
            artifact = torch.load(source_path, map_location="cpu", weights_only=True)
            antibody = artifact.get("antibody_backbone")
            antigen = artifact.get("antigen_backbone")
            antibody_valid = artifact.get("antibody_frame_valid_mask")
            antigen_valid = artifact.get("antigen_frame_valid_mask")
            antibody_length = int(bound_record["antibody_length"])
            antigen_length = int(bound_record["antigen_length"])
            if (
                artifact.get("instance_id") != identifier
                or artifact.get("antibody_sequence_sha256")
                != bound_record["antibody_sequence_sha256"]
                or artifact.get("antigen_sequence_sha256")
                != bound_record["antigen_sequence_sha256"]
                or not torch.is_tensor(antibody)
                or not torch.is_tensor(antigen)
                or tuple(antibody.shape) != (antibody_length, 3, 3)
                or tuple(antigen.shape) != (antigen_length, 3, 3)
                or not torch.is_tensor(antibody_valid)
                or not torch.is_tensor(antigen_valid)
                or antibody_valid.dtype != torch.bool
                or antigen_valid.dtype != torch.bool
                or tuple(antibody_valid.shape) != (antibody_length,)
                or tuple(antigen_valid.shape) != (antigen_length,)
                or not bool(torch.isfinite(antibody).all())
                or not bool(torch.isfinite(antigen).all())
            ):
                raise ValueError(f"{identifier}: bound structure artifact schema differs")
            distances = torch.cdist(
                antibody[:, 1, :].float(), antigen[:, 1, :].float()
            )
            labels = torch.bucketize(distances, boundaries).to(torch.uint8).numpy()
            antibody_valid_numpy = antibody_valid.numpy().astype(np.bool_, copy=True)
            antigen_valid_numpy = antigen_valid.numpy().astype(np.bool_, copy=True)
            identifier_sha = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
            relative_path = (
                Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.npz"
            )
            target = temporary_root / relative_path
            write_deterministic_npz(
                target,
                {
                    "format_version": np.asarray(ARTIFACT_FORMAT),
                    "instance_id": np.asarray(identifier),
                    "antibody_length": np.asarray(antibody_length, dtype=np.int64),
                    "antigen_length": np.asarray(antigen_length, dtype=np.int64),
                    "antibody_sequence_sha256": np.asarray(
                        bound_record["antibody_sequence_sha256"]
                    ),
                    "antigen_sequence_sha256": np.asarray(
                        bound_record["antigen_sequence_sha256"]
                    ),
                    "distance_bin_labels": labels,
                    "antibody_frame_valid_mask": antibody_valid_numpy,
                    "antigen_frame_valid_mask": antigen_valid_numpy,
                },
            )
            local_pairs = antibody_length * antigen_length
            local_observed = int(antibody_valid.sum()) * int(antigen_valid.sum())
            total_pairs += local_pairs
            observed_pairs += local_observed
            records.append(
                {
                    "instance_id": identifier,
                    "split_role": role,
                    "antibody_length": antibody_length,
                    "antigen_length": antigen_length,
                    "antibody_sequence_sha256": bound_record[
                        "antibody_sequence_sha256"
                    ],
                    "antigen_sequence_sha256": bound_record[
                        "antigen_sequence_sha256"
                    ],
                    "candidate_pair_count": local_pairs,
                    "observed_distance_pair_count": local_observed,
                    "source_bound_artifact_sha256": bound_record[
                        "artifact_sha256"
                    ],
                    "artifact_path": relative_path.as_posix(),
                    "artifact_size_bytes": target.stat().st_size,
                    "artifact_sha256": sha256_file(target),
                }
            )
            if ordinal % 100 == 0 or ordinal == len(selected_ids):
                print(
                    json.dumps(
                        {"materialized": ordinal, "records": len(selected_ids)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        manifest = {
            "format_version": MANIFEST_FORMAT,
            "authority": {
                "source_bound_structure_manifest_sha256": args.bound_structure_manifest_sha256,
                "model_input_structure_manifest_sha256": args.model_input_structure_manifest_sha256,
                "development_split_sha256": args.development_split_sha256,
                "distance_boundaries_angstrom": list(BOUNDARIES),
                "distance_definition": "C-alpha to C-alpha Euclidean distance in the authenticated experimental bound complex",
                "distance_bin_assignment": "torch.bucketize right=False",
                "observed_pair_mask": "outer product of authenticated bound antibody and antigen frame-valid masks",
                "bound_cross_partner_distances_used_as_auxiliary_training_labels": True,
                "bound_coordinates_exposed_to_model_input": False,
                "cross_partner_geometry_exposed_to_model_input": False,
                "task_labels_read": False,
                "test_structure_artifacts_read": False,
                "temporal_holdout_access": False,
            },
            "counts": {
                "records": len(records),
                "by_role": dict(sorted(role_counts.items())),
                "candidate_pairs": total_pairs,
                "observed_distance_pairs": observed_pairs,
            },
            "records": records,
        }
        manifest_path = temporary_root / "DISTANCE_LABEL_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_bytes(manifest))
        receipt = {
            "format_version": RECEIPT_FORMAT,
            "status": "PASS_DEVELOPMENT_ONLY_BOUND_DISTANCE_LABEL_CACHE",
            "manifest_sha256": sha256_file(manifest_path),
            "record_count": len(records),
            "role_counts": dict(sorted(role_counts.items())),
            "all_source_artifact_sha256_verified": True,
            "all_sequence_and_length_axes_verified": True,
            "all_distance_labels_finite_and_in_range": True,
            "bound_coordinates_exposed_to_model_input": False,
            "test_structure_artifacts_read": False,
            "task_labels_read": False,
            "temporal_holdout_access": False,
            "claim_boundary": "Development-only auxiliary distance supervision. Bound coordinates are reduced to five-class labels and are never provided as model inputs.",
        }
        receipt_path = temporary_root / "DISTANCE_LABEL_CACHE_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        os.replace(temporary_root, args.output_root)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "manifest_sha256": receipt["manifest_sha256"],
                    "receipt_sha256": sha256_file(
                        args.output_root / "DISTANCE_LABEL_CACHE_RECEIPT.json"
                    ),
                    "records": len(records),
                    "candidate_pairs": total_pairs,
                    "observed_distance_pairs": observed_pairs,
                },
                sort_keys=True,
            )
        )
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


if __name__ == "__main__":
    main()
