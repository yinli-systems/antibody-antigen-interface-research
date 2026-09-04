#!/usr/bin/env python3
"""Export an authenticated, model-ready subset of SAbDab2 ESM-2 embeddings."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import torch


EMBEDDING_FORMAT = "sppider_seq_embedding_v1"
MANIFEST_FORMAT = "sabdab2_esm_embedding_manifest_v1"
EXPORT_FORMAT = "sabdab2_esm_subset_export_v1"
MAPPING_COLUMNS = (
    "instance_id",
    "antibody_sequence_sha256",
    "antibody_length",
    "antibody_artifact_path",
    "antigen_sequence_sha256",
    "antigen_length",
    "antigen_artifact_path",
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_file(path: Path, expected: str, description: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} is absent or symbolic: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{description} SHA-256 differs: expected {expected}, observed {observed}"
        )
    return observed


def read_request_ids(path: Path, column: str) -> list[str]:
    if path.suffix.lower() in {".txt", ".list"}:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [value for value in values if value and not value.startswith("#")]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise ValueError(f"Request CSV lacks {column!r}")
            return [str(row[column]).strip() for row in reader]
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("instance_ids", "sample_ids", "records"):
                if key in value:
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError("Request JSON must contain a list of identifiers or records")
        return [
            str(item[column] if isinstance(item, dict) else item).strip()
            for item in value
        ]
    raise ValueError("Request IDs must be .txt, .list, .csv, or .json")


def validate_request_ids(values: list[str], expected_count: int) -> list[str]:
    if any(not value for value in values):
        raise ValueError("Request contains an empty instance identifier")
    if len(values) != expected_count:
        raise ValueError(
            f"Request count differs: expected {expected_count}, observed {len(values)}"
        )
    if len(set(values)) != len(values):
        raise ValueError("Request contains duplicate instance identifiers")
    return sorted(values)


def validate_embedding_artifact(
    artifact: Any,
    *,
    expected_sha256: str,
    expected_length: int,
    expected_dimension: int,
    expected_model_name: str,
    expected_model_revision: str,
    expected_model_weight_sha256: str,
) -> None:
    if not isinstance(artifact, dict):
        raise ValueError("Embedding artifact is not a dictionary")
    if artifact.get("format_version") != EMBEDDING_FORMAT:
        raise ValueError("Embedding artifact format differs")
    if artifact.get("sequence_sha256") != expected_sha256:
        raise ValueError("Embedding artifact sequence SHA-256 differs")
    if int(artifact.get("sequence_length", -1)) != expected_length:
        raise ValueError("Embedding artifact sequence length differs")
    if artifact.get("model_name") != expected_model_name:
        raise ValueError("Embedding artifact model name differs")
    if (
        artifact.get("model_revision") != expected_model_revision
        and artifact.get("model_weight_sha256") != expected_model_weight_sha256
    ):
        raise ValueError("Embedding artifact lacks the pinned model identity")
    starts = artifact.get("starts")
    ends = artifact.get("ends")
    chunks = artifact.get("chunks")
    if not isinstance(starts, list) or not isinstance(ends, list) or not isinstance(chunks, list):
        raise ValueError("Embedding artifact chunk metadata is malformed")
    if not chunks or not (len(starts) == len(ends) == len(chunks)):
        raise ValueError("Embedding artifact chunk counts differ")
    covered = torch.zeros(expected_length, dtype=torch.bool)
    for start_value, end_value, chunk in zip(starts, ends, chunks):
        start, end = int(start_value), int(end_value)
        if start < 0 or start >= end or end > expected_length:
            raise ValueError("Embedding artifact chunk interval is invalid")
        if not torch.is_tensor(chunk) or tuple(chunk.shape) != (
            end - start,
            expected_dimension,
        ):
            raise ValueError("Embedding artifact chunk shape differs")
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("Embedding artifact contains non-finite values")
        covered[start:end] = True
    if not bool(covered.all()):
        raise ValueError("Embedding artifact does not cover the residue axis")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--embedding-root", required=True, type=Path)
    parser.add_argument("--embedding-manifest-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-instance-count", type=int, default=455)
    parser.add_argument("--request-ids", type=Path)
    parser.add_argument("--request-ids-sha256")
    parser.add_argument("--request-id-column", default="instance_id")
    parser.add_argument("--split-column", default="frozen_split")
    parser.add_argument("--split-role", default="test")
    parser.add_argument("--transfer-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_instance_count <= 0:
        raise ValueError("--expected-instance-count must be positive")
    if (args.request_ids is None) != (args.request_ids_sha256 is None):
        raise ValueError("--request-ids and --request-ids-sha256 must be supplied together")
    for path in (args.output_root,):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    prepared_sha256 = authenticate_file(
        args.prepared, args.prepared_sha256, "prepared SAbDab2 authority"
    )
    source_manifest_path = args.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
    source_manifest_sha256 = authenticate_file(
        source_manifest_path,
        args.embedding_manifest_sha256,
        "source ESM-2 embedding manifest",
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema_version") != MANIFEST_FORMAT
        or source_manifest.get("complete") is not True
    ):
        raise ValueError("Source embedding manifest is not a complete authenticated cohort")
    source_records = source_manifest.get("records")
    if not isinstance(source_records, list):
        raise ValueError("Source embedding manifest records are malformed")
    source_by_sha: dict[str, dict[str, Any]] = {}
    for record in source_records:
        key = str(record["sequence_sha256"])
        if key in source_by_sha:
            raise ValueError("Source embedding manifest contains duplicate sequence SHA-256")
        source_by_sha[key] = record

    columns = [
        "instance_id",
        "ab_sequence_sha256",
        "ag_sequence_sha256",
        "ab_length",
        "ag_length",
        args.split_column,
    ]
    prepared = pd.read_parquet(args.prepared, columns=columns)
    if prepared["instance_id"].duplicated().any():
        raise ValueError("Prepared authority contains duplicate instance identifiers")
    prepared["instance_id"] = prepared["instance_id"].astype(str)
    prepared_by_id = prepared.set_index("instance_id", drop=False)

    if args.request_ids is not None:
        request_sha256 = authenticate_file(
            args.request_ids, args.request_ids_sha256, "request instance manifest"
        )
        request_ids = validate_request_ids(
            read_request_ids(args.request_ids, args.request_id_column),
            args.expected_instance_count,
        )
        selection_authority = {
            "kind": "external_request_manifest",
            "path": str(args.request_ids.resolve()),
            "sha256": request_sha256,
            "id_column": args.request_id_column,
        }
    else:
        selected = prepared[prepared[args.split_column].astype(str) == args.split_role]
        request_ids = validate_request_ids(
            selected["instance_id"].tolist(), args.expected_instance_count
        )
        selection_authority = {
            "kind": "authenticated_prepared_split",
            "column": args.split_column,
            "role": args.split_role,
        }
    missing = sorted(set(request_ids) - set(prepared_by_id.index))
    if missing:
        raise ValueError(f"Requested identifiers absent from prepared authority: {missing[:5]}")

    temporary_root = args.output_root.with_name(
        args.output_root.name + f".tmp-{os.getpid()}"
    )
    if temporary_root.exists() or temporary_root.is_symlink():
        raise FileExistsError(f"Temporary output already exists: {temporary_root}")
    temporary_root.mkdir(parents=True)
    try:
        mapping: list[dict[str, Any]] = []
        required: dict[str, int] = {}
        for instance_id in request_ids:
            row = prepared_by_id.loc[instance_id]
            ab_sha = str(row["ab_sequence_sha256"])
            ag_sha = str(row["ag_sequence_sha256"])
            ab_length = int(row["ab_length"])
            ag_length = int(row["ag_length"])
            for key, length in ((ab_sha, ab_length), (ag_sha, ag_length)):
                prior = required.setdefault(key, length)
                if prior != length:
                    raise ValueError("A sequence SHA-256 maps to inconsistent lengths")
            mapping.append(
                {
                    "instance_id": instance_id,
                    "antibody_sequence_sha256": ab_sha,
                    "antibody_length": ab_length,
                    "antibody_artifact_path": f"{ab_sha[:2]}/{ab_sha}.pt",
                    "antigen_sequence_sha256": ag_sha,
                    "antigen_length": ag_length,
                    "antigen_artifact_path": f"{ag_sha[:2]}/{ag_sha}.pt",
                }
            )

        subset_records: list[dict[str, Any]] = []
        logical_bytes = 0
        for index, key in enumerate(sorted(required), start=1):
            if key not in source_by_sha:
                raise ValueError(f"Sequence absent from source embedding manifest: {key}")
            record = source_by_sha[key]
            expected_relative = Path(key[:2]) / f"{key}.pt"
            if Path(str(record["artifact_path"])) != expected_relative:
                raise ValueError("Source artifact path is not canonical")
            if int(record["sequence_length"]) != required[key]:
                raise ValueError("Source manifest sequence length differs from prepared authority")
            source_path = args.embedding_root / expected_relative
            artifact_sha256 = authenticate_file(
                source_path, str(record["artifact_sha256"]), "source embedding artifact"
            )
            artifact = torch.load(source_path, map_location="cpu", weights_only=True)
            validate_embedding_artifact(
                artifact,
                expected_sha256=key,
                expected_length=required[key],
                expected_dimension=int(source_manifest["embedding_dimension"]),
                expected_model_name=str(source_manifest["model_name"]),
                expected_model_revision=str(source_manifest["model_revision"]),
                expected_model_weight_sha256=str(
                    source_manifest["model_weight_sha256"]
                ),
            )
            target_path = temporary_root / expected_relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if args.transfer_mode == "hardlink":
                os.link(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
            if sha256_file(target_path) != artifact_sha256:
                raise ValueError("Exported artifact SHA-256 differs")
            logical_bytes += target_path.stat().st_size
            subset_records.append(
                {
                    "sequence_sha256": key,
                    "sequence_length": required[key],
                    "artifact_path": expected_relative.as_posix(),
                    "artifact_sha256": artifact_sha256,
                    "state": f"exported_{args.transfer_mode}",
                }
            )
            if index % 50 == 0 or index == len(required):
                print(f"verified/exported {index}/{len(required)} embeddings", flush=True)

        ids_path = temporary_root / "REQUESTED_INSTANCE_IDS.txt"
        ids_path.write_text("\n".join(request_ids) + "\n", encoding="utf-8")
        mapping_path = temporary_root / "INSTANCE_EMBEDDING_MAP.csv"
        write_csv(mapping_path, mapping)
        subset_manifest = {
            "schema_version": MANIFEST_FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "authority": {
                "prepared_path": str(args.prepared.resolve()),
                "prepared_sha256": prepared_sha256,
                "prepared_columns_read": columns,
                "source_embedding_manifest_path": str(source_manifest_path.resolve()),
                "source_embedding_manifest_sha256": source_manifest_sha256,
                "selection": selection_authority,
                "task_labels_read": False,
            },
            "model_name": source_manifest["model_name"],
            "model_revision": source_manifest["model_revision"],
            "model_weight_sha256": source_manifest["model_weight_sha256"],
            "embedding_dimension": source_manifest["embedding_dimension"],
            "selected_sequence_count": len(subset_records),
            "complete_cohort_sequence_count": source_manifest[
                "complete_cohort_sequence_count"
            ],
            "complete": False,
            "records": subset_records,
            "claim_boundary": (
                "Frozen sequence embeddings for the requested instances only; no task "
                "labels, structures, model predictions, or performance results are included."
            ),
        }
        manifest_path = temporary_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
        manifest_path.write_bytes(canonical_json_bytes(subset_manifest))
        instance_set_sha256 = hashlib.sha256(ids_path.read_bytes()).hexdigest()
        tree_digest = hashlib.sha256()
        for record in subset_records:
            tree_digest.update(record["artifact_path"].encode("ascii"))
            tree_digest.update(b"\0")
            tree_digest.update(record["artifact_sha256"].encode("ascii"))
            tree_digest.update(b"\n")
        receipt = {
            "format_version": EXPORT_FORMAT,
            "status": "PASS_MODEL_READY_ESM2_SUBSET_EXPORT",
            "counts": {
                "requested_instances": len(request_ids),
                "mapping_rows": len(mapping),
                "unique_sequences": len(subset_records),
            },
            "requested_instance_ids_sha256": instance_set_sha256,
            "instance_embedding_map_sha256": sha256_file(mapping_path),
            "subset_embedding_manifest_sha256": sha256_file(manifest_path),
            "source_embedding_manifest_sha256": source_manifest_sha256,
            "artifact_tree_sha256": tree_digest.hexdigest(),
            "logical_embedding_bytes": logical_bytes,
            "transfer_mode": args.transfer_mode,
            "all_artifact_hashes_verified": True,
            "all_artifacts_schema_length_shape_and_finite_verified": True,
            "task_labels_read": False,
            "structures_read": False,
            "model_predictions_read": False,
        }
        receipt_path = temporary_root / "EXPORT_RECEIPT.json"
        receipt_path.write_bytes(canonical_json_bytes(receipt))
        os.replace(temporary_root, args.output_root)
        print(json.dumps({**receipt, "output_root": str(args.output_root)}, indent=2))
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
