"""Prepare and finalize the complete AsEP-test x SAbDab2-dev Foldseek grid."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .asep_decontamination import CROSS_FOLDSEEK_RECEIPT_FORMAT
from .asep_frozen_test import LABEL_FREE_TEST_FORMAT
from .sabdab2_antigen_export import ANTIGEN_EXPORT_FORMAT
from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file


CROSS_PREPARATION_FORMAT = "framecontact_asep_sabdab2_cross_foldseek_preparation_v1"


class AsepCrossFoldseekError(ValueError):
    """Raised when a cross-dataset Foldseek rectangle is not complete."""


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AsepCrossFoldseekError(f"Cannot parse {name}") from error
    if not isinstance(payload, dict):
        raise AsepCrossFoldseekError(f"{name} must be a JSON object")
    return payload


def _csv(path: Path, required: set[str], name: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise AsepCrossFoldseekError(f"{name} columns differ")
            return [
                {str(key): str(value).strip() for key, value in row.items()}
                for row in reader
            ]
    except OSError as error:
        raise AsepCrossFoldseekError(f"Cannot read {name}") from error


def prepare_asep_sabdab2_cross_foldseek(
    *,
    label_free_test_root: Path,
    label_free_test_receipt_sha256: str,
    sabdab2_split_manifest: Path,
    sabdab2_split_manifest_sha256: str,
    sabdab2_antigen_export_root: Path,
    sabdab2_antigen_export_receipt_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Create immutable query/target directories from authenticated inputs."""
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite cross Foldseek prep: {output_root}")
    label_receipt_path = (
        label_free_test_root / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json"
    )
    label_manifest_path = label_free_test_root / "ASEP_LABEL_FREE_TEST_METADATA.csv"
    authenticate_file(
        label_receipt_path,
        label_free_test_receipt_sha256,
        "AsEP label-free test receipt",
    )
    label_receipt = _json(label_receipt_path, "AsEP label-free test receipt")
    authenticate_file(
        label_manifest_path,
        str(label_receipt.get("manifest_sha256", "")),
        "AsEP label-free test manifest",
    )
    if (
        label_receipt.get("format_version") != LABEL_FREE_TEST_FORMAT
        or label_receipt.get("test_label_open_count") != 0
        or label_receipt.get("checks", {}).get("official_test_graph_tensors_opened")
        is not False
    ):
        raise AsepCrossFoldseekError("AsEP label-free authority is ineligible")
    queries = _csv(
        label_manifest_path,
        {
            "instance_id",
            "role",
            "foldseek_structure_id",
            "antigen_structure_path",
            "antigen_structure_sha256",
        },
        "AsEP label-free test manifest",
    )
    if any(row["role"] != "test" for row in queries):
        raise AsepCrossFoldseekError("AsEP query manifest contains a non-test role")

    authenticate_file(
        sabdab2_split_manifest,
        sabdab2_split_manifest_sha256,
        "SAbDab2 split manifest",
    )
    split_rows = _csv(
        sabdab2_split_manifest,
        {"instance_id", "role", "antigen_structure_id"},
        "SAbDab2 split manifest",
    )
    selected_targets = {
        row["antigen_structure_id"]: row["role"]
        for row in split_rows
        if row["role"] in {"train", "validation"}
    }
    if not selected_targets or len(selected_targets) != sum(
        row["role"] in {"train", "validation"} for row in split_rows
    ):
        raise AsepCrossFoldseekError("SAbDab2 development structure axis is invalid")

    export_receipt_path = (
        sabdab2_antigen_export_root / "ANTIGEN_STRUCTURE_EXPORT_RECEIPT.json"
    )
    export_manifest_path = (
        sabdab2_antigen_export_root / "ANTIGEN_STRUCTURE_EXPORT_MANIFEST.csv"
    )
    authenticate_file(
        export_receipt_path,
        sabdab2_antigen_export_receipt_sha256,
        "SAbDab2 antigen export receipt",
    )
    export_receipt = _json(export_receipt_path, "SAbDab2 antigen export receipt")
    authenticate_file(
        export_manifest_path,
        str(export_receipt.get("manifest_sha256", "")),
        "SAbDab2 antigen export manifest",
    )
    if (
        export_receipt.get("format_version") != ANTIGEN_EXPORT_FORMAT
        or export_receipt.get("authority", {}).get("task_labels_read") is not False
    ):
        raise AsepCrossFoldseekError("SAbDab2 antigen export is ineligible")
    export_rows = _csv(
        export_manifest_path,
        {"antigen_structure_id", "foldseek_structure_id", "relative_path", "sha256"},
        "SAbDab2 antigen export manifest",
    )
    export_by_structure = {
        row["antigen_structure_id"]: row for row in export_rows
    }
    if not set(selected_targets).issubset(export_by_structure):
        raise AsepCrossFoldseekError("SAbDab2 export lacks development targets")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    query_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    try:
        query_directory = temporary / "queries"
        target_directory = temporary / "targets"
        query_directory.mkdir()
        target_directory.mkdir()
        for row in queries:
            source = label_free_test_root / row["antigen_structure_path"]
            authenticate_file(source, row["antigen_structure_sha256"], "AsEP query PDB")
            destination = query_directory / f"{row['foldseek_structure_id']}.pdb"
            shutil.copyfile(source, destination)
            query_records.append(
                {
                    "foldseek_structure_id": row["foldseek_structure_id"],
                    "instance_id": row["instance_id"],
                    "sha256": sha256_file(destination),
                }
            )
        for structure_id, role in sorted(selected_targets.items()):
            row = export_by_structure[structure_id]
            source = sabdab2_antigen_export_root / row["relative_path"]
            authenticate_file(source, row["sha256"], "SAbDab2 target PDB")
            destination = target_directory / f"{row['foldseek_structure_id']}.pdb"
            shutil.copyfile(source, destination)
            target_records.append(
                {
                    "foldseek_structure_id": row["foldseek_structure_id"],
                    "antigen_structure_id": structure_id,
                    "role": role,
                    "sha256": sha256_file(destination),
                }
            )
        if (
            len({row["foldseek_structure_id"] for row in query_records})
            != len(query_records)
            or len({row["foldseek_structure_id"] for row in target_records})
            != len(target_records)
        ):
            raise AsepCrossFoldseekError("Foldseek query or target IDs collide")
        query_records.sort(key=lambda row: row["foldseek_structure_id"])
        target_records.sort(key=lambda row: row["foldseek_structure_id"])
        query_manifest = temporary / "QUERY_MANIFEST.csv"
        target_manifest = temporary / "TARGET_MANIFEST.csv"
        for path, records in (
            (query_manifest, query_records),
            (target_manifest, target_records),
        ):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
                writer.writeheader()
                writer.writerows(records)
        receipt = {
            "format_version": CROSS_PREPARATION_FORMAT,
            "authority": {
                "label_free_test_receipt_sha256": label_free_test_receipt_sha256,
                "label_free_test_manifest_sha256": label_receipt["manifest_sha256"],
                "sabdab2_split_manifest_sha256": sabdab2_split_manifest_sha256,
                "sabdab2_antigen_export_receipt_sha256": (
                    sabdab2_antigen_export_receipt_sha256
                ),
                "sabdab2_antigen_export_manifest_sha256": export_receipt[
                    "manifest_sha256"
                ],
            },
            "query_manifest_sha256": sha256_file(query_manifest),
            "target_manifest_sha256": sha256_file(target_manifest),
            "query_count": len(query_records),
            "target_count": len(target_records),
            "expected_cross_pair_count": len(query_records) * len(target_records),
            "test_labels_read": False,
            "model_predictions_or_scores_read": False,
            "claim_boundary": (
                "Foldseek input custody only. Test PDB inputs are copied after "
                "development selection; no graph labels, predictions, or metrics are read."
            ),
        }
        (temporary / "PREPARATION_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def finalize_asep_sabdab2_cross_foldseek(
    *,
    preparation_root: Path,
    preparation_receipt_sha256: str,
    raw_alignments: Path,
    raw_alignments_sha256: str,
    command_path: Path,
    foldseek_version: str,
    normalized_output: Path,
    receipt_output: Path,
    audit_tm_threshold: float = 0.50,
    audit_score_rule: str = "reciprocal_min",
) -> dict[str, Any]:
    """Verify every cross pair and freeze a canonical normalized table."""
    if (
        normalized_output.exists()
        or normalized_output.is_symlink()
        or receipt_output.exists()
        or receipt_output.is_symlink()
    ):
        raise FileExistsError("Refusing to overwrite cross Foldseek finalization")
    receipt_path = preparation_root / "PREPARATION_RECEIPT.json"
    query_manifest_path = preparation_root / "QUERY_MANIFEST.csv"
    target_manifest_path = preparation_root / "TARGET_MANIFEST.csv"
    authenticate_file(
        receipt_path, preparation_receipt_sha256, "cross Foldseek preparation receipt"
    )
    preparation = _json(receipt_path, "cross Foldseek preparation receipt")
    authenticate_file(
        query_manifest_path,
        str(preparation.get("query_manifest_sha256", "")),
        "cross Foldseek query manifest",
    )
    authenticate_file(
        target_manifest_path,
        str(preparation.get("target_manifest_sha256", "")),
        "cross Foldseek target manifest",
    )
    if (
        preparation.get("format_version") != CROSS_PREPARATION_FORMAT
        or preparation.get("test_labels_read") is not False
        or preparation.get("model_predictions_or_scores_read") is not False
    ):
        raise AsepCrossFoldseekError("Cross Foldseek preparation is ineligible")
    queries = _csv(
        query_manifest_path,
        {"foldseek_structure_id", "instance_id", "sha256"},
        "cross Foldseek query manifest",
    )
    targets = _csv(
        target_manifest_path,
        {"foldseek_structure_id", "antigen_structure_id", "role", "sha256"},
        "cross Foldseek target manifest",
    )
    query_ids = {row["foldseek_structure_id"] for row in queries}
    target_ids = {row["foldseek_structure_id"] for row in targets}
    expected = len(query_ids) * len(target_ids)
    if (
        len(query_ids) != len(queries)
        or len(target_ids) != len(targets)
        or expected != int(preparation.get("expected_cross_pair_count", -1))
    ):
        raise AsepCrossFoldseekError("Cross Foldseek preparation axes differ")
    authenticate_file(raw_alignments, raw_alignments_sha256, "raw cross alignments")
    command_sha256 = sha256_file(command_path)
    command = command_path.read_text(encoding="utf-8")
    required_command_fragments = (
        "--alignment-type 1",
        "--exhaustive-search 1",
        "-e inf",
        "query,target,qtmscore,ttmscore",
    )
    if any(fragment not in command for fragment in required_command_fragments):
        raise AsepCrossFoldseekError("Foldseek command lacks exhaustive global-TM flags")
    rows: list[dict[str, Any]] = []
    observed: set[tuple[str, str]] = set()
    try:
        with raw_alignments.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(
                handle,
                fieldnames=("query", "target", "qtmscore", "ttmscore"),
                delimiter="\t",
            )
            for row in reader:
                query, target = str(row["query"]), str(row["target"])
                pair = (query, target)
                try:
                    query_tm = float(row["qtmscore"])
                    target_tm = float(row["ttmscore"])
                except (TypeError, ValueError) as error:
                    raise AsepCrossFoldseekError("Foldseek TM scores are not numeric") from error
                if (
                    query not in query_ids
                    or target not in target_ids
                    or pair in observed
                    or not all(
                        math.isfinite(value) and 0 <= value <= 1
                        for value in (query_tm, target_tm)
                    )
                ):
                    raise AsepCrossFoldseekError(
                        "Foldseek row has invalid IDs, scores, or duplication"
                    )
                rows.append(
                    {
                        "query": query,
                        "target": target,
                        "qtmscore": format(query_tm, ".17g"),
                        "ttmscore": format(target_tm, ".17g"),
                    }
                )
                observed.add(pair)
    except OSError as error:
        raise AsepCrossFoldseekError("Cannot read raw cross alignments") from error
    if len(observed) != expected:
        raise AsepCrossFoldseekError(
            f"Foldseek cross grid is incomplete: {len(observed)} != {expected}"
        )
    rows.sort(key=lambda row: (row["query"], row["target"]))
    normalized_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = normalized_output.with_name(
        f".{normalized_output.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("query", "target", "qtmscore", "ttmscore"),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, normalized_output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    authority = preparation["authority"]
    result = {
        "format_version": CROSS_FOLDSEEK_RECEIPT_FORMAT,
        "query_manifest_sha256": authority["label_free_test_manifest_sha256"],
        "target_manifest_sha256": authority["sabdab2_split_manifest_sha256"],
        "target_export_manifest_sha256": authority[
            "sabdab2_antigen_export_manifest_sha256"
        ],
        "preparation_receipt_sha256": preparation_receipt_sha256,
        "preparation_query_manifest_sha256": preparation["query_manifest_sha256"],
        "preparation_target_manifest_sha256": preparation["target_manifest_sha256"],
        "raw_alignments_sha256": raw_alignments_sha256,
        "normalized_alignments_sha256": sha256_file(normalized_output),
        "command_sha256": command_sha256,
        "foldseek_version": foldseek_version,
        "query_role": "test",
        "target_roles": ["train", "validation"],
        "query_count": len(query_ids),
        "target_structure_count": len(target_ids),
        "expected_pair_count": expected,
        "raw_row_count": len(rows),
        "complete_ordered_cross_grid": True,
        "alignment_type": 1,
        "audit_score_rule": audit_score_rule,
        "audit_tm_threshold": audit_tm_threshold,
        "test_labels_read": False,
        "model_predictions_or_scores_read": False,
        "claim_boundary": (
            "Complete label-free cross-structure search authority only; no test "
            "graph labels, model predictions, scores, or performance metrics were read."
        ),
    }
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.write_bytes(canonical_json_bytes(result))
    return result


__all__ = [
    "AsepCrossFoldseekError",
    "CROSS_PREPARATION_FORMAT",
    "finalize_asep_sabdab2_cross_foldseek",
    "prepare_asep_sabdab2_cross_foldseek",
]
