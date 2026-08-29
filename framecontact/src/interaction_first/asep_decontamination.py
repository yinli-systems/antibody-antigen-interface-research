"""Label-free SAbDab2-to-AsEP test contamination audit."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .asep_frozen_test import LABEL_FREE_TEST_FORMAT
from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file
from .structural_novelty import symmetric_tm_score


DECONTAMINATION_FORMAT = "framecontact_asep_sabdab2_decontamination_v1"
CROSS_FOLDSEEK_RECEIPT_FORMAT = (
    "framecontact_asep_sabdab2_cross_foldseek_receipt_v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PDB_ID = re.compile(r"[0-9a-z]{4}")


class AsepDecontaminationError(ValueError):
    """Raised when cross-dataset contamination authority is incomplete."""


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AsepDecontaminationError(f"Cannot parse {name}") from error
    if not isinstance(payload, dict):
        raise AsepDecontaminationError(f"{name} must be a JSON object")
    return payload


def _csv(path: Path, required: set[str], name: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise AsepDecontaminationError(f"{name} columns differ")
            return [
                {str(key): str(value).strip() for key, value in row.items()}
                for row in reader
            ]
    except OSError as error:
        raise AsepDecontaminationError(f"Cannot read {name}") from error


def _canonical_sabdab_pdb_id(assembly_id: str) -> str:
    if not assembly_id.startswith("pdb_") or len(assembly_id) < 8:
        raise AsepDecontaminationError("SAbDab2 PDB assembly identifier is malformed")
    identifier = assembly_id[-4:].lower()
    if _PDB_ID.fullmatch(identifier) is None:
        raise AsepDecontaminationError("SAbDab2 canonical PDB identifier is malformed")
    return identifier


def audit_asep_sabdab2_decontamination(
    *,
    label_free_test_root: Path,
    label_free_test_receipt_sha256: str,
    sabdab2_split_manifest: Path,
    sabdab2_split_manifest_sha256: str,
    sabdab2_antigen_export_manifest: Path,
    sabdab2_antigen_export_manifest_sha256: str,
    cross_alignments: Path,
    cross_alignments_sha256: str,
    cross_foldseek_receipt: Path,
    cross_foldseek_receipt_sha256: str,
    output_root: Path,
    tm_threshold: float = 0.50,
    foldseek_score_rule: str = "reciprocal_min",
) -> dict[str, Any]:
    """Freeze contamination flags before graph-label or prediction access."""
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite AsEP decontamination audit: {output_root}"
        )
    if not math.isfinite(tm_threshold) or not 0 < tm_threshold <= 1:
        raise ValueError("tm_threshold must lie in (0, 1]")
    symmetric_tm_score(0.0, 0.0, foldseek_score_rule)
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
        or label_receipt.get("checks", {}).get(
            "official_test_contact_or_residue_labels_read"
        )
        is not False
        or label_receipt.get("checks", {}).get("model_predictions_or_scores_read")
        is not False
    ):
        raise AsepDecontaminationError("AsEP label-free test authority is ineligible")
    asep_rows = _csv(
        label_manifest_path,
        {
            "instance_id",
            "role",
            "pdb_id",
            "antigen_pdb_seqres_sha256",
            "antibody_pdb_seqres_sha256",
            "antigen_structure_path",
            "antigen_structure_sha256",
            "foldseek_structure_id",
        },
        "AsEP label-free test manifest",
    )
    if (
        len(asep_rows) != int(label_receipt.get("test_instance_count", -1))
        or any(row["role"] != "test" for row in asep_rows)
        or len({row["instance_id"] for row in asep_rows}) != len(asep_rows)
    ):
        raise AsepDecontaminationError("AsEP label-free test axis differs")
    for row in asep_rows:
        if (
            _PDB_ID.fullmatch(row["pdb_id"]) is None
            or _SHA256.fullmatch(row["antigen_pdb_seqres_sha256"]) is None
            or _SHA256.fullmatch(row["antibody_pdb_seqres_sha256"]) is None
        ):
            raise AsepDecontaminationError("AsEP label-free metadata is malformed")
        structure = label_free_test_root / row["antigen_structure_path"]
        authenticate_file(
            structure, row["antigen_structure_sha256"], "AsEP antigen structure"
        )
    authenticate_file(
        sabdab2_split_manifest,
        sabdab2_split_manifest_sha256,
        "SAbDab2 split manifest",
    )
    sab_rows_all = _csv(
        sabdab2_split_manifest,
        {
            "instance_id",
            "role",
            "antigen_structure_id",
            "antigen_sequence_sha256",
            "antibody_sequence_sha256",
            "pdb_assembly_id",
        },
        "SAbDab2 split manifest",
    )
    sab_rows = [row for row in sab_rows_all if row["role"] in {"train", "validation"}]
    if not sab_rows or any(
        _SHA256.fullmatch(row["antigen_sequence_sha256"]) is None
        or _SHA256.fullmatch(row["antibody_sequence_sha256"]) is None
        for row in sab_rows
    ):
        raise AsepDecontaminationError("SAbDab2 development authority is malformed")
    target_structures = {row["antigen_structure_id"] for row in sab_rows}
    if len(target_structures) != len(sab_rows):
        raise AsepDecontaminationError(
            "SAbDab2 development antigen structure IDs must be unique"
        )
    sab_pdb_ids = {_canonical_sabdab_pdb_id(row["pdb_assembly_id"]) for row in sab_rows}
    sab_antigen_sequences = {row["antigen_sequence_sha256"] for row in sab_rows}
    sab_antibody_sequences = {row["antibody_sequence_sha256"] for row in sab_rows}

    authenticate_file(
        sabdab2_antigen_export_manifest,
        sabdab2_antigen_export_manifest_sha256,
        "SAbDab2 antigen export manifest",
    )
    export_rows = _csv(
        sabdab2_antigen_export_manifest,
        {"antigen_structure_id", "foldseek_structure_id", "relative_path", "sha256"},
        "SAbDab2 antigen export manifest",
    )
    foldseek_target_by_structure = {
        row["antigen_structure_id"]: row["foldseek_structure_id"]
        for row in export_rows
        if row["antigen_structure_id"] in target_structures
    }
    if (
        set(foldseek_target_by_structure) != target_structures
        or len(set(foldseek_target_by_structure.values())) != len(target_structures)
        or any(
            _SHA256.fullmatch(value) is None
            for value in foldseek_target_by_structure.values()
        )
    ):
        raise AsepDecontaminationError(
            "SAbDab2 export does not uniquely cover development structures"
        )
    target_by_foldseek = {
        foldseek_id: structure_id
        for structure_id, foldseek_id in foldseek_target_by_structure.items()
    }

    authenticate_file(
        cross_alignments, cross_alignments_sha256, "cross-dataset Foldseek alignments"
    )
    authenticate_file(
        cross_foldseek_receipt,
        cross_foldseek_receipt_sha256,
        "cross-dataset Foldseek receipt",
    )
    foldseek = _json(cross_foldseek_receipt, "cross-dataset Foldseek receipt")
    query_by_foldseek = {
        row["foldseek_structure_id"]: row["instance_id"] for row in asep_rows
    }
    if (
        len(query_by_foldseek) != len(asep_rows)
        or any(_SHA256.fullmatch(value) is None for value in query_by_foldseek)
    ):
        raise AsepDecontaminationError("AsEP Foldseek identifiers are invalid")
    query_ids = set(query_by_foldseek)
    target_ids = set(target_by_foldseek)
    expected_pairs = len(query_ids) * len(target_ids)
    if (
        foldseek.get("format_version") != CROSS_FOLDSEEK_RECEIPT_FORMAT
        or foldseek.get("query_manifest_sha256") != label_receipt["manifest_sha256"]
        or foldseek.get("target_manifest_sha256") != sabdab2_split_manifest_sha256
        or foldseek.get("target_export_manifest_sha256")
        != sabdab2_antigen_export_manifest_sha256
        or foldseek.get("normalized_alignments_sha256") != cross_alignments_sha256
        or foldseek.get("audit_score_rule") != foldseek_score_rule
        or not math.isclose(
            float(foldseek.get("audit_tm_threshold", -1.0)),
            tm_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or foldseek.get("query_role") != "test"
        or foldseek.get("target_roles") != ["train", "validation"]
        or foldseek.get("complete_ordered_cross_grid") is not True
        or int(foldseek.get("query_count", -1)) != len(query_ids)
        or int(foldseek.get("target_structure_count", -1)) != len(target_ids)
        or int(foldseek.get("expected_pair_count", -1)) != expected_pairs
        or int(foldseek.get("raw_row_count", -1)) != expected_pairs
        or int(foldseek.get("alignment_type", -1)) != 1
        or foldseek.get("test_labels_read") is not False
        or foldseek.get("model_predictions_or_scores_read") is not False
    ):
        raise AsepDecontaminationError(
            "Cross-dataset Foldseek receipt does not prove the required grid"
        )
    maximum_score = {identifier: 0.0 for identifier in query_ids}
    hard_targets: dict[str, list[str]] = {identifier: [] for identifier in query_ids}
    observed: set[tuple[str, str]] = set()
    try:
        with cross_alignments.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"query", "target", "qtmscore", "ttmscore"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise AsepDecontaminationError("Cross Foldseek table columns differ")
            for row in reader:
                query, target = str(row["query"]), str(row["target"])
                pair = (query, target)
                if (
                    query not in query_ids
                    or target not in target_ids
                    or pair in observed
                ):
                    raise AsepDecontaminationError(
                        "Cross Foldseek table contains an unknown or duplicate pair"
                    )
                try:
                    score = symmetric_tm_score(
                        float(row["qtmscore"]),
                        float(row["ttmscore"]),
                        foldseek_score_rule,
                    )
                except ValueError as error:
                    raise AsepDecontaminationError(
                        "Cross Foldseek table has invalid TM scores"
                    ) from error
                maximum_score[query] = max(maximum_score[query], score)
                if score >= tm_threshold:
                    hard_targets[query].append(target_by_foldseek[target])
                observed.add(pair)
    except OSError as error:
        raise AsepDecontaminationError("Cannot read cross Foldseek table") from error
    if len(observed) != expected_pairs:
        raise AsepDecontaminationError("Cross Foldseek table is not a complete grid")

    records = []
    for row in asep_rows:
        query = row["foldseek_structure_id"]
        flags = {
            "exact_pdb_identity": row["pdb_id"] in sab_pdb_ids,
            "exact_antigen_pdb_seqres": row["antigen_pdb_seqres_sha256"]
            in sab_antigen_sequences,
            "exact_antibody_pdb_seqres": row["antibody_pdb_seqres_sha256"]
            in sab_antibody_sequences,
            "antigen_structural_similarity": bool(hard_targets[query]),
        }
        contaminated = any(flags.values())
        records.append(
            {
                "instance_id": row["instance_id"],
                "contaminated": contaminated,
                "eligible_for_decontaminated_primary": not contaminated,
                "flags": flags,
                "maximum_reciprocal_min_tm_score": maximum_score[query],
                "structural_hard_target_count": len(hard_targets[query]),
                "structural_hard_target_examples": sorted(hard_targets[query])[:20],
            }
        )
    flag_counts = {
        flag: sum(record["flags"][flag] for record in records)
        for flag in (
            "exact_pdb_identity",
            "exact_antigen_pdb_seqres",
            "exact_antibody_pdb_seqres",
            "antigen_structural_similarity",
        )
    }
    clean_ids = [
        record["instance_id"]
        for record in records
        if record["eligible_for_decontaminated_primary"]
    ]
    payload = {
        "format_version": DECONTAMINATION_FORMAT,
        "policy": {
            "reference_roles": ["SAbDab2 train", "SAbDab2 validation"],
            "contaminated_if_any": list(flag_counts),
            "antigen_structural_rule": (
                f"{foldseek_score_rule}(query_tm,target_tm) >= {tm_threshold:.2f}"
            ),
            "freeze_time_use": (
                "Report the full official test and the pre-flagged decontaminated "
                "subset separately; never choose exclusions from test performance."
            ),
        },
        "authority": {
            "label_free_test_receipt_sha256": label_free_test_receipt_sha256,
            "label_free_test_manifest_sha256": label_receipt["manifest_sha256"],
            "sabdab2_split_manifest_sha256": sabdab2_split_manifest_sha256,
            "sabdab2_antigen_export_manifest_sha256": (
                sabdab2_antigen_export_manifest_sha256
            ),
            "cross_alignments_sha256": cross_alignments_sha256,
            "cross_foldseek_receipt_sha256": cross_foldseek_receipt_sha256,
            "test_graph_tensors_opened": False,
            "test_labels_read": False,
            "model_predictions_or_scores_read": False,
        },
        "counts": {
            "asep_test_instances": len(records),
            "sabdab2_development_instances": len(sab_rows),
            "cross_structure_pairs": expected_pairs,
            "contaminated_instances": len(records) - len(clean_ids),
            "decontaminated_primary_instances": len(clean_ids),
            "flag_counts": flag_counts,
        },
        "decontaminated_primary_instance_ids": clean_ids,
        "records": records,
        "sequence_comparison_boundary": (
            "AsEP hashes are PDB-SEQRES hashes and may be deletion-only views of "
            "official graph sequences. Exact PDB identity and reciprocal-min structural "
            "similarity provide independent conservative controls."
        ),
        "claim_boundary": (
            "Label-free contamination audit only. No AsEP test graph tensor, label, "
            "model prediction, score, or performance metric was read."
        ),
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        result_path = temporary / "ASEP_SABDAB2_DECONTAMINATION.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        receipt = {
            "format_version": DECONTAMINATION_FORMAT,
            "result_sha256": sha256_file(result_path),
            "test_labels_read": False,
            "model_predictions_or_scores_read": False,
            "decontaminated_primary_instance_count": len(clean_ids),
        }
        (temporary / "DECONTAMINATION_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "AsepDecontaminationError",
    "CROSS_FOLDSEEK_RECEIPT_FORMAT",
    "DECONTAMINATION_FORMAT",
    "audit_asep_sabdab2_decontamination",
]
