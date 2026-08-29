"""Label-free reciprocal partner panels for the frozen AsEP test role."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .asep_decontamination import DECONTAMINATION_FORMAT
from .asep_frozen_test import LABEL_FREE_TEST_FORMAT
from .reciprocal_partner_panel import RECIPROCAL_PARTNER_PANEL_FORMAT
from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file


class AsepTestPartnerPanelError(ValueError):
    """Raised when a label-free external partner panel cannot be frozen."""


def _deciles(values: list[int]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return np.minimum(9, (10 * ranks) // max(1, len(values)))


def _tie(seed: int, direction: str, focal: str, candidate: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{direction}\0{focal}\0{candidate}".encode("utf-8")
    ).hexdigest()


def build_asep_label_free_test_partner_panel(
    *,
    label_free_test_root: Path,
    label_free_test_receipt_sha256: str,
    output_path: Path,
    panel_size_per_direction: int = 31,
    protocol_seed: int = 20_260_730,
    decontamination_root: Path | None = None,
    decontamination_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze full-test or pre-flagged-clean counterfactual partners."""
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite AsEP test panel: {output_path}")
    if (
        isinstance(panel_size_per_direction, bool)
        or not isinstance(panel_size_per_direction, int)
        or panel_size_per_direction <= 0
    ):
        raise ValueError("panel_size_per_direction must be positive")
    receipt_path = label_free_test_root / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json"
    manifest_path = label_free_test_root / "ASEP_LABEL_FREE_TEST_METADATA.csv"
    authenticate_file(
        receipt_path, label_free_test_receipt_sha256, "AsEP label-free test receipt"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authenticate_file(
        manifest_path,
        str(receipt.get("manifest_sha256", "")),
        "AsEP label-free test metadata",
    )
    if (
        receipt.get("format_version") != LABEL_FREE_TEST_FORMAT
        or receipt.get("test_label_open_count") != 0
        or receipt.get("checks", {}).get("official_test_graph_tensors_opened")
        is not False
        or receipt.get("checks", {}).get(
            "official_test_contact_or_residue_labels_read"
        )
        is not False
    ):
        raise AsepTestPartnerPanelError("Label-free test metadata is ineligible")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "instance_id",
            "role",
            "pdb_id",
            "antigen_pdb_seqres_sha256",
            "antibody_pdb_seqres_sha256",
            "antigen_pdb_seqres_length",
            "antibody_pdb_seqres_length",
            "antigen_observed_residue_count",
            "antibody_observed_residue_count",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AsepTestPartnerPanelError("Label-free test metadata columns differ")
        rows = [
            {str(key): str(value).strip() for key, value in row.items()}
            for row in reader
        ]
    if (
        len(rows) != int(receipt.get("test_instance_count", -1))
        or any(row["role"] != "test" for row in rows)
        or len({row["instance_id"] for row in rows}) != len(rows)
    ):
        raise AsepTestPartnerPanelError("Label-free test instance axis differs")

    decontamination_sha = None
    panel_scope = "full_official_test"
    if (decontamination_root is None) != (decontamination_receipt_sha256 is None):
        raise ValueError("Decontamination root and receipt SHA must be provided together")
    if decontamination_root is not None:
        decontamination_receipt = decontamination_root / "DECONTAMINATION_RECEIPT.json"
        decontamination_result = (
            decontamination_root / "ASEP_SABDAB2_DECONTAMINATION.json"
        )
        authenticate_file(
            decontamination_receipt,
            str(decontamination_receipt_sha256),
            "AsEP decontamination receipt",
        )
        decon_receipt = json.loads(decontamination_receipt.read_text(encoding="utf-8"))
        authenticate_file(
            decontamination_result,
            str(decon_receipt.get("result_sha256", "")),
            "AsEP decontamination result",
        )
        decon = json.loads(decontamination_result.read_text(encoding="utf-8"))
        if (
            decon_receipt.get("format_version") != DECONTAMINATION_FORMAT
            or decon.get("format_version") != DECONTAMINATION_FORMAT
            or decon_receipt.get("test_labels_read") is not False
            or decon_receipt.get("model_predictions_or_scores_read") is not False
            or decon.get("authority", {}).get("test_labels_read") is not False
            or decon.get("authority", {}).get("model_predictions_or_scores_read")
            is not False
            or decon.get("authority", {}).get("label_free_test_manifest_sha256")
            != receipt["manifest_sha256"]
        ):
            raise AsepTestPartnerPanelError("Decontamination authority is ineligible")
        eligible = set(decon.get("decontaminated_primary_instance_ids", []))
        rows = [row for row in rows if row["instance_id"] in eligible]
        if len(rows) != int(
            decon.get("counts", {}).get("decontaminated_primary_instances", -1)
        ):
            raise AsepTestPartnerPanelError("Decontaminated instance axis differs")
        decontamination_sha = decon_receipt["result_sha256"]
        panel_scope = "sabdab2_decontaminated_primary"
    if len(rows) <= panel_size_per_direction:
        raise AsepTestPartnerPanelError(
            "Panel scope has too few instances for the requested candidates"
        )

    def integers(name: str) -> list[int]:
        values = []
        for row in rows:
            try:
                value = int(row[name])
            except ValueError as error:
                raise AsepTestPartnerPanelError(f"{name} must be integer") from error
            if value <= 0:
                raise AsepTestPartnerPanelError(f"{name} must be positive")
            values.append(value)
        return values

    antigen_lengths = integers("antigen_pdb_seqres_length")
    antibody_lengths = integers("antibody_pdb_seqres_length")
    antigen_observed = integers("antigen_observed_residue_count")
    antibody_observed = integers("antibody_observed_residue_count")
    antigen_deciles = _deciles(antigen_lengths)
    antibody_deciles = _deciles(antibody_lengths)
    antigen_observed_deciles = _deciles(antigen_observed)
    antibody_observed_deciles = _deciles(antibody_observed)

    entries = []
    for query_index, query in enumerate(rows):
        wrong_antigens = []
        wrong_antibodies = []
        for candidate_index, candidate in enumerate(rows):
            if query_index == candidate_index or query["pdb_id"] == candidate["pdb_id"]:
                continue
            if query["antigen_pdb_seqres_sha256"] != candidate[
                "antigen_pdb_seqres_sha256"
            ]:
                wrong_antigens.append(
                    (
                        abs(
                            int(antigen_deciles[query_index])
                            - int(antigen_deciles[candidate_index])
                        ),
                        abs(
                            int(antigen_observed_deciles[query_index])
                            - int(antigen_observed_deciles[candidate_index])
                        ),
                        abs(
                            math.log(
                                antigen_lengths[query_index]
                                / antigen_lengths[candidate_index]
                            )
                        ),
                        abs(
                            math.log(
                                antigen_observed[query_index]
                                / antigen_observed[candidate_index]
                            )
                        ),
                        _tie(
                            protocol_seed,
                            "wrong_antigen",
                            query["instance_id"],
                            candidate["instance_id"],
                        ),
                        candidate["instance_id"],
                    )
                )
            if query["antibody_pdb_seqres_sha256"] != candidate[
                "antibody_pdb_seqres_sha256"
            ]:
                wrong_antibodies.append(
                    (
                        abs(
                            int(antibody_deciles[query_index])
                            - int(antibody_deciles[candidate_index])
                        ),
                        abs(
                            int(antibody_observed_deciles[query_index])
                            - int(antibody_observed_deciles[candidate_index])
                        ),
                        abs(
                            math.log(
                                antibody_lengths[query_index]
                                / antibody_lengths[candidate_index]
                            )
                        ),
                        abs(
                            math.log(
                                antibody_observed[query_index]
                                / antibody_observed[candidate_index]
                            )
                        ),
                        _tie(
                            protocol_seed,
                            "wrong_antibody",
                            query["instance_id"],
                            candidate["instance_id"],
                        ),
                        candidate["instance_id"],
                    )
                )
        wrong_antigens.sort()
        wrong_antibodies.sort()
        if len(wrong_antigens) < panel_size_per_direction or len(
            wrong_antibodies
        ) < panel_size_per_direction:
            raise AsepTestPartnerPanelError(
                f"{query['instance_id']}: insufficient label-free counterfactuals"
            )
        entries.append(
            {
                "instance_id": query["instance_id"],
                "antigen_cluster_id": (
                    f"pdbseq:{query['antigen_pdb_seqres_sha256']}"
                ),
                "antibody_cluster_id": (
                    f"pdbseq:{query['antibody_pdb_seqres_sha256']}"
                ),
                "wrong_antigen_instance_ids": [
                    value[-1]
                    for value in wrong_antigens[:panel_size_per_direction]
                ],
                "wrong_antibody_instance_ids": [
                    value[-1]
                    for value in wrong_antibodies[:panel_size_per_direction]
                ],
            }
        )
    entries.sort(key=lambda entry: entry["instance_id"])
    payload = {
        "format_version": RECIPROCAL_PARTNER_PANEL_FORMAT,
        "authority": {
            "label_free_test_receipt_sha256": label_free_test_receipt_sha256,
            "label_free_test_manifest_sha256": receipt["manifest_sha256"],
            "decontamination_result_sha256": decontamination_sha,
            "task_label_columns_read": False,
            "official_test_graph_tensors_opened": False,
            "model_predictions_or_scores_read": False,
            "protocol_seed": protocol_seed,
            "panel_scope": panel_scope,
        },
        "role": "test",
        "panel_size_per_direction": panel_size_per_direction,
        "directions": {
            "fixed_antibody_wrong_antigen": {
                "match": [
                    "antigen_PDB_SEQRES_length_decile",
                    "antigen_observed_residue_count_decile",
                ],
                "exclude": [
                    "same_instance",
                    "same_PDB_identity",
                    "same_exact_antigen_PDB_SEQRES",
                ],
            },
            "fixed_antigen_wrong_antibody": {
                "match": [
                    "antibody_PDB_SEQRES_length_decile",
                    "antibody_observed_residue_count_decile",
                ],
                "exclude": [
                    "same_instance",
                    "same_PDB_identity",
                    "same_exact_antibody_PDB_SEQRES",
                ],
            },
        },
        "unknown_binding_boundary": (
            "Counterfactual partners are ranking controls selected without binding "
            "labels. They are not asserted to be experimentally non-binding negatives."
        ),
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(payload))
    return {**payload, "sha256": sha256_file(output_path)}


__all__ = [
    "AsepTestPartnerPanelError",
    "build_asep_label_free_test_partner_panel",
]
