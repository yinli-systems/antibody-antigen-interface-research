"""Label-free reciprocal non-cognate panels for partner-specific evaluation.

The legacy panel keeps the focal antibody fixed and replaces the antigen.  That
tests one direction of partner specificity, but it cannot answer whether the
same antigen receives a different epitope prediction for different antibodies.
This module freezes both counterfactual directions before training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .sabdab2_structure import authenticate_file, canonical_json_bytes


RECIPROCAL_PARTNER_PANEL_FORMAT = "framecontact_reciprocal_partner_panel_v2"
PROHIBITED_COLUMN_FRAGMENTS = (
    "label",
    "contact",
    "epitope",
    "paratope",
    "prediction",
)
REQUIRED_COLUMNS = {
    "instance_id",
    "role",
    "antigen_cluster_id",
    "antigen_sequence_sha256",
    "antigen_length",
    "surface_residue_count",
    "antigen_class",
    "antibody_cluster_id",
    "antibody_sequence_sha256",
    "antibody_length",
    "cdr_residue_count",
    "antibody_class",
    "documented_cognate_group_id",
}


class ReciprocalPartnerPanelError(ValueError):
    """Raised when either counterfactual direction is ambiguous or unsafe."""


def _deciles(values: list[int]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return np.minimum(9, (10 * ranks) // max(1, len(values)))


def _integer_column(rows: list[dict[str, str]], name: str) -> list[int]:
    values = []
    for row in rows:
        try:
            value = int(row[name])
        except ValueError as error:
            raise ReciprocalPartnerPanelError(f"{name} must contain integers") from error
        if value <= 0:
            raise ReciprocalPartnerPanelError(f"{name} must be positive")
        values.append(value)
    return values


def _tie_digest(seed: int, direction: str, focal: str, candidate: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{direction}\0{focal}\0{candidate}".encode("utf-8")
    ).hexdigest()


def _eligible_common(query: dict[str, str], candidate: dict[str, str]) -> bool:
    query_group = query["documented_cognate_group_id"]
    return not (
        query_group
        and query_group == candidate["documented_cognate_group_id"]
    )


def _wrong_antigen_candidates(
    rows: list[dict[str, str]],
    query_index: int,
    *,
    length_deciles: np.ndarray,
    surface_deciles: np.ndarray,
    lengths: list[int],
    surface_counts: list[int],
    seed: int,
    documented_cluster_edges: set[tuple[str, str]],
) -> list[tuple[Any, ...]]:
    query = rows[query_index]
    candidates = []
    for candidate_index, candidate in enumerate(rows):
        if query_index == candidate_index or not _eligible_common(query, candidate):
            continue
        if query["antigen_class"] != candidate["antigen_class"]:
            continue
        if query["antigen_cluster_id"] == candidate["antigen_cluster_id"]:
            continue
        if query["antigen_sequence_sha256"] == candidate["antigen_sequence_sha256"]:
            continue
        if (
            query["antibody_cluster_id"],
            candidate["antigen_cluster_id"],
        ) in documented_cluster_edges:
            continue
        candidates.append(
            (
                abs(int(length_deciles[query_index]) - int(length_deciles[candidate_index])),
                abs(int(surface_deciles[query_index]) - int(surface_deciles[candidate_index])),
                abs(math.log(lengths[query_index] / lengths[candidate_index])),
                abs(math.log(surface_counts[query_index] / surface_counts[candidate_index])),
                _tie_digest(
                    seed,
                    "wrong_antigen",
                    query["instance_id"],
                    candidate["instance_id"],
                ),
                candidate["instance_id"],
            )
        )
    return sorted(candidates)


def _wrong_antibody_candidates(
    rows: list[dict[str, str]],
    query_index: int,
    *,
    length_deciles: np.ndarray,
    cdr_deciles: np.ndarray,
    lengths: list[int],
    cdr_counts: list[int],
    seed: int,
    documented_cluster_edges: set[tuple[str, str]],
) -> list[tuple[Any, ...]]:
    query = rows[query_index]
    candidates = []
    for candidate_index, candidate in enumerate(rows):
        if query_index == candidate_index or not _eligible_common(query, candidate):
            continue
        if query["antibody_class"] != candidate["antibody_class"]:
            continue
        if query["antibody_cluster_id"] == candidate["antibody_cluster_id"]:
            continue
        if query["antibody_sequence_sha256"] == candidate["antibody_sequence_sha256"]:
            continue
        if (
            candidate["antibody_cluster_id"],
            query["antigen_cluster_id"],
        ) in documented_cluster_edges:
            continue
        candidates.append(
            (
                abs(int(length_deciles[query_index]) - int(length_deciles[candidate_index])),
                abs(int(cdr_deciles[query_index]) - int(cdr_deciles[candidate_index])),
                abs(math.log(lengths[query_index] / lengths[candidate_index])),
                abs(math.log(cdr_counts[query_index] / cdr_counts[candidate_index])),
                _tie_digest(
                    seed,
                    "wrong_antibody",
                    query["instance_id"],
                    candidate["instance_id"],
                ),
                candidate["instance_id"],
            )
        )
    return sorted(candidates)


def build_reciprocal_partner_panel(
    *,
    metadata_path: Path,
    metadata_sha256: str,
    output_path: Path,
    role: str,
    panel_size_per_direction: int = 31,
    protocol_seed: int = 20_260_730,
) -> dict[str, Any]:
    """Freeze fixed-antibody and fixed-antigen counterfactual panels."""
    if role not in {"train", "validation", "test"}:
        raise ValueError("role must be train, validation, or test")
    if (
        isinstance(panel_size_per_direction, bool)
        or not isinstance(panel_size_per_direction, int)
        or panel_size_per_direction <= 0
    ):
        raise ValueError("panel_size_per_direction must be positive")
    authenticated = authenticate_file(
        metadata_path, metadata_sha256, "reciprocal partner metadata"
    )
    try:
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ReciprocalPartnerPanelError("Partner metadata has no header")
            fieldnames = {str(name) for name in reader.fieldnames}
            prohibited = sorted(
                name
                for name in fieldnames
                if any(
                    fragment in name.casefold()
                    for fragment in PROHIBITED_COLUMN_FRAGMENTS
                )
            )
            if prohibited:
                raise ReciprocalPartnerPanelError(
                    f"Partner metadata contains prohibited task columns: {prohibited}"
                )
            missing = REQUIRED_COLUMNS - fieldnames
            if missing:
                raise ReciprocalPartnerPanelError(
                    f"Partner metadata lacks columns: {sorted(missing)}"
                )
            all_rows = [
                {str(key): str(value).strip() for key, value in row.items()}
                for row in reader
            ]
    except OSError as error:
        raise ReciprocalPartnerPanelError("Cannot read partner metadata") from error
    rows = [row for row in all_rows if row["role"] == role]
    if len(rows) <= panel_size_per_direction:
        raise ReciprocalPartnerPanelError(
            "Role has too few instances for the requested reciprocal panel"
        )
    identifiers = [row["instance_id"] for row in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ReciprocalPartnerPanelError(
            "Selected instance identifiers must be nonempty and unique"
        )
    for row in rows:
        for name in (
            "antigen_cluster_id",
            "antigen_sequence_sha256",
            "antigen_class",
            "antibody_cluster_id",
            "antibody_sequence_sha256",
            "antibody_class",
        ):
            if not row[name]:
                raise ReciprocalPartnerPanelError(f"{name} must be nonempty")

    antigen_lengths = _integer_column(rows, "antigen_length")
    surface_counts = _integer_column(rows, "surface_residue_count")
    antibody_lengths = _integer_column(rows, "antibody_length")
    cdr_counts = _integer_column(rows, "cdr_residue_count")
    if any(surface > length for surface, length in zip(surface_counts, antigen_lengths)):
        raise ReciprocalPartnerPanelError(
            "surface_residue_count cannot exceed antigen_length"
        )
    if any(cdr > length for cdr, length in zip(cdr_counts, antibody_lengths)):
        raise ReciprocalPartnerPanelError(
            "cdr_residue_count cannot exceed antibody_length"
        )

    antigen_length_deciles = _deciles(antigen_lengths)
    surface_deciles = _deciles(surface_counts)
    antibody_length_deciles = _deciles(antibody_lengths)
    cdr_deciles = _deciles(cdr_counts)
    documented_cluster_edges = {
        (row["antibody_cluster_id"], row["antigen_cluster_id"])
        for row in all_rows
        if row["antibody_cluster_id"] and row["antigen_cluster_id"]
    }
    entries = []
    for query_index, query in enumerate(rows):
        wrong_antigens = _wrong_antigen_candidates(
            rows,
            query_index,
            length_deciles=antigen_length_deciles,
            surface_deciles=surface_deciles,
            lengths=antigen_lengths,
            surface_counts=surface_counts,
            seed=protocol_seed,
            documented_cluster_edges=documented_cluster_edges,
        )
        wrong_antibodies = _wrong_antibody_candidates(
            rows,
            query_index,
            length_deciles=antibody_length_deciles,
            cdr_deciles=cdr_deciles,
            lengths=antibody_lengths,
            cdr_counts=cdr_counts,
            seed=protocol_seed,
            documented_cluster_edges=documented_cluster_edges,
        )
        if len(wrong_antigens) < panel_size_per_direction:
            raise ReciprocalPartnerPanelError(
                f"{query['instance_id']}: only {len(wrong_antigens)} eligible wrong antigens"
            )
        if len(wrong_antibodies) < panel_size_per_direction:
            raise ReciprocalPartnerPanelError(
                f"{query['instance_id']}: only {len(wrong_antibodies)} eligible wrong antibodies"
            )
        entries.append(
            {
                "instance_id": query["instance_id"],
                "antigen_cluster_id": query["antigen_cluster_id"],
                "antibody_cluster_id": query["antibody_cluster_id"],
                "wrong_antigen_instance_ids": [
                    value[-1] for value in wrong_antigens[:panel_size_per_direction]
                ],
                "wrong_antibody_instance_ids": [
                    value[-1] for value in wrong_antibodies[:panel_size_per_direction]
                ],
            }
        )
    entries.sort(key=lambda value: value["instance_id"])
    payload = {
        "format_version": RECIPROCAL_PARTNER_PANEL_FORMAT,
        "authority": {
            "metadata_sha256": authenticated,
            "metadata_path": str(metadata_path),
            "task_label_columns_read": False,
            "protocol_seed": protocol_seed,
            "documented_cluster_edges_scope": "complete_metadata_cohort",
        },
        "role": role,
        "panel_size_per_direction": panel_size_per_direction,
        "directions": {
            "fixed_antibody_wrong_antigen": {
                "match": [
                    "antigen_class",
                    "antigen_length_decile",
                    "surface_residue_count_decile",
                ],
                "exclude": [
                    "same_antigen_cluster",
                    "same_exact_antigen_sequence",
                    "same_documented_cognate_group",
                    "documented_antibody_cluster_antigen_cluster_edge",
                ],
            },
            "fixed_antigen_wrong_antibody": {
                "match": [
                    "antibody_class",
                    "antibody_length_decile",
                    "cdr_residue_count_decile",
                ],
                "exclude": [
                    "same_antibody_cluster",
                    "same_exact_antibody_sequence",
                    "same_documented_cognate_group",
                    "documented_antibody_cluster_antigen_cluster_edge",
                ],
            },
        },
        "entries": entries,
    }
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite reciprocal panel: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "RECIPROCAL_PARTNER_PANEL_FORMAT",
    "ReciprocalPartnerPanelError",
    "build_reciprocal_partner_panel",
]
