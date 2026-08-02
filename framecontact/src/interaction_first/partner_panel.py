"""Label-free, deterministic non-cognate partner panels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .sabdab2_structure import authenticate_file, canonical_json_bytes


PARTNER_PANEL_FORMAT = "framecontact_partner_panel_v1"
REQUIRED_COLUMNS = {
    "instance_id",
    "role",
    "antigen_cluster_id",
    "antigen_sequence_sha256",
    "antigen_length",
    "surface_residue_count",
    "antigen_class",
    "documented_cognate_group_id",
}
PROHIBITED_COLUMN_FRAGMENTS = ("label", "contact", "epitope", "paratope", "prediction")


class PartnerPanelError(ValueError):
    """Raised when a negative panel would be ambiguous or label-dependent."""


def _deciles(values: list[int]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return np.minimum(9, (10 * ranks) // max(1, len(values)))


def build_partner_panel(
    *,
    metadata_path: Path,
    metadata_sha256: str,
    output_path: Path,
    role: str,
    panel_size: int = 31,
    protocol_seed: int = 20_260_730,
) -> dict[str, Any]:
    """Freeze matched wrong-antigen identifiers without reading task labels."""
    if role not in {"train", "validation", "test"}:
        raise ValueError("role must be train, validation, or test")
    if isinstance(panel_size, bool) or not isinstance(panel_size, int) or panel_size <= 0:
        raise ValueError("panel_size must be positive")
    authenticated = authenticate_file(metadata_path, metadata_sha256, "partner metadata")
    try:
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise PartnerPanelError("Partner metadata has no header")
            fieldnames = {str(name) for name in reader.fieldnames}
            prohibited = sorted(
                name
                for name in fieldnames
                if any(fragment in name.casefold() for fragment in PROHIBITED_COLUMN_FRAGMENTS)
            )
            if prohibited:
                raise PartnerPanelError(
                    f"Partner metadata contains prohibited task columns: {prohibited}"
                )
            missing = REQUIRED_COLUMNS - fieldnames
            if missing:
                raise PartnerPanelError(f"Partner metadata lacks columns: {sorted(missing)}")
            rows = [
                {str(key): str(value).strip() for key, value in row.items()}
                for row in reader
                if str(row.get("role", "")).strip() == role
            ]
    except OSError as error:
        raise PartnerPanelError("Cannot read partner metadata") from error
    if len(rows) <= panel_size:
        raise PartnerPanelError("Role has too few instances for the requested panel")
    identifiers = [row["instance_id"] for row in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise PartnerPanelError("Selected instance identifiers must be nonempty and unique")
    lengths: list[int] = []
    surface_counts: list[int] = []
    for row in rows:
        try:
            length = int(row["antigen_length"])
            surface_count = int(row["surface_residue_count"])
        except ValueError as error:
            raise PartnerPanelError("Length and surface counts must be integers") from error
        if length <= 0 or not 0 < surface_count <= length:
            raise PartnerPanelError("Length/surface metadata are out of range")
        if not row["antigen_cluster_id"] or not row["antigen_sequence_sha256"] or not row[
            "antigen_class"
        ]:
            raise PartnerPanelError("Cluster, sequence, and antigen class must be nonempty")
        lengths.append(length)
        surface_counts.append(surface_count)
    length_deciles = _deciles(lengths)
    surface_deciles = _deciles(surface_counts)
    entries = []
    for query_index, query in enumerate(rows):
        candidates = []
        for candidate_index, candidate in enumerate(rows):
            if query_index == candidate_index:
                continue
            if query["antigen_class"] != candidate["antigen_class"]:
                continue
            if query["antigen_cluster_id"] == candidate["antigen_cluster_id"]:
                continue
            if query["antigen_sequence_sha256"] == candidate["antigen_sequence_sha256"]:
                continue
            query_group = query["documented_cognate_group_id"]
            if query_group and query_group == candidate["documented_cognate_group_id"]:
                continue
            tie_digest = hashlib.sha256(
                f"{protocol_seed}\0{query['instance_id']}\0{candidate['instance_id']}".encode(
                    "utf-8"
                )
            ).hexdigest()
            candidates.append(
                (
                    abs(int(length_deciles[query_index]) - int(length_deciles[candidate_index])),
                    abs(
                        int(surface_deciles[query_index])
                        - int(surface_deciles[candidate_index])
                    ),
                    abs(math.log(lengths[query_index] / lengths[candidate_index])),
                    abs(
                        math.log(
                            surface_counts[query_index] / surface_counts[candidate_index]
                        )
                    ),
                    tie_digest,
                    candidate["instance_id"],
                )
            )
        if len(candidates) < panel_size:
            raise PartnerPanelError(
                f"{query['instance_id']}: only {len(candidates)} eligible wrong partners"
            )
        candidates.sort()
        entries.append(
            {
                "instance_id": query["instance_id"],
                "antigen_cluster_id": query["antigen_cluster_id"],
                "antigen_class": query["antigen_class"],
                "antigen_length_decile": int(length_deciles[query_index]),
                "surface_residue_count_decile": int(surface_deciles[query_index]),
                "wrong_partner_instance_ids": [
                    candidate[-1] for candidate in candidates[:panel_size]
                ],
            }
        )
    entries.sort(key=lambda value: value["instance_id"])
    payload = {
        "format_version": PARTNER_PANEL_FORMAT,
        "authority": {
            "metadata_sha256": authenticated,
            "metadata_path": str(metadata_path),
            "task_label_columns_read": False,
            "protocol_seed": protocol_seed,
        },
        "role": role,
        "panel_size": panel_size,
        "matching_priority": [
            "exact_antigen_class",
            "antigen_length_decile",
            "surface_residue_count_decile",
            "relative_antigen_length",
            "relative_surface_residue_count",
            "sha256_tie_break",
        ],
        "exclusions": [
            "self",
            "same_antigen_cluster",
            "same_exact_antigen_sequence",
            "same_nonempty_documented_cognate_group",
        ],
        "entries": entries,
    }
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite partner panel: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["PARTNER_PANEL_FORMAT", "PartnerPanelError", "build_partner_panel"]
