"""Tests for reciprocal, label-free partner counterfactual panels."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(path: Path, *, add_contact_column: bool = False) -> None:
    rows = []
    for index in range(8):
        row = {
            "instance_id": f"case_{index}",
            "role": "validation",
            "antigen_cluster_id": f"ag_cluster_{index}",
            "antigen_sequence_sha256": hashlib.sha256(
                f"ag_sequence_{index}".encode()
            ).hexdigest(),
            "antigen_length": 100 + 10 * index,
            "surface_residue_count": 40 + 3 * index,
            "antigen_class": "protein",
            "antibody_cluster_id": f"ab_cluster_{index}",
            "antibody_sequence_sha256": hashlib.sha256(
                f"ab_sequence_{index}".encode()
            ).hexdigest(),
            "antibody_length": 210 + index,
            "cdr_residue_count": 54 + index,
            "antibody_class": "conventional_fv",
            "documented_cognate_group_id": "shared" if index in {0, 1} else "",
        }
        if add_contact_column:
            row["contact_count"] = index
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_reciprocal_panel_is_deterministic_and_excludes_known_cognates(
    tmp_path: Path,
) -> None:
    from interaction_first.reciprocal_partner_panel import (
        build_reciprocal_partner_panel,
    )

    metadata = tmp_path / "metadata.csv"
    _metadata(metadata)
    first = build_reciprocal_partner_panel(
        metadata_path=metadata,
        metadata_sha256=_sha256(metadata),
        output_path=tmp_path / "a.json",
        role="validation",
        panel_size_per_direction=3,
        protocol_seed=71,
    )
    second = build_reciprocal_partner_panel(
        metadata_path=metadata,
        metadata_sha256=_sha256(metadata),
        output_path=tmp_path / "b.json",
        role="validation",
        panel_size_per_direction=3,
        protocol_seed=71,
    )
    assert first == second
    assert first["authority"]["task_label_columns_read"] is False
    entry = {value["instance_id"]: value for value in first["entries"]}["case_0"]
    assert "case_1" not in entry["wrong_antigen_instance_ids"]
    assert "case_1" not in entry["wrong_antibody_instance_ids"]
    assert len(entry["wrong_antigen_instance_ids"]) == 3
    assert len(entry["wrong_antibody_instance_ids"]) == 3


def test_reciprocal_panel_refuses_task_columns(tmp_path: Path) -> None:
    from interaction_first.reciprocal_partner_panel import (
        ReciprocalPartnerPanelError,
        build_reciprocal_partner_panel,
    )

    metadata = tmp_path / "metadata.csv"
    _metadata(metadata, add_contact_column=True)
    with pytest.raises(ReciprocalPartnerPanelError, match="prohibited task columns"):
        build_reciprocal_partner_panel(
            metadata_path=metadata,
            metadata_sha256=_sha256(metadata),
            output_path=tmp_path / "panel.json",
            role="validation",
            panel_size_per_direction=2,
        )


def test_reciprocal_panel_excludes_documented_cluster_edges(tmp_path: Path) -> None:
    from interaction_first.reciprocal_partner_panel import (
        build_reciprocal_partner_panel,
    )

    metadata = tmp_path / "metadata.csv"
    _metadata(metadata)
    with metadata.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # case_7 documents that antibody cluster 0 also binds antigen cluster 2.
    # case_1 documents that antibody cluster 1 binds antigen cluster 0.
    rows[7]["antibody_cluster_id"] = rows[0]["antibody_cluster_id"]
    rows[7]["antigen_cluster_id"] = rows[2]["antigen_cluster_id"]
    rows[7]["role"] = "train"
    rows[1]["antigen_cluster_id"] = rows[0]["antigen_cluster_id"]
    rows[0]["documented_cognate_group_id"] = ""
    rows[1]["documented_cognate_group_id"] = ""
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = build_reciprocal_partner_panel(
        metadata_path=metadata,
        metadata_sha256=_sha256(metadata),
        output_path=tmp_path / "panel.json",
        role="validation",
        panel_size_per_direction=2,
        protocol_seed=71,
    )
    entry = {value["instance_id"]: value for value in result["entries"]}["case_0"]
    assert "case_2" not in entry["wrong_antigen_instance_ids"]
    assert "case_1" not in entry["wrong_antibody_instance_ids"]
