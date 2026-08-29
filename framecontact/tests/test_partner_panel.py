"""Tests for label-free, pre-training wrong-partner selection."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(path: Path, *, add_label: bool = False) -> None:
    rows = []
    for index in range(6):
        row = {
            "instance_id": f"case_{index}",
            "role": "validation",
            "antigen_cluster_id": f"cluster_{index}",
            "antigen_sequence_sha256": hashlib.sha256(
                f"sequence_{index}".encode()
            ).hexdigest(),
            "antigen_length": 100 + 10 * index,
            "surface_residue_count": 40 + 3 * index,
            "antigen_class": "protein",
            "documented_cognate_group_id": "shared" if index in {0, 1} else "",
        }
        if add_label:
            row["epitope_label_count"] = index
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_partner_panel_is_deterministic_matched_and_excludes_known_relations(
    tmp_path: Path,
) -> None:
    from interaction_first.partner_panel import build_partner_panel

    metadata = tmp_path / "metadata.csv"
    _metadata(metadata)
    first = build_partner_panel(
        metadata_path=metadata,
        metadata_sha256=_sha256(metadata),
        output_path=tmp_path / "panel_a.json",
        role="validation",
        panel_size=2,
        protocol_seed=61,
    )
    second = build_partner_panel(
        metadata_path=metadata,
        metadata_sha256=_sha256(metadata),
        output_path=tmp_path / "panel_b.json",
        role="validation",
        panel_size=2,
        protocol_seed=61,
    )
    assert first == second
    assert first["authority"]["task_label_columns_read"] is False
    entries = {entry["instance_id"]: entry for entry in first["entries"]}
    assert "case_1" not in entries["case_0"]["wrong_partner_instance_ids"]
    assert all(len(entry["wrong_partner_instance_ids"]) == 2 for entry in entries.values())


def test_partner_panel_refuses_any_task_label_column(tmp_path: Path) -> None:
    from interaction_first.partner_panel import PartnerPanelError, build_partner_panel

    metadata = tmp_path / "metadata.csv"
    _metadata(metadata, add_label=True)
    with pytest.raises(PartnerPanelError, match="prohibited task columns"):
        build_partner_panel(
            metadata_path=metadata,
            metadata_sha256=_sha256(metadata),
            output_path=tmp_path / "panel.json",
            role="validation",
            panel_size=2,
        )
