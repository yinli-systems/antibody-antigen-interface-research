"""Atomic matrix tests for split-bound partner panels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_split_panel_matrix_freezes_all_nine_panels(tmp_path: Path) -> None:
    from interaction_first.panel_matrix import build_split_panel_matrix
    from interaction_first.sabdab2_partner_metadata import (
        BASE_COLUMNS,
        PARTNER_METADATA_BASE_FORMAT,
    )
    from interaction_first.sabdab2_split_matrix import SPLIT_MATRIX_FORMAT

    base = tmp_path / "base.csv"
    rows = []
    roles = ("train", "validation", "test")
    for role_index, role in enumerate(roles):
        for index in range(3):
            identifier = f"{role}_{index}"
            rows.append(
                {
                    "instance_id": identifier,
                    "antigen_cluster_id": f"ag:{role_index}:{index}",
                    "antigen_sequence_sha256": hashlib.sha256(
                        f"ag:{identifier}".encode()
                    ).hexdigest(),
                    "antigen_length": 100,
                    "surface_residue_count": 30,
                    "surface_observed_residue_count": 100,
                    "antigen_class": "protein",
                    "antibody_cluster_id": f"ab:{role_index}:{index}",
                    "antibody_sequence_sha256": hashlib.sha256(
                        f"ab:{identifier}".encode()
                    ).hexdigest(),
                    "antibody_length": 220,
                    "cdr_residue_count": 35,
                    "antibody_class": "fab",
                    "documented_cognate_group_id": f"pdb:{identifier}",
                }
            )
    with base.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    base_receipt = tmp_path / "base-receipt.json"
    base_receipt.write_text(
        json.dumps(
            {
                "format_version": PARTNER_METADATA_BASE_FORMAT,
                "metadata_sha256": _sha256(base),
                "authority": {"task_labels_read": False},
            }
        ),
        encoding="utf-8",
    )
    split_root = tmp_path / "splits"
    split_root.mkdir()
    manifests = {}
    for name in ("original", "temporal", "antigen_fold"):
        path = split_root / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["instance_id", "role"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "instance_id": row["instance_id"],
                        "role": row["instance_id"].split("_", 1)[0],
                    }
                )
        manifests[name] = {"path": path.name, "sha256": _sha256(path)}
    split_receipt = split_root / "SPLIT_MATRIX_RECEIPT.json"
    split_receipt.write_text(
        json.dumps(
            {
                "format_version": SPLIT_MATRIX_FORMAT,
                "authority": {"task_labels_read": False, "predictions_read": False},
                "manifests": manifests,
            }
        ),
        encoding="utf-8",
    )
    result = build_split_panel_matrix(
        partner_metadata_base_path=base,
        partner_metadata_base_sha256=_sha256(base),
        partner_metadata_base_receipt_path=base_receipt,
        partner_metadata_base_receipt_sha256=_sha256(base_receipt),
        split_matrix_root=split_root,
        split_matrix_receipt_sha256=_sha256(split_receipt),
        output_root=tmp_path / "matrix",
        panel_size_per_direction=1,
        protocol_seed=17,
    )
    assert set(result["matrix"]) == {"original", "temporal", "antigen_fold"}
    for split in result["matrix"].values():
        assert split["metadata_role_counts"] == {
            "train": 3,
            "validation": 3,
            "test": 3,
        }
        assert set(split["panels"]) == {"train", "validation", "test"}
        assert all(value["entry_count"] == 3 for value in split["panels"].values())
