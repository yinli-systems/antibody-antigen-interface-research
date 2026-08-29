"""Tests for the train/validation-only reciprocal panel bundle."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from interaction_first.development_partner_panels import (
    build_development_partner_panel_bundle,
)
from interaction_first.sabdab2_partner_metadata import PARTNER_METADATA_SPLIT_FORMAT
from interaction_first.sabdab2_structure import canonical_json_bytes


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_builds_development_roles_but_no_test_panel(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    columns = [
        "instance_id", "role", "antigen_cluster_id", "antigen_sequence_sha256",
        "antigen_length", "surface_residue_count", "antigen_class",
        "antibody_cluster_id", "antibody_sequence_sha256", "antibody_length",
        "cdr_residue_count", "antibody_class", "documented_cognate_group_id",
    ]
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for role in ("train", "validation", "test"):
            for index in range(4):
                writer.writerow(
                    {
                        "instance_id": f"{role}_{index}",
                        "role": role,
                        "antigen_cluster_id": f"ag:{role}:{index}",
                        "antigen_sequence_sha256": f"{index + 1:064x}",
                        "antigen_length": 100 + index,
                        "surface_residue_count": 50 + index,
                        "antigen_class": "protein",
                        "antibody_cluster_id": f"ab:{role}:{index}",
                        "antibody_sequence_sha256": f"{index + 11:064x}",
                        "antibody_length": 220 + index,
                        "cdr_residue_count": 50 + index,
                        "antibody_class": "conventional_fv",
                        "documented_cognate_group_id": f"pdb:{role}:{index}",
                    }
                )
    receipt = tmp_path / "metadata-receipt.json"
    receipt.write_bytes(
        canonical_json_bytes(
            {
                "format_version": PARTNER_METADATA_SPLIT_FORMAT,
                "metadata_sha256": _sha(metadata),
                "authority": {"task_labels_read": False},
            }
        )
    )
    output = tmp_path / "bundle"
    result = build_development_partner_panel_bundle(
        metadata_path=metadata,
        metadata_sha256=_sha(metadata),
        metadata_receipt_path=receipt,
        metadata_receipt_sha256=_sha(receipt),
        output_root=output,
        panel_size_per_direction=2,
        protocol_seed=17,
    )
    assert result["roles_materialized"] == ["train", "validation"]
    assert result["test_panel_materialized"] is False
    assert (output / "train-reciprocal-2.json").is_file()
    assert (output / "validation-reciprocal-2.json").is_file()
    assert not any("test" in path.name for path in output.iterdir())
