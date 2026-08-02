"""Tests for label-free official-test reciprocal partner panels."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(tmp_path: Path, count: int = 40) -> tuple[Path, Path]:
    from interaction_first.asep_frozen_test import LABEL_FREE_TEST_FORMAT

    root = tmp_path / "label_free"
    root.mkdir()
    rows = []
    for index in range(count):
        rows.append(
            {
                "instance_id": f"{index:04x}_0P",
                "role": "test",
                "pdb_id": f"{index:04x}",
                "antigen_pdb_seqres_sha256": hashlib.sha256(
                    f"antigen-{index}".encode()
                ).hexdigest(),
                "antibody_pdb_seqres_sha256": hashlib.sha256(
                    f"antibody-{index}".encode()
                ).hexdigest(),
                "antigen_pdb_seqres_length": 100 + index,
                "antibody_pdb_seqres_length": 210 + index,
                "antigen_observed_residue_count": 90 + index,
                "antibody_observed_residue_count": 200 + index,
            }
        )
    manifest = root / "ASEP_LABEL_FREE_TEST_METADATA.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = root / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json"
    receipt.write_text(
        json.dumps(
            {
                "format_version": LABEL_FREE_TEST_FORMAT,
                "manifest_sha256": _sha256(manifest),
                "test_instance_count": count,
                "test_label_open_count": 0,
                "checks": {
                    "official_test_graph_tensors_opened": False,
                    "official_test_contact_or_residue_labels_read": False,
                },
            }
        )
    )
    return root, receipt


def _decontamination(
    tmp_path: Path, *, manifest_sha256: str, eligible: list[str]
) -> tuple[Path, Path]:
    from interaction_first.asep_decontamination import DECONTAMINATION_FORMAT

    root = tmp_path / "decontamination"
    root.mkdir()
    result = root / "ASEP_SABDAB2_DECONTAMINATION.json"
    result.write_text(
        json.dumps(
            {
                "format_version": DECONTAMINATION_FORMAT,
                "authority": {
                    "label_free_test_manifest_sha256": manifest_sha256,
                    "test_labels_read": False,
                    "model_predictions_or_scores_read": False,
                },
                "counts": {"decontaminated_primary_instances": len(eligible)},
                "decontaminated_primary_instance_ids": eligible,
            }
        )
    )
    receipt = root / "DECONTAMINATION_RECEIPT.json"
    receipt.write_text(
        json.dumps(
            {
                "format_version": DECONTAMINATION_FORMAT,
                "result_sha256": _sha256(result),
                "test_labels_read": False,
                "model_predictions_or_scores_read": False,
            }
        )
    )
    return root, receipt


def test_full_and_decontaminated_panels_are_deterministic_and_label_free(
    tmp_path: Path,
) -> None:
    from interaction_first.asep_test_partner_panel import (
        build_asep_label_free_test_partner_panel,
    )

    root, receipt = _metadata(tmp_path)
    full_a = build_asep_label_free_test_partner_panel(
        label_free_test_root=root,
        label_free_test_receipt_sha256=_sha256(receipt),
        output_path=tmp_path / "full_a.json",
        panel_size_per_direction=3,
        protocol_seed=91,
    )
    full_b = build_asep_label_free_test_partner_panel(
        label_free_test_root=root,
        label_free_test_receipt_sha256=_sha256(receipt),
        output_path=tmp_path / "full_b.json",
        panel_size_per_direction=3,
        protocol_seed=91,
    )
    assert full_a == full_b
    assert full_a["authority"]["task_label_columns_read"] is False
    assert full_a["authority"]["official_test_graph_tensors_opened"] is False
    assert len(full_a["entries"]) == 40
    first = full_a["entries"][0]
    assert first["instance_id"] not in first["wrong_antigen_instance_ids"]
    assert first["instance_id"] not in first["wrong_antibody_instance_ids"]

    eligible = [f"{index:04x}_0P" for index in range(35)]
    decon, decon_receipt = _decontamination(
        tmp_path,
        manifest_sha256=json.loads(receipt.read_text())["manifest_sha256"],
        eligible=eligible,
    )
    clean = build_asep_label_free_test_partner_panel(
        label_free_test_root=root,
        label_free_test_receipt_sha256=_sha256(receipt),
        decontamination_root=decon,
        decontamination_receipt_sha256=_sha256(decon_receipt),
        output_path=tmp_path / "clean.json",
        panel_size_per_direction=31,
    )
    assert len(clean["entries"]) == 35
    assert clean["authority"]["panel_scope"] == "sabdab2_decontaminated_primary"
    assert clean["authority"]["decontamination_result_sha256"] == _sha256(
        decon / "ASEP_SABDAB2_DECONTAMINATION.json"
    )
