"""Tests for pre-performance SAbDab2-to-AsEP contamination flags."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_authorities(tmp_path: Path) -> dict[str, Path]:
    from interaction_first.asep_decontamination import (
        CROSS_FOLDSEEK_RECEIPT_FORMAT,
    )
    from interaction_first.asep_frozen_test import LABEL_FREE_TEST_FORMAT

    label_root = tmp_path / "label_free"
    structures = label_root / "antigen_structures"
    structures.mkdir(parents=True)
    structure_rows = []
    for identifier, pdb_id, antigen_digest, antibody_digest in (
        ("1abc_0P", "1abc", "1" * 64, "2" * 64),
        ("3ghi_0P", "3ghi", "3" * 64, "4" * 64),
    ):
        structure = structures / f"asep_{identifier}.pdb"
        structure.write_text("ATOM      1  CA  ALA a   1       0.000   0.000   0.000\nEND\n")
        structure_rows.append(
            {
                "dataset_index": len(structure_rows),
                "instance_id": identifier,
                "role": "test",
                "foldseek_structure_id": hashlib.sha256(
                    f"asep:{identifier}".encode("ascii")
                ).hexdigest(),
                "pdb_id": pdb_id,
                "antigen_pdb_seqres_sha256": antigen_digest,
                "antibody_pdb_seqres_sha256": antibody_digest,
                "antigen_structure_path": structure.relative_to(label_root).as_posix(),
                "antigen_structure_sha256": _sha256(structure),
            }
        )
    label_manifest = label_root / "ASEP_LABEL_FREE_TEST_METADATA.csv"
    with label_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(structure_rows[0]))
        writer.writeheader()
        writer.writerows(structure_rows)
    label_receipt = label_root / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json"
    label_receipt.write_text(
        json.dumps(
            {
                "format_version": LABEL_FREE_TEST_FORMAT,
                "manifest_sha256": _sha256(label_manifest),
                "test_instance_count": 2,
                "test_label_open_count": 0,
                "checks": {
                    "official_test_graph_tensors_opened": False,
                    "official_test_contact_or_residue_labels_read": False,
                    "model_predictions_or_scores_read": False,
                },
            }
        )
    )

    sab = tmp_path / "sabdab2.csv"
    sab_rows = [
        {
            "instance_id": "pdb_00001abc_H_L",
            "role": "train",
            "antigen_structure_id": "pdb_00001abc_H_L",
            "antigen_sequence_sha256": "1" * 64,
            "antibody_sequence_sha256": "9" * 64,
            "pdb_assembly_id": "pdb_00001abc",
        },
        {
            "instance_id": "pdb_00002def_H_L",
            "role": "validation",
            "antigen_structure_id": "pdb_00002def_H_L",
            "antigen_sequence_sha256": "8" * 64,
            "antibody_sequence_sha256": "2" * 64,
            "pdb_assembly_id": "pdb_00002def",
        },
        {
            "instance_id": "pdb_00003ghi_H_L",
            "role": "test",
            "antigen_structure_id": "pdb_00003ghi_H_L",
            "antigen_sequence_sha256": "3" * 64,
            "antibody_sequence_sha256": "4" * 64,
            "pdb_assembly_id": "pdb_00003ghi",
        },
    ]
    with sab.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(sab_rows[0]))
        writer.writeheader()
        writer.writerows(sab_rows)

    export = tmp_path / "sabdab2_export.csv"
    export_rows = []
    for row in sab_rows:
        foldseek_id = hashlib.sha256(row["antigen_structure_id"].encode()).hexdigest()
        export_rows.append(
            {
                "antigen_structure_id": row["antigen_structure_id"],
                "foldseek_structure_id": foldseek_id,
                "relative_path": f"structures/{foldseek_id}.pdb",
                "sha256": "a" * 64,
            }
        )
    with export.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(export_rows[0]))
        writer.writeheader()
        writer.writerows(export_rows)

    alignments = tmp_path / "cross.tsv"
    alignment_rows = []
    query_ids = {
        row["instance_id"]: row["foldseek_structure_id"] for row in structure_rows
    }
    target_ids = {
        row["antigen_structure_id"]: row["foldseek_structure_id"]
        for row in export_rows[:2]
    }
    for query_instance, query in query_ids.items():
        for target_structure, target in target_ids.items():
            hard = query_instance == "1abc_0P" and target_structure == "pdb_00002def_H_L"
            alignment_rows.append(
                {
                    "query": query,
                    "target": target,
                    "qtmscore": 0.7 if hard else 0.2,
                    "ttmscore": 0.6 if hard else 0.3,
                }
            )
    with alignments.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("query", "target", "qtmscore", "ttmscore"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(alignment_rows)
    foldseek = tmp_path / "foldseek.json"
    foldseek.write_text(
        json.dumps(
            {
                "format_version": CROSS_FOLDSEEK_RECEIPT_FORMAT,
                "query_manifest_sha256": _sha256(label_manifest),
                "target_manifest_sha256": _sha256(sab),
                "target_export_manifest_sha256": _sha256(export),
                "normalized_alignments_sha256": _sha256(alignments),
                "audit_score_rule": "reciprocal_min",
                "audit_tm_threshold": 0.5,
                "query_role": "test",
                "target_roles": ["train", "validation"],
                "complete_ordered_cross_grid": True,
                "query_count": 2,
                "target_structure_count": 2,
                "expected_pair_count": 4,
                "raw_row_count": 4,
                "alignment_type": 1,
                "test_labels_read": False,
                "model_predictions_or_scores_read": False,
            }
        )
    )
    return {
        "label_root": label_root,
        "label_receipt": label_receipt,
        "sab": sab,
        "export": export,
        "alignments": alignments,
        "foldseek": foldseek,
    }


def test_decontamination_flags_are_frozen_without_test_labels(tmp_path: Path) -> None:
    from interaction_first.asep_decontamination import (
        audit_asep_sabdab2_decontamination,
    )

    paths = _write_authorities(tmp_path)
    output = tmp_path / "audit"
    result = audit_asep_sabdab2_decontamination(
        label_free_test_root=paths["label_root"],
        label_free_test_receipt_sha256=_sha256(paths["label_receipt"]),
        sabdab2_split_manifest=paths["sab"],
        sabdab2_split_manifest_sha256=_sha256(paths["sab"]),
        sabdab2_antigen_export_manifest=paths["export"],
        sabdab2_antigen_export_manifest_sha256=_sha256(paths["export"]),
        cross_alignments=paths["alignments"],
        cross_alignments_sha256=_sha256(paths["alignments"]),
        cross_foldseek_receipt=paths["foldseek"],
        cross_foldseek_receipt_sha256=_sha256(paths["foldseek"]),
        output_root=output,
    )
    assert result["decontaminated_primary_instance_count"] == 1
    payload = json.loads((output / "ASEP_SABDAB2_DECONTAMINATION.json").read_text())
    assert payload["decontaminated_primary_instance_ids"] == ["3ghi_0P"]
    first = {row["instance_id"]: row for row in payload["records"]}["1abc_0P"]
    assert first["flags"]["exact_pdb_identity"] is True
    assert first["flags"]["exact_antigen_pdb_seqres"] is True
    assert first["flags"]["antigen_structural_similarity"] is True
    assert payload["authority"]["test_labels_read"] is False


def test_decontamination_rejects_incomplete_cross_grid(tmp_path: Path) -> None:
    from interaction_first.asep_decontamination import (
        AsepDecontaminationError,
        audit_asep_sabdab2_decontamination,
    )

    paths = _write_authorities(tmp_path)
    lines = paths["alignments"].read_text().splitlines()
    paths["alignments"].write_text("\n".join(lines[:-1]) + "\n")
    foldseek = json.loads(paths["foldseek"].read_text())
    foldseek["normalized_alignments_sha256"] = _sha256(paths["alignments"])
    paths["foldseek"].write_text(json.dumps(foldseek))
    with pytest.raises(AsepDecontaminationError, match="complete grid"):
        audit_asep_sabdab2_decontamination(
            label_free_test_root=paths["label_root"],
            label_free_test_receipt_sha256=_sha256(paths["label_receipt"]),
            sabdab2_split_manifest=paths["sab"],
            sabdab2_split_manifest_sha256=_sha256(paths["sab"]),
            sabdab2_antigen_export_manifest=paths["export"],
            sabdab2_antigen_export_manifest_sha256=_sha256(paths["export"]),
            cross_alignments=paths["alignments"],
            cross_alignments_sha256=_sha256(paths["alignments"]),
            cross_foldseek_receipt=paths["foldseek"],
            cross_foldseek_receipt_sha256=_sha256(paths["foldseek"]),
            output_root=tmp_path / "audit",
        )
