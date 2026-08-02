"""End-to-end contracts for the exhaustive AsEP/SAbDab2 Foldseek rectangle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdb(path: Path) -> str:
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND\n"
    )
    return _sha256(path)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    from interaction_first.asep_frozen_test import LABEL_FREE_TEST_FORMAT
    from interaction_first.sabdab2_antigen_export import ANTIGEN_EXPORT_FORMAT

    label = tmp_path / "label"
    label_structures = label / "antigen_structures"
    label_structures.mkdir(parents=True)
    query_rows = []
    for index, instance in enumerate(("1abc_0P", "2def_0P")):
        foldseek_id = hashlib.sha256(f"asep:{instance}".encode()).hexdigest()
        path = label_structures / f"{foldseek_id}.pdb"
        query_rows.append(
            {
                "dataset_index": index,
                "instance_id": instance,
                "role": "test",
                "foldseek_structure_id": foldseek_id,
                "antigen_structure_path": path.relative_to(label).as_posix(),
                "antigen_structure_sha256": _pdb(path),
            }
        )
    label_manifest = label / "ASEP_LABEL_FREE_TEST_METADATA.csv"
    with label_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(query_rows[0]))
        writer.writeheader()
        writer.writerows(query_rows)
    label_receipt = label / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json"
    label_receipt.write_text(
        json.dumps(
            {
                "format_version": LABEL_FREE_TEST_FORMAT,
                "manifest_sha256": _sha256(label_manifest),
                "test_label_open_count": 0,
                "checks": {"official_test_graph_tensors_opened": False},
            }
        )
    )

    split = tmp_path / "sabdab2.csv"
    split_rows = [
        {"instance_id": "sab_a", "role": "train", "antigen_structure_id": "sab_a"},
        {
            "instance_id": "sab_b",
            "role": "validation",
            "antigen_structure_id": "sab_b",
        },
        {"instance_id": "sab_c", "role": "test", "antigen_structure_id": "sab_c"},
    ]
    with split.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    export = tmp_path / "export"
    export_structures = export / "structures"
    export_structures.mkdir(parents=True)
    export_rows = []
    for structure_id in ("sab_a", "sab_b", "sab_c"):
        foldseek_id = hashlib.sha256(structure_id.encode()).hexdigest()
        path = export_structures / f"{foldseek_id}.pdb"
        export_rows.append(
            {
                "antigen_structure_id": structure_id,
                "foldseek_structure_id": foldseek_id,
                "relative_path": path.relative_to(export).as_posix(),
                "sha256": _pdb(path),
            }
        )
    export_manifest = export / "ANTIGEN_STRUCTURE_EXPORT_MANIFEST.csv"
    with export_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(export_rows[0]))
        writer.writeheader()
        writer.writerows(export_rows)
    export_receipt = export / "ANTIGEN_STRUCTURE_EXPORT_RECEIPT.json"
    export_receipt.write_text(
        json.dumps(
            {
                "format_version": ANTIGEN_EXPORT_FORMAT,
                "authority": {"task_labels_read": False},
                "manifest_sha256": _sha256(export_manifest),
            }
        )
    )
    return label, split, export


def test_prepare_and_finalize_complete_cross_grid(tmp_path: Path) -> None:
    from interaction_first.asep_cross_foldseek import (
        finalize_asep_sabdab2_cross_foldseek,
        prepare_asep_sabdab2_cross_foldseek,
    )

    label, split, export = _write_inputs(tmp_path)
    prep = tmp_path / "prep"
    prepare_asep_sabdab2_cross_foldseek(
        label_free_test_root=label,
        label_free_test_receipt_sha256=_sha256(
            label / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json"
        ),
        sabdab2_split_manifest=split,
        sabdab2_split_manifest_sha256=_sha256(split),
        sabdab2_antigen_export_root=export,
        sabdab2_antigen_export_receipt_sha256=_sha256(
            export / "ANTIGEN_STRUCTURE_EXPORT_RECEIPT.json"
        ),
        output_root=prep,
    )
    query_ids = sorted(path.stem for path in (prep / "queries").glob("*.pdb"))
    target_ids = sorted(path.stem for path in (prep / "targets").glob("*.pdb"))
    assert len(query_ids) == 2
    assert len(target_ids) == 2
    raw = tmp_path / "raw.tsv"
    raw.write_text(
        "".join(f"{q}\t{t}\t0.2\t0.3\n" for q in query_ids for t in target_ids)
    )
    command = tmp_path / "COMMAND.txt"
    command.write_text(
        "foldseek easy-search queries targets raw tmp --alignment-type 1 "
        "--exhaustive-search 1 -e inf --format-output "
        "query,target,qtmscore,ttmscore\n"
    )
    normalized = tmp_path / "normalized.tsv"
    receipt = tmp_path / "receipt.json"
    result = finalize_asep_sabdab2_cross_foldseek(
        preparation_root=prep,
        preparation_receipt_sha256=_sha256(prep / "PREPARATION_RECEIPT.json"),
        raw_alignments=raw,
        raw_alignments_sha256=_sha256(raw),
        command_path=command,
        foldseek_version="foldseek-test",
        normalized_output=normalized,
        receipt_output=receipt,
    )
    assert result["complete_ordered_cross_grid"] is True
    assert result["raw_row_count"] == 4
    assert result["normalized_alignments_sha256"] == _sha256(normalized)
    assert normalized.read_text().splitlines()[0] == "query\ttarget\tqtmscore\tttmscore"


def test_finalize_rejects_incomplete_rectangle(tmp_path: Path) -> None:
    from interaction_first.asep_cross_foldseek import (
        AsepCrossFoldseekError,
        finalize_asep_sabdab2_cross_foldseek,
        prepare_asep_sabdab2_cross_foldseek,
    )

    label, split, export = _write_inputs(tmp_path)
    prep = tmp_path / "prep"
    prepare_asep_sabdab2_cross_foldseek(
        label_free_test_root=label,
        label_free_test_receipt_sha256=_sha256(
            label / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json"
        ),
        sabdab2_split_manifest=split,
        sabdab2_split_manifest_sha256=_sha256(split),
        sabdab2_antigen_export_root=export,
        sabdab2_antigen_export_receipt_sha256=_sha256(
            export / "ANTIGEN_STRUCTURE_EXPORT_RECEIPT.json"
        ),
        output_root=prep,
    )
    query = next((prep / "queries").glob("*.pdb")).stem
    target = next((prep / "targets").glob("*.pdb")).stem
    raw = tmp_path / "raw.tsv"
    raw.write_text(f"{query}\t{target}\t0.2\t0.3\n")
    command = tmp_path / "COMMAND.txt"
    command.write_text(
        "--alignment-type 1 --exhaustive-search 1 -e inf "
        "query,target,qtmscore,ttmscore"
    )
    with pytest.raises(AsepCrossFoldseekError, match="incomplete"):
        finalize_asep_sabdab2_cross_foldseek(
            preparation_root=prep,
            preparation_receipt_sha256=_sha256(prep / "PREPARATION_RECEIPT.json"),
            raw_alignments=raw,
            raw_alignments_sha256=_sha256(raw),
            command_path=command,
            foldseek_version="foldseek-test",
            normalized_output=tmp_path / "normalized.tsv",
            receipt_output=tmp_path / "receipt.json",
        )
