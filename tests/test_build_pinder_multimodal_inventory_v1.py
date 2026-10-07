"""Contract tests for the train-only multimodal inventory."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "remote_tools"
    / "abpairfm_multimodal_pretraining_v1"
    / "build_pinder_multimodal_inventory_v1.py"
)
SPEC = importlib.util.spec_from_file_location("multimodal_inventory", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


FIELDS = [
    "split",
    "id",
    "cluster_id",
    "predicted_R_pdb",
    "predicted_L_pdb",
    "sequence_R",
    "sequence_L",
    "sequence_sha256_R",
    "sequence_sha256_L",
    "system_weight",
]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            import io

            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)


def row(identifier: str, right: str, left: str) -> dict[str, str]:
    sequence_r = "ACDE"
    sequence_l = "FGHI"
    return {
        "split": "train",
        "id": identifier,
        "cluster_id": "cluster-1",
        "predicted_R_pdb": right,
        "predicted_L_pdb": left,
        "sequence_R": sequence_r,
        "sequence_L": sequence_l,
        "sequence_sha256_R": hashlib.sha256(sequence_r.encode()).hexdigest(),
        "sequence_sha256_L": hashlib.sha256(sequence_l.encode()).hexdigest(),
        "system_weight": "1.0",
    }


def arguments(tmp_path: Path, rows: list[dict[str, str]]):
    manifest = tmp_path / "B-full.tsv.gz"
    write_manifest(manifest, rows)
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="ascii")
    return SimpleNamespace(
        curriculum_manifest=manifest,
        curriculum_manifest_sha256=MODULE.sha256_file(manifest),
        protocol=protocol,
        protocol_sha256=MODULE.sha256_file(protocol),
        expected_input_systems=len(rows),
        expected_selected_systems=sum(
            bool(item["predicted_R_pdb"] and item["predicted_L_pdb"])
            for item in rows
        ),
        expected_structure_members=2,
        output_inventory=tmp_path / "inventory.tsv.gz",
        output_result=tmp_path / "result.json",
    )


def test_builds_only_both_predicted_train_rows(tmp_path: Path) -> None:
    rows = [
        row("one", "af__one.pdb", "af__two.pdb"),
        row("two", "", "af__two.pdb"),
    ]
    args = arguments(tmp_path, rows)
    result = MODULE.build(args)
    assert result["status"] == "PASS_TRAIN_ONLY_PREDICTED_STRUCTURE_INVENTORY"
    assert result["counts"] == {
        "input_train_systems": 2,
        "selected_both_predicted_systems": 1,
        "unique_predicted_structure_members": 2,
    }
    with gzip.open(args.output_inventory, "rt", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle, delimiter="\t"))
    assert [item["id"] for item in selected] == ["one"]
    assert json.loads(args.output_result.read_text())["test_access"] == {
        "pinder_test_rows_read": False,
        "pinder_test_structures_read": False,
        "pinder_validation_rows_read": False,
    }


def test_rejects_test_rows_and_unsafe_members(tmp_path: Path) -> None:
    test_row = row("one", "af__one.pdb", "af__two.pdb")
    test_row["split"] = "test"
    with pytest.raises(ValueError, match="train-only"):
        MODULE.build(arguments(tmp_path / "test", [test_row]))

    unsafe = row("one", "../one.pdb", "af__two.pdb")
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.build(arguments(tmp_path / "unsafe", [unsafe]))
