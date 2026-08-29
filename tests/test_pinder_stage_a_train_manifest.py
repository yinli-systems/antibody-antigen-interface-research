"""Tests for the PINDER train-only system-manifest builder."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pinder_stage_a_train_manifest.py"
RECEIPT = ROOT / "data_registry/materialization_receipts/pinder_2024_02_train_system_manifest_audit_v1.json"


def _module():
    specification = importlib.util.spec_from_file_location("pinder_manifest", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_manifest(module, path: Path) -> None:
    rows = [
        {
            column: {
                "split": "train",
                "id": "a",
                "cluster_id": "cluster-1",
                "holo_R_pdb": "a-R.pdb",
                "holo_L_pdb": "a-L.pdb",
                "predicted_R_pdb": "a-pred.pdb",
            }.get(column, "")
            for column in module.MANIFEST_COLUMNS
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=module.MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_sql_projects_only_frozen_columns_and_train_rows(tmp_path: Path) -> None:
    module = _module()
    sql = module._manifest_sql(
        tmp_path / "index.parquet",
        tmp_path / "metadata.parquet",
        tmp_path / "manifest.tsv",
    )
    assert "SELECT *" not in sql
    assert "i.split='train'" in sql
    assert "ORDER BY i.id" in sql
    assert "m.resolution<=4.0" in sql
    assert "m.length1+m.length2<=1020" in sql


def test_manifest_scan_counts_views_and_required_members(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(module, manifest)
    rows, clusters, views, members = module._scan_manifest(manifest)
    assert rows == 1
    assert clusters == 1
    assert views == {
        "canonical_apo_any": 0,
        "canonical_apo_both": 0,
        "canonical_holo_both": 1,
        "predicted_any": 1,
        "predicted_both": 0,
    }
    assert members == {"pdbs/a-R.pdb", "pdbs/a-L.pdb", "pdbs/a-pred.pdb"}


def test_archive_validation_is_exact_and_rejects_test_names() -> None:
    module = _module()
    required = {"pdbs/a-R.pdb", "pdbs/a-L.pdb"}
    audit = module._validate_archive_members(
        Path("unused.zip"), required, ["pdbs/a-L.pdb\n", "pdbs/a-R.pdb\n"]
    )
    assert audit["required_members_present"] is True
    with pytest.raises(RuntimeError, match="absent"):
        module._validate_archive_members(Path("unused.zip"), required, ["pdbs/a-R.pdb\n"])
    with pytest.raises(RuntimeError, match="test-named"):
        module._validate_archive_members(
            Path("unused.zip"), required, ["pdbs/a-R.pdb\n", "pdbs/a-L.pdb\n", "test/x.pdb\n"]
        )


def test_gzip_output_is_deterministic(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "input.tsv"
    source.write_bytes(b"a\tb\n1\t2\n")
    first = tmp_path / "first.tsv.gz"
    second = tmp_path / "second.tsv.gz"
    module._deterministic_gzip(source, first)
    module._deterministic_gzip(source, second)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rb") as handle:
        assert handle.read() == source.read_bytes()


def test_real_manifest_receipt_proves_inventory_but_not_training_authority() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["inventory"]["strict_quality_system_count"] == 982_075
    assert receipt["inventory"]["strict_quality_cluster_count"] == 18_099
    assert receipt["structure_archive_audit"]["required_member_count"] == 741_188
    assert receipt["structure_archive_audit"]["required_members_present"] is True
    assert receipt["authorization"]["stage_a_training"] is False
    assert not any(receipt["test_access"].values())
