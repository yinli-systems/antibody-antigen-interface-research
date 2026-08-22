"""Unit contracts for selective PINDER holo mapping extraction."""

from __future__ import annotations

import csv
import gzip
import importlib.util
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/extract_pinder_holo_mappings.py"


def _module():
    specification = importlib.util.spec_from_file_location("pinder_extract", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_mapping_member_is_strict() -> None:
    module = _module()
    assert module.mapping_member("1abc__A1_P12345-R.pdb") == (
        "mappings/1abc__A1_P12345-R.parquet"
    )
    for invalid in ("../bad.pdb", "bad.parquet", ""):
        try:
            module.mapping_member(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid PINDER filename: {invalid!r}")


def test_selective_extraction_uses_only_manifest_holo_members(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.tsv.gz"
    with gzip.open(manifest, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("id", "holo_R_pdb", "holo_L_pdb"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "x",
                "holo_R_pdb": "1abc__A1_P1-R.pdb",
                "holo_L_pdb": "1abc__B1_P2-L.pdb",
            }
        )
    archive = tmp_path / "mappings.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("mappings/1abc__A1_P1-R.parquet", b"right")
        output.writestr("mappings/1abc__B1_P2-L.parquet", b"left")
        output.writestr("mappings/unselected.parquet", b"hidden")

    selected = module.required_members(manifest)
    output_root = tmp_path / "selected"
    count, reused, total = module.extract_selected(archive, output_root, selected)
    assert (count, reused, total) == (2, 0, 9)
    assert sorted(path.name for path in output_root.iterdir()) == [
        "1abc__A1_P1-R.parquet",
        "1abc__B1_P2-L.parquet",
    ]
    count, reused, total = module.extract_selected(archive, output_root, selected)
    assert (count, reused, total) == (2, 2, 9)
