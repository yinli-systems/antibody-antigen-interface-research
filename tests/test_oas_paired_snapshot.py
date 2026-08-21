"""Tests for the official OAS paired snapshot and materializer contracts."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_oas_paired_snapshot.py"
MATERIALIZER = ROOT / "scripts/materialize_oas_paired_snapshot.py"


def _module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_snapshot_parser_freezes_summary_and_safe_paths() -> None:
    module = _module(FREEZER, "oas_freezer")
    html = (
        "Your search yielded <b>12</b> filtered sequences from <b>2</b> studies."
        '<script>var CSV = ["wget https://opig.stats.ox.ac.uk/webapps/ngsdb/'
        'paired/StudyA/csv/run_a_paired.csv.gz","wget https://opig.stats.ox.ac.uk/'
        'webapps/ngsdb/paired/StudyB/csv_paired/run_b_Paired_All.csv.gz"];</script>'
    )
    plan = module.parse_snapshot(html)
    assert plan["expected_sequence_count"] == 12
    assert plan["expected_study_count"] == 2
    assert [item["relative_path"] for item in plan["files"]] == [
        "StudyA/run_a_paired.csv.gz",
        "StudyB/run_b_Paired_All.csv.gz",
    ]
    assert plan["authorization"]["stage_b_training"] is False


def test_data_unit_inspection_matches_provider_row_count(tmp_path: Path) -> None:
    module = _module(MATERIALIZER, "oas_materializer")
    path = tmp_path / "unit.csv.gz"
    header = [
        "sequence_id_heavy",
        "sequence_alignment_aa_heavy",
        "sequence_id_light",
        "sequence_alignment_aa_light",
    ]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([json.dumps({"Unique sequences": 2, "Run": "SRR1"})])
        writer.writerow(header)
        writer.writerow(["h1", "AAA", "l1", "CCC"])
        writer.writerow(["h2", "DDD", "l2", "EEE"])
    record = module.inspect_data_unit(path)
    assert record["rows"] == 2
    assert record["run"] == "SRR1"
