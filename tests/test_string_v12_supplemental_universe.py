from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_string_v12_supplemental_universe.py"


def module():
    spec = importlib.util.spec_from_file_location("string_supplement", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_expected_rows_exclude_zero_direct_bins() -> None:
    matrix = {
        combined: {
            "zero": 100,
            "1_99": 1,
            "100_299": 2,
            "300_499": 3,
            "500_699": 4,
            "ge700": 5,
        }
        for combined in ("600_649", "650_699")
    }
    grid = {"matrix": matrix, "status": "AUDITED_NOT_SELECTED_NOT_AUTHORIZED", "test_access": False}
    assert module().expected_candidate_rows(grid) == 30


def test_ineligible_grid_fails_closed() -> None:
    with pytest.raises(ValueError, match="not eligible"):
        module().expected_candidate_rows({"status": "WRONG", "test_access": False})


def test_filter_stats_schema_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "stats"
    path.write_text("candidate_rows=10\nmalformed_rows=0\nsource_data_rows=100\n", encoding="utf-8")
    assert module().parse_stats(path)["candidate_rows"] == 10
    path.write_text("candidate_rows=10\nmalformed_rows=1\nsource_data_rows=100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        module().parse_stats(path)
