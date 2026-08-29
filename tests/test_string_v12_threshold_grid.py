"""Tests for the preregistered STRING v12 direct-evidence threshold grid."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_string_v12_threshold_grid.py"


def module():
    specification = importlib.util.spec_from_file_location("string_grid", SCRIPT)
    assert specification is not None and specification.loader is not None
    value = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(value)
    return value


def write_stats(path: Path, *, malformed: int = 0, source_rows: int = 42) -> None:
    value = module()
    lines = [f"source_data_rows={source_rows}", f"malformed_rows={malformed}"]
    remaining = source_rows
    for combined in value.COMBINED_BINS:
        for direct in value.DIRECT_BINS:
            count = remaining if combined == "650_699" and direct == "ge700" else 0
            lines.append(f"bin_{combined}_{direct}={count}")
            if count:
                remaining = 0
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_stats_partition_and_candidate_order_are_frozen(tmp_path: Path) -> None:
    value = module()
    path = tmp_path / "stats"
    write_stats(path)
    source_rows, malformed, matrix = value.parse_stats(path)
    assert (source_rows, malformed) == (42, 0)
    candidates = value.cumulative_candidates(matrix)
    assert [(row["direct_channel_maximum_minimum_inclusive"], row["combined_minimum_inclusive"]) for row in candidates] == [
        (700, 650), (700, 600), (500, 650), (500, 600), (300, 650),
        (300, 600), (100, 650), (100, 600), (1, 650), (1, 600),
    ]
    assert all(row["raw_rows"] == 42 for row in candidates)


def test_stats_fail_closed_on_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "stats"
    write_stats(path, malformed=1)
    with pytest.raises(ValueError, match="malformed"):
        module().parse_stats(path)


def test_stats_fail_closed_when_bins_do_not_partition_source(tmp_path: Path) -> None:
    path = tmp_path / "stats"
    write_stats(path, source_rows=43)
    text = path.read_text(encoding="utf-8").replace("bin_650_699_ge700=43", "bin_650_699_ge700=42")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="partition"):
        module().parse_stats(path)
