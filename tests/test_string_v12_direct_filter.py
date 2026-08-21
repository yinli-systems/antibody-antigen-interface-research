"""Tests for fail-closed STRING direct-evidence filter receipts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_string_v12_direct_filter.py"


def _module():
    specification = importlib.util.spec_from_file_location("string_filter", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_stats_schema_and_counts_are_frozen(tmp_path: Path) -> None:
    module = _module()
    stats = tmp_path / "stats"
    stats.write_text(
        "combined_score_ge_700_rows=100\n"
        "direct_evidence_kept_rows=40\n"
        "malformed_rows=0\n"
        "source_data_rows=1000\n"
        "transferred_only_ge_700_rows=60\n",
        encoding="utf-8",
    )
    assert module.parse_stats(stats)["direct_evidence_kept_rows"] == 40


def test_malformed_source_rows_fail_closed(tmp_path: Path) -> None:
    module = _module()
    stats = tmp_path / "stats"
    stats.write_text(
        "combined_score_ge_700_rows=100\n"
        "direct_evidence_kept_rows=40\n"
        "malformed_rows=1\n"
        "source_data_rows=1000\n"
        "transferred_only_ge_700_rows=59\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed"):
        module.parse_stats(stats)
