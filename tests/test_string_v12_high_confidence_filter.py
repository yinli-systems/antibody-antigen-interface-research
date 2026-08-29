"""Checks for the MINT-calibrated STRING high-confidence scale filter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_string_v12_high_confidence_filter.py"


def _module():
    specification = importlib.util.spec_from_file_location("high_confidence_filter", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_stats_require_an_exact_evidence_partition(tmp_path: Path) -> None:
    module = _module()
    stats = tmp_path / "stats.txt"
    stats.write_text(
        "direct_channel_rows=14\n"
        "high_confidence_rows=226\n"
        "malformed_rows=0\n"
        "other_calibrated_channel_rows=1\n"
        "source_data_rows=2365\n"
        "transferred_channel_rows=211\n",
        encoding="utf-8",
    )
    assert module.parse_stats(stats)["high_confidence_rows"] == 226


def test_stats_reject_a_nonpartitioning_receipt(tmp_path: Path) -> None:
    module = _module()
    stats = tmp_path / "stats.txt"
    stats.write_text(
        "direct_channel_rows=14\n"
        "high_confidence_rows=226\n"
        "malformed_rows=0\n"
        "other_calibrated_channel_rows=0\n"
        "source_data_rows=2365\n"
        "transferred_channel_rows=211\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="partition"):
        module.parse_stats(stats)
