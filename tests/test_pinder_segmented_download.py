"""Deterministic range contracts for the PINDER segmented downloader."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/download_pinder_pdbs_segmented.py"


def _module():
    specification = importlib.util.spec_from_file_location("pinder_download", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_byte_ranges_cover_exactly_once() -> None:
    ranges = _module().byte_ranges(103, 8)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 102
    assert sum(end - start + 1 for start, end in ranges) == 103
    for previous, current in zip(ranges, ranges[1:]):
        assert previous[1] + 1 == current[0]
    assert max(end - start + 1 for start, end in ranges) - min(
        end - start + 1 for start, end in ranges
    ) <= 1
