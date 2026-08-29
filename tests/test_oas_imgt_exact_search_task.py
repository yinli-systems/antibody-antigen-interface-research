"""Regression tests for exact-search runtime telemetry and publication."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import stat
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_oas_imgt_exact_search_task.py"
SPEC = importlib.util.spec_from_file_location("oas_imgt_exact_search_task", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _noncanonical_stats(path: Path) -> None:
    path.write_text(
        '{"batch_records":7,"batches":1,"bucket_member_visits":1,'
        '"cuda_device":"synthetic","cuda_device_index":0,'
        '"elapsed_seconds":0.10000000000000001,'
        '"format_version":"oas_imgt_exact_cuda_stats_v1","matches":0,'
        '"query_records":3,"quorum_candidates":0,"records_scanned":7,'
        '"seed_lookups":175,"target_file_records":7}\n',
        encoding="ascii",
    )


def test_kernel_stats_accept_strict_source_pinned_double_rendering(
    tmp_path: Path,
) -> None:
    stats = tmp_path / "stats.json"
    _noncanonical_stats(stats)
    with pytest.raises(MODULE.OASIMGTSearchTaskError, match="not canonical"):
        MODULE._strict_json(stats, "ordinary receipt")
    parsed = MODULE._validate_kernel_stats(
        path=stats,
        query_count=3,
        target_records=7,
        match_records=0,
        batch_records=7,
    )
    assert parsed["elapsed_seconds"] == 0.1


def test_task_root_stays_writable_until_atomic_publication(tmp_path: Path) -> None:
    work = tmp_path / "work"
    destination_parent = tmp_path / "fixed-imgt"
    work.mkdir()
    destination_parent.mkdir()
    temporary = work / "partial"
    temporary.mkdir()
    child = temporary / "receipt.json"
    child.write_text("{}\n", encoding="ascii")

    MODULE._freeze_tree(temporary, include_root=False)
    assert temporary.stat().st_mode & stat.S_IWUSR
    assert not child.stat().st_mode & stat.S_IWUSR
    final = destination_parent / "mapper-0000"
    temporary.rename(final)
    metadata = final.lstat()
    final.chmod(
        metadata.st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    )
    assert not final.stat().st_mode & stat.S_IWUSR
