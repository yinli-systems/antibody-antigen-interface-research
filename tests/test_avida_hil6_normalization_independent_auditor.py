"""Tests for the independent AVIDa normalization auditor."""

from __future__ import annotations

import inspect
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import interaction_first.avida_hil6_normalization_independent_auditor as auditor
from interaction_first.avida_hil6_normalization_independent_auditor import (
    AvidaNormalizationAuditError,
    _content_id,
    _next,
    _row_iterator,
)


def test_auditor_does_not_import_primary_normalizer() -> None:
    source = inspect.getsource(auditor)
    assert "from interaction_first.avida_hil6_normalization import" not in source
    assert "import interaction_first.avida_hil6_normalization" not in source


def test_row_replay_preserves_physical_order_across_batches(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"value": index} for index in range(7)]),
        path,
        row_group_size=2,
    )
    assert [row["value"] for row in _row_iterator(path, batch_size=3)] == list(
        range(7)
    )


def test_replay_fails_closed_when_output_ends_early() -> None:
    with pytest.raises(AvidaNormalizationAuditError, match="ended early"):
        _next(iter(()), "normalized pair")


def test_pair_identity_is_content_addressed_over_both_normalized_endpoints() -> None:
    first = _content_id("pair", "ab_sha256:a\nag_sha256:b")
    second = _content_id("pair", "ab_sha256:a\nag_sha256:c")
    assert first.startswith("pair_sha256:")
    assert len(first) == len("pair_sha256:") + 64
    assert first != second
