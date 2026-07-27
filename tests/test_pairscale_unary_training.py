"""Tests for fixed-epoch PAIRScale unary-control execution."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from interaction_first.pairscale_unary_training import (
    _create_json,
    locked_unary_config,
    run_fixed_epoch_unary_control,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_locked_config_matches_frozen_protocol() -> None:
    protocol = json.loads(
        (REPOSITORY_ROOT / "protocols/pairscale_unary_controls_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config = locked_unary_config().canonical_payload()
    optimization = protocol["optimization"]
    for name, value in optimization.items():
        assert config[name] == value
    assert config["expected_complex_count"] == 192
    assert config["partner_ranking_weight"] == 0.0


def test_holdout_load_occurs_once_after_epoch_260_checkpoint_check() -> None:
    source = inspect.getsource(run_fixed_epoch_unary_control)
    checkpoint_check = source.index(
        'if int(last_receipt["epoch"]) != LOCKED_EPOCHS:'
    )
    holdout_load = source.index(
        "holdout_examples = _load_examples(dataset, holdout_indices)"
    )
    access_receipt = source.index('"HOLDOUT_ACCESS_RECEIPT.json"')
    evaluation = source.index('"HOLDOUT_EVALUATION.json"')
    assert checkpoint_check < holdout_load < access_receipt < evaluation
    assert source.count(
        "holdout_examples = _load_examples(dataset, holdout_indices)"
    ) == 1
    assert "if train_gate_passed" not in source
    assert "break" not in source


def test_canonical_receipts_are_create_only(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _create_json(path, {"b": 2, "a": 1})
    assert path.read_bytes() == b'{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        _create_json(path, {"a": 1, "b": 2})
