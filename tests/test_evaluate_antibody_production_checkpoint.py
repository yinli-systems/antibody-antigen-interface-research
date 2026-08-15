"""Test the full-OAS checkpoint evaluator's fail-closed panel contract."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate_antibody_production_checkpoint.py"


def _load():
    for item in (ROOT / "scripts", ROOT / "src"):
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)
    specification = importlib.util.spec_from_file_location(
        "production_checkpoint_evaluator_test", EVALUATOR
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_evaluator_is_ascii_and_syntax_valid() -> None:
    source = EVALUATOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "split\": \"validation" in source
    assert "official_test_accessed\": False" in source
    assert "frozen_test_accessed\": False" in source
    assert "external_test_accessed\": False" in source
    assert "Loaded evaluation model differs from checkpoint" in source


def test_panel_is_deterministic_unique_little_endian_int64() -> None:
    evaluator = _load()
    first = evaluator._panel_indices(record_count=10_000, sample_count=512, seed=73)
    second = evaluator._panel_indices(record_count=10_000, sample_count=512, seed=73)
    assert first.dtype == np.dtype("<i8")
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 512
    assert hashlib.sha256(first.tobytes()).hexdigest() == hashlib.sha256(
        second.tobytes()
    ).hexdigest()


def test_panel_refuses_to_exceed_validation_universe() -> None:
    evaluator = _load()
    with pytest.raises(RuntimeError, match="exceeds the audited split"):
        evaluator._panel_indices(record_count=10, sample_count=11, seed=17)


def test_evaluation_receipt_is_non_authorizing() -> None:
    evaluator = _load()
    assert evaluator.AUTHORIZATION == {
        "checkpoint_evaluation": True,
        "configuration_promotion": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }
    assert not any(evaluator.TEST_ACCESS.values())


def test_only_local_rank_zero_hashes_array_files_before_followers_open() -> None:
    source = EVALUATOR.read_text(encoding="ascii")
    leader = "if local_rank == 0:"
    barrier = "dist.barrier()"
    follower = "if local_rank != 0:"
    assert source.index(leader) < source.index(barrier) < source.index(follower)
    assert source.count("verify_array_file_hashes=True") == 1
    assert source.count("verify_array_file_hashes=False") == 1
    assert "Distributed validation hash-verification mode differs" in source
    assert "Distributed validation record counts differ" in source
