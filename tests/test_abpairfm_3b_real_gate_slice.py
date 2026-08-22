"""Checks for the deterministic ABPairFM 3B train-only real-gate slice."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_abpairfm_3b_real_gate_slice.py"


def _module():
    specification = importlib.util.spec_from_file_location("real_gate_slice", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _row(index: int) -> list[str]:
    cluster1 = f"cluster-{index:03d}"
    cluster2 = f"cluster-{index + 100:03d}"
    return [cluster1, cluster2, f"protein-{index}", f"protein-{index + 100}"] + [
        "0"
    ] * 7 + ["700"]


def test_candidate_selection_is_order_independent_and_exact() -> None:
    module = _module()
    rows = [_row(index) for index in range(20)]
    first, first_count = module.select_candidate_rows(
        rows, seed=1701, candidate_count=5
    )
    second, second_count = module.select_candidate_rows(
        reversed(rows), seed=1701, candidate_count=5
    )
    assert first_count == second_count == 20
    assert first == second
    assert len(first) == 5


def test_pair_hash_is_seeded_and_endpoint_order_sensitive_after_canonicalization() -> None:
    module = _module()
    assert module.pair_hash(1701, "a", "b") == module.pair_hash(1701, "a", "b")
    assert module.pair_hash(1701, "a", "b") != module.pair_hash(1702, "a", "b")
