from __future__ import annotations

from scripts.build_abpairfm_stage_a1_pilot_manifest import (
    canonical_pair_hash,
    selection_hash,
)


def test_canonical_pair_hash_is_order_invariant() -> None:
    assert canonical_pair_hash("AAAA", "BBBB") == canonical_pair_hash("BBBB", "AAAA")
    assert canonical_pair_hash("AAAA", "BBBB") != canonical_pair_hash("AAAA", "CCCC")


def test_selection_hash_is_source_and_seed_bound() -> None:
    value = selection_hash(1701, "ppiref50k", "example")
    assert value == selection_hash(1701, "ppiref50k", "example")
    assert value != selection_hash(1702, "ppiref50k", "example")
    assert value != selection_hash(1701, "string", "example")
