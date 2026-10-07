from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SOURCE_DIR = (
    Path(__file__).parents[1]
    / "remote_tools"
    / "abf_paircontact_v1"
    / "pinder_scale_v1"
)
PARENT_SOURCE = SOURCE_DIR / "audit_pinder_antibody_s2_exact_four_view_v1.py"
PARENT_SPEC = importlib.util.spec_from_file_location(
    "audit_pinder_antibody_s2_exact_four_view_v1", PARENT_SOURCE
)
assert PARENT_SPEC is not None and PARENT_SPEC.loader is not None
PARENT = importlib.util.module_from_spec(PARENT_SPEC)
sys.modules[PARENT_SPEC.name] = PARENT
PARENT_SPEC.loader.exec_module(PARENT)

SOURCE = SOURCE_DIR / "audit_pinder_antibody_s2_full_four_view_relation_v1.py"
SPEC = importlib.util.spec_from_file_location("pinder_s2_full_relation", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_complete_inventory_deduplicates_every_view_and_skips_empty_vl():
    rows = [
        PARENT.SourceRow("a", "1aaa", "AAAA", "CCCC", "DDDD"),
        PARENT.SourceRow("b", "2bbb", "AAAA", "", "EEEE"),
        PARENT.SourceRow("c", "3ccc", "FFFF", "CCCC", "DDDD"),
    ]
    inventory = MODULE.build_complete_inventories(rows)
    assert len(inventory["vh"]) == 2
    assert len(inventory["vl"]) == 1
    assert len(inventory["concatenated_antibody"]) == 3
    assert len(inventory["antigen"]) == 2


def test_unordered_edge_filter_keeps_one_digest_orientation():
    forward = {"left_sequence_sha256": "a", "right_sequence_sha256": "b"}
    reverse = {"left_sequence_sha256": "b", "right_sequence_sha256": "a"}
    self_edge = {"left_sequence_sha256": "a", "right_sequence_sha256": "a"}
    assert MODULE._unordered_edge_key(forward) == ("a", "b")
    assert MODULE._unordered_edge_key(reverse) is None
    assert MODULE._unordered_edge_key(self_edge) is None
