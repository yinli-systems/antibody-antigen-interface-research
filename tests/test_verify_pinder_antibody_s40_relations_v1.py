from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "remote_tools"
    / "abf_paircontact_v1"
    / "pinder_scale_v1"
    / "verify_pinder_antibody_s40_relations_v1.py"
)
SPEC = importlib.util.spec_from_file_location("verify_pinder_s40", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_view_order_is_four_axis() -> None:
    assert MODULE.VIEWS == ("vh", "vl", "concatenated_antibody", "antigen")


def test_canonical_sha_is_order_independent() -> None:
    assert MODULE.canonical_sha256({"a": 1, "b": 2}) == MODULE.canonical_sha256(
        {"b": 2, "a": 1}
    )
