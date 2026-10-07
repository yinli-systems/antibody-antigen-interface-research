"""Reproducing AsEP's own split exactly, or refusing to produce one.

Nothing in this project is comparable to published work yet. AsEP is the
benchmark those numbers are quoted on, so its split has to be reproduced
exactly for a comparison to mean anything. The official file stores integer
indices into a canonical identifier list, not identifiers, and a conversion that
resolved them loosely would yield a split that resembles AsEP's without being
it -- worse than having no comparison at all.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "convert_asep_official_split_under_test",
    REPOSITORY_ROOT / "scripts" / "convert_asep_official_split.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

AsepSplitError = _MODULE.AsepSplitError
FORMAT_VERSION = _MODULE.FORMAT_VERSION
build_manifest = _MODULE.build_manifest
load_identifiers = _MODULE.load_identifiers
resolve = _MODULE.resolve

IDENTIFIERS = [f"{n}abc_0P" for n in range(10)]
SPLIT = {"epitope_group": {"train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]}}


def test_indices_resolve_through_the_identifier_list_in_file_order() -> None:
    """The official entries are positions, not names."""
    assert resolve([0, 3, 9], IDENTIFIERS, "t") == ["0abc_0P", "3abc_0P", "9abc_0P"]


def test_a_torch_style_index_resolves_too() -> None:
    """Entries arrive as tensor(0), whose int() is the position."""

    class _Scalar:
        def __init__(self, value: int) -> None:
            self.value = value

        def __int__(self) -> int:
            return self.value

    assert resolve([_Scalar(2)], IDENTIFIERS, "t") == ["2abc_0P"]


def test_an_index_past_the_end_is_refused(tmp_path: Path) -> None:
    """The failure this exists for: a silently truncated split."""
    with pytest.raises(AsepSplitError, match="outside the identifier list"):
        resolve([10], IDENTIFIERS, "t")


def test_a_negative_index_is_refused() -> None:
    """Python would wrap it to the end of the list and produce a plausible id."""
    with pytest.raises(AsepSplitError, match="outside the identifier list"):
        resolve([-1], IDENTIFIERS, "t")


def test_a_non_index_entry_is_refused() -> None:
    with pytest.raises(AsepSplitError, match="is not an index"):
        resolve(["7abc_0P"], IDENTIFIERS, "t")


def test_val_is_renamed_to_validation_without_changing_membership() -> None:
    manifest = build_manifest(SPLIT, IDENTIFIERS, "epitope_group", None)

    assert manifest["splits"]["validation"] == ["6abc_0P", "7abc_0P"]
    assert "val" not in manifest["splits"]


def test_the_manifest_uses_the_format_the_pipeline_requires() -> None:
    manifest = build_manifest(SPLIT, IDENTIFIERS, "epitope_group", None)

    assert manifest["format_version"] == FORMAT_VERSION
    assert set(manifest["splits"]) == {"train", "validation", "test"}


def test_an_unknown_split_name_lists_what_is_available() -> None:
    with pytest.raises(AsepSplitError, match="epitope_ratio"):
        build_manifest(
            {"epitope_ratio": SPLIT["epitope_group"]}, IDENTIFIERS, "epitope_group", None
        )


def test_an_instance_in_two_splits_is_refused() -> None:
    """Membership must be unique; the training loader assumes it."""
    overlapping = {"g": {"train": [0, 1], "val": [1], "test": [2]}}

    with pytest.raises(AsepSplitError, match="is in both"):
        build_manifest(overlapping, IDENTIFIERS, "g", None)


def test_an_empty_split_is_refused() -> None:
    with pytest.raises(AsepSplitError, match="has no"):
        build_manifest({"g": {"train": [0], "val": [], "test": [1]}}, IDENTIFIERS, "g", None)


def test_split_members_without_a_prepared_instance_are_refused() -> None:
    """Two AsEP entries were dropped in preparation; a silent gap would bias the split."""
    known = set(IDENTIFIERS) - {"9abc_0P"}

    with pytest.raises(AsepSplitError, match="no prepared instance"):
        build_manifest(SPLIT, IDENTIFIERS, "epitope_group", known)


def test_a_complete_split_passes_the_instance_check() -> None:
    manifest = build_manifest(SPLIT, IDENTIFIERS, "epitope_group", set(IDENTIFIERS))

    assert sum(len(v) for v in manifest["splits"].values()) == len(IDENTIFIERS)


def test_no_cluster_components_are_invented() -> None:
    """Faking components from the split assignment would resample three clusters."""
    manifest = build_manifest(SPLIT, IDENTIFIERS, "epitope_group", None)

    assert manifest["component_id_by_instance"] == {}


def test_the_manifest_says_its_test_split_is_the_benchmark_test_set() -> None:
    manifest = build_manifest(SPLIT, IDENTIFIERS, "epitope_group", None)

    assert any("benchmark's test set" in line for line in manifest["limitations"])


def test_duplicate_identifiers_are_refused(tmp_path: Path) -> None:
    """A repeated line would make two positions resolve to the same instance."""
    path = tmp_path / "ids.txt"
    path.write_text("a_0P\nb_0P\na_0P\n", encoding="utf-8")

    with pytest.raises(AsepSplitError, match="duplicates"):
        load_identifiers(path)


def test_blank_lines_do_not_shift_the_indices(tmp_path: Path) -> None:
    """A trailing newline must not become an entry and move every later index."""
    path = tmp_path / "ids.txt"
    path.write_text("a_0P\nb_0P\nc_0P\n\n", encoding="utf-8")

    assert load_identifiers(path) == ["a_0P", "b_0P", "c_0P"]
