"""Selecting extraction rows by instance rather than by manifest split.

The canonical manifest's train/test division is not the double-cold split. Rows
the manifest calls ``train`` include instances the split manifest assigns to
validation, and instances the split manifest assigns to train sit under the
manifest's ``test``. Selecting by manifest split therefore cannot express "the
development instances and nothing else": measured on the real data, manifest
``train`` reaches 88.7% of the split's train and 73.3% of its validation, and
the only way to reach the rest by manifest split is to parse ``test`` -- which
holds the 217 instances this project does not read.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "extract_pretrained_representations_under_test",
    REPOSITORY_ROOT / "scripts" / "extract_pretrained_representations.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

RepresentationError = _MODULE.RepresentationError
SEALED_SPLIT = _MODULE.SEALED_SPLIT
load_split_instances = _MODULE.load_split_instances
load_units = _MODULE.load_units


def _split_manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": ["i1", "i3"],
                    "validation": ["i2"],
                    "test": ["i4"],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _manifest(path: Path) -> Path:
    """A manifest whose own split disagrees with the split manifest, as the real one does."""
    rows = [
        {"INSTANCE": "i1", "split": "train", "vh_seq": "AAAA", "vl_seq": "CCCC", "ag_seq": "DDDD"},
        {"INSTANCE": "i2", "split": "train", "vh_seq": "EEEE", "vl_seq": "FFFF", "ag_seq": "GGGG"},
        {"INSTANCE": "i3", "split": "test", "vh_seq": "HHHH", "vl_seq": "IIII", "ag_seq": "KKKK"},
        {"INSTANCE": "i4", "split": "test", "vh_seq": "LLLL", "vl_seq": "MMMM", "ag_seq": "NNNN"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_the_sealed_split_is_refused_not_counted(tmp_path: Path) -> None:
    """Every other split here is a development split; a frozen one is not recoverable."""
    manifest = _split_manifest(tmp_path / "split.json")

    with pytest.raises(RepresentationError, match="does not read it"):
        load_split_instances(manifest, ["train", SEALED_SPLIT])


def test_selecting_development_splits_returns_their_instances(tmp_path: Path) -> None:
    selected, counts = load_split_instances(
        _split_manifest(tmp_path / "split.json"), ["train", "validation"]
    )

    assert selected == {"i1", "i2", "i3"}
    assert counts == {"train": 2, "validation": 1}


def test_an_unknown_split_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RepresentationError, match="no split"):
        load_split_instances(_split_manifest(tmp_path / "split.json"), ["holdout"])


def test_a_manifest_without_splits_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"components": []}), encoding="utf-8")

    with pytest.raises(RepresentationError, match="no 'splits' mapping"):
        load_split_instances(path, ["train"])


def test_instance_selection_reaches_rows_the_manifest_files_elsewhere(tmp_path: Path) -> None:
    """i3 is a split-train instance sitting under the manifest's test rows."""
    units, parsed, dropped = load_units(
        _manifest(tmp_path / "manifest.csv"),
        ["ag"],
        {"train", "test"},
        None,
        instances={"i1", "i2", "i3"},
    )

    assert len(units) == 3
    assert parsed == {"train": 2, "test": 1}
    assert dropped["instance_not_selected"] == 1


def test_the_sealed_instance_is_not_parsed_even_when_its_manifest_split_is(tmp_path: Path) -> None:
    """The failure this exists for: reaching i3 must not also read i4."""
    units, _, _ = load_units(
        _manifest(tmp_path / "manifest.csv"),
        ["ag"],
        {"train", "test"},
        None,
        instances={"i1", "i2", "i3"},
    )

    assert all(unit.sequence != "NNNN" for unit in units)


def test_selecting_by_manifest_split_alone_still_works(tmp_path: Path) -> None:
    """The existing behaviour is unchanged when no instance set is given."""
    units, parsed, dropped = load_units(
        _manifest(tmp_path / "manifest.csv"), ["ag"], {"train"}, None
    )

    assert len(units) == 2
    assert parsed == {"train": 2}
    assert "instance_not_selected" not in dropped


def test_rows_outside_the_named_manifest_splits_are_still_skipped(tmp_path: Path) -> None:
    """Instance selection is applied in addition to the split filter, not instead."""
    units, parsed, _ = load_units(
        _manifest(tmp_path / "manifest.csv"),
        ["ag"],
        {"train"},
        None,
        instances={"i1", "i3"},
    )

    assert len(units) == 1
    assert parsed == {"train": 1}


def test_the_antibody_chain_concatenates_heavy_and_light(tmp_path: Path) -> None:
    """The ab axis must match the digest the model's own store is keyed by."""
    units, _, _ = load_units(
        _manifest(tmp_path / "manifest.csv"), ["ab"], {"train"}, None, instances={"i1"}
    )

    assert [unit.sequence for unit in units] == ["AAAACCCC"]


def test_a_missing_light_chain_drops_that_unit_not_the_row(tmp_path: Path) -> None:
    """A quarter of this dataset has no light chain, so this is the common case."""
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["INSTANCE", "split", "vh_seq", "vl_seq", "ag_seq"]
        )
        writer.writeheader()
        writer.writerow(
            {"INSTANCE": "i1", "split": "train", "vh_seq": "AAAA", "vl_seq": "", "ag_seq": "DDDD"}
        )

    units, _, dropped = load_units(path, ["vh", "vl"], {"train"}, None, instances={"i1"})

    assert [unit.sequence for unit in units] == ["AAAA"]
    assert dropped["empty_sequence"] == 1
