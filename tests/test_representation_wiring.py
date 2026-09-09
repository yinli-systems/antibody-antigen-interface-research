"""Wiring the concatenated representations into train and evaluate.

The widths of the two sides diverge once AntiBERTy is added, because it covers
antibodies only. They are therefore computed from the cohort rather than assumed
equal, and a width that varied by instance would build a model that cannot
accept some of its own training data -- so that case raises instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "sabdab2_representation_wiring_under_test",
    REPOSITORY_ROOT / "scripts" / "sabdab2_interaction_first.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_wrap = _MODULE._wrap_with_representations

import pandas  # noqa: E402

from interaction_first.composite_embeddings import PRESENCE_COLUMNS  # noqa: E402

BASE_DIM = 1280
EXTRA_DIM = 512


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _manifest(path: Path) -> Path:
    rows = [
        {"INSTANCE": "i1", "vh_seq": "AAAA", "vl_seq": "CC", "ag_seq": "GGGG"},
        {"INSTANCE": "i2", "vh_seq": "TTTT", "vl_seq": "", "ag_seq": "GGGGG"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _frame() -> pandas.DataFrame:
    return pandas.DataFrame(
        {
            "ab_sequence_sha256": [_digest("AAAACC"), _digest("TTTT")],
            "ag_sequence_sha256": [_digest("GGGG"), _digest("GGGGG")],
        }
    )


def _arguments(**overrides) -> argparse.Namespace:
    base = {
        "input_dim": BASE_DIM,
        "representation_manifest": None,
        "antiberty_root": None,
        "antiberty_dim": EXTRA_DIM,
        "esm_if1_root": None,
        "esm_if1_dim": EXTRA_DIM,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_naming_no_encoder_is_the_identity() -> None:
    store = object()

    wrapped, antibody, antigen, summary = _wrap(store, _arguments(), _frame(), BASE_DIM)

    assert wrapped is store
    assert (antibody, antigen) == (BASE_DIM, BASE_DIM)
    assert summary is None


def test_antiberty_widens_only_the_antibody_side(tmp_path: Path) -> None:
    """The asymmetry this design exists to carry: no antigen has an AntiBERTy vector."""
    arguments = _arguments(
        representation_manifest=_manifest(tmp_path / "m.csv"),
        antiberty_root=tmp_path / "ab",
    )

    _, antibody, antigen, _ = _wrap(object(), arguments, _frame(), BASE_DIM)

    assert antibody == BASE_DIM + EXTRA_DIM + PRESENCE_COLUMNS
    assert antigen == BASE_DIM


def test_esm_if1_widens_both_sides(tmp_path: Path) -> None:
    arguments = _arguments(
        representation_manifest=_manifest(tmp_path / "m.csv"),
        esm_if1_root=tmp_path / "if1",
    )

    _, antibody, antigen, _ = _wrap(object(), arguments, _frame(), BASE_DIM)

    assert antibody == antigen == BASE_DIM + EXTRA_DIM + PRESENCE_COLUMNS


def test_both_encoders_stack(tmp_path: Path) -> None:
    arguments = _arguments(
        representation_manifest=_manifest(tmp_path / "m.csv"),
        antiberty_root=tmp_path / "ab",
        esm_if1_root=tmp_path / "if1",
    )

    _, antibody, antigen, summary = _wrap(object(), arguments, _frame(), BASE_DIM)

    assert antibody == BASE_DIM + 2 * (EXTRA_DIM + PRESENCE_COLUMNS)
    assert antigen == BASE_DIM + EXTRA_DIM + PRESENCE_COLUMNS
    assert [source["name"] for source in summary["sources"]] == ["antiberty", "esm_if1"]


def test_an_instance_outside_the_manifest_makes_the_width_inconsistent(tmp_path: Path) -> None:
    """The failure this exists for: a model that cannot accept its own training data."""
    arguments = _arguments(
        representation_manifest=_manifest(tmp_path / "m.csv"),
        antiberty_root=tmp_path / "ab",
    )
    frame = _frame()
    frame.loc[len(frame)] = {
        "ab_sequence_sha256": _digest("NOT_IN_MANIFEST"),
        "ag_sequence_sha256": _digest("GGGG"),
    }

    with pytest.raises(ValueError, match="antibody input width is not constant"):
        _wrap(object(), arguments, frame, BASE_DIM)


def test_naming_an_encoder_without_the_manifest_is_refused(tmp_path: Path) -> None:
    """AntiBERTy's keys are per chain; without the manifest they cannot be derived."""
    arguments = _arguments(antiberty_root=tmp_path / "ab")

    with pytest.raises(ValueError, match="--representation-manifest is required"):
        _wrap(object(), arguments, _frame(), BASE_DIM)


def test_the_summary_names_every_source_and_its_width(tmp_path: Path) -> None:
    arguments = _arguments(
        representation_manifest=_manifest(tmp_path / "m.csv"),
        esm_if1_root=tmp_path / "if1",
    )

    _, _, _, summary = _wrap(object(), arguments, _frame(), BASE_DIM)

    source = summary["sources"][0]
    assert source["representation_dim"] == EXTRA_DIM
    assert source["presence_columns"] == PRESENCE_COLUMNS
    assert source["keys_covered"] == 4


def test_evaluate_has_no_input_dim_flag_so_the_base_width_is_a_parameter() -> None:
    """The failure this exists for: ten trained runs died at the evaluate step.

    ``--input-dim`` is a train-only argument. Reading it from the namespace made
    every evaluation raise AttributeError after its training had finished, which
    is the same shape of defect as the aggregation crashes -- the work was done
    and the run still reported failure.
    """
    arguments = _arguments()
    del arguments.input_dim

    _, antibody, antigen, summary = _wrap(object(), arguments, _frame(), 777)

    assert (antibody, antigen) == (777, 777)
    assert summary is None


def test_the_base_width_comes_from_the_caller_not_the_namespace(tmp_path: Path) -> None:
    """Evaluate passes the checkpoint's recorded width, which may not be 1280."""
    arguments = _arguments(
        representation_manifest=_manifest(tmp_path / "m.csv"),
        esm_if1_root=tmp_path / "if1",
        input_dim=1,
    )

    _, antibody, _, _ = _wrap(object(), arguments, _frame(), 640)

    assert antibody == 640 + EXTRA_DIM + PRESENCE_COLUMNS
