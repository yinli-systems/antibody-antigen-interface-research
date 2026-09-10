"""Rebuilding AsEP backbone coordinates from its structures.

The stored file covers 1,101 of 1,721 instances while 1,723 structures are on
disk, and the gap is uneven across AsEP's own split -- 65.5% of train, 59.8% of
validation, 55.0% of test. A structure encoder built on it would train and be
tested on differently-covered cohorts for a reason that lies in the extraction
rather than in the data, so the gap has to be closed rather than described.

The recipe is documented nowhere and was recovered by reproducing entries that
already exist. These tests hold the parts that are easy to get wrong and
impossible to notice: the antibody chain order, the mask alignment, and the
refusal to invent a coordinate for a residue that has none.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

numpy = pytest.importorskip("numpy")

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "build_asep_backbones_under_test",
    REPOSITORY_ROOT / "scripts" / "build_asep_backbones.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ANTIBODY_CHAIN_ORDER = _MODULE.ANTIBODY_CHAIN_ORDER
BackboneError = _MODULE.BackboneError
chain_backbones = _MODULE.chain_backbones
select = _MODULE.select
verify = _MODULE.verify


def _residue(value: float):
    return [[value, 0.0, 0.0], [value, 1.0, 0.0], [value, 2.0, 0.0]]


class _Atom:
    def __init__(self, coord) -> None:
        self.coord = numpy.asarray(coord, dtype=numpy.float32)


class _Residue:
    def __init__(self, coords) -> None:
        self._atoms = None if coords is None else dict(zip(("N", "CA", "C"), coords))

    def __getitem__(self, name):
        if self._atoms is None or name not in self._atoms:
            raise KeyError(name)
        return _Atom(self._atoms[name])


class _Chain:
    def __init__(self, identifier: str, residues) -> None:
        self.id = identifier
        self._residues = [_Residue(entry) for entry in residues]

    def __iter__(self):
        return iter(self._residues)


def test_the_antibody_is_heavy_then_light_not_file_order() -> None:
    """The files store L before H; the mask is indexed H then L."""
    assert ANTIBODY_CHAIN_ORDER == ("H", "L")


def test_a_residue_without_backbone_atoms_becomes_none() -> None:
    chains = chain_backbones([_Chain("H", [_residue(1.0), None])])

    assert chains["H"][1] is None


def test_a_mask_shorter_than_the_chain_is_refused() -> None:
    """The failure this exists for: silently selecting the wrong residues."""
    residues = [_residue(float(n)) for n in range(5)]

    with pytest.raises(BackboneError, match="mask covers"):
        select(residues, numpy.array([1, 0, 1]), "antibody", "x_0P")


def test_a_mask_longer_than_the_chain_is_refused() -> None:
    residues = [_residue(float(n)) for n in range(3)]

    with pytest.raises(BackboneError, match="mask covers"):
        select(residues, numpy.array([1, 0, 1, 1, 0]), "antibody", "x_0P")


def test_the_mask_picks_the_indices_it_marks_in_order() -> None:
    residues = [_residue(float(n)) for n in range(5)]

    chosen = select(residues, numpy.array([0, 1, 0, 1, 1]), "antibody", "x_0P")

    assert chosen.shape == (3, 3, 3)
    assert [float(row[0][0]) for row in chosen] == [1.0, 3.0, 4.0]


def test_a_selected_residue_without_coordinates_is_refused() -> None:
    """Inventing a coordinate would put a fabricated atom into a structure encoder."""
    residues = [_residue(0.0), None, _residue(2.0)]

    with pytest.raises(BackboneError, match="missing backbone atoms"):
        select(residues, numpy.array([1, 1, 0]), "antigen", "x_0P")


def test_an_unselected_residue_without_coordinates_is_tolerated() -> None:
    """Only the residues the mask keeps have to be complete."""
    residues = [_residue(0.0), None, _residue(2.0)]

    chosen = select(residues, numpy.array([1, 0, 1]), "antigen", "x_0P")

    assert chosen.shape == (2, 3, 3)


def test_an_empty_mask_is_refused() -> None:
    with pytest.raises(BackboneError, match="selects nothing"):
        select([_residue(0.0)], numpy.array([0]), "antibody", "x_0P")


def test_verification_accepts_an_exact_rebuild() -> None:
    entry = {
        "antibody_backbone": numpy.zeros((4, 3, 3), dtype=numpy.float32),
        "antigen_backbone": numpy.ones((7, 3, 3), dtype=numpy.float32),
    }

    verify(entry, entry, "x_0P", 1e-3)


def test_verification_rejects_a_shape_difference() -> None:
    rebuilt = {
        "antibody_backbone": numpy.zeros((4, 3, 3), dtype=numpy.float32),
        "antigen_backbone": numpy.ones((7, 3, 3), dtype=numpy.float32),
    }
    stored = dict(rebuilt, antibody_backbone=numpy.zeros((5, 3, 3), dtype=numpy.float32))

    with pytest.raises(BackboneError, match="shape"):
        verify(rebuilt, stored, "x_0P", 1e-3)


def test_verification_rejects_a_coordinate_difference() -> None:
    """A wrong chain order produces the right shape and the wrong coordinates."""
    rebuilt = {
        "antibody_backbone": numpy.zeros((4, 3, 3), dtype=numpy.float32),
        "antigen_backbone": numpy.ones((7, 3, 3), dtype=numpy.float32),
    }
    stored = dict(rebuilt, antibody_backbone=numpy.full((4, 3, 3), 9.0, dtype=numpy.float32))

    with pytest.raises(BackboneError, match="differs from the stored one"):
        verify(rebuilt, stored, "x_0P", 1e-3)
