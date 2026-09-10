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
    class _NamedOnly(_Residue):
        def __init__(self, coords) -> None:
            super().__init__(coords)
            self.id = (" ", 1, " ")

        def get_resname(self) -> str:
            return "ALA"

    class _Chain2:
        id = "H"

        def __iter__(self):
            return iter([_NamedOnly(_residue(1.0)), _NamedOnly(None)])

    chains = chain_backbones([_Chain2()])

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


def test_a_selected_residue_without_coordinates_becomes_nan_not_zero() -> None:
    """NaN is ESM-IF1's marker for unresolved; zeros would be an atom at the origin."""
    residues = [_residue(0.0), None, _residue(2.0)]

    chosen = select(residues, numpy.array([1, 1, 0]), "antigen", "x_0P")

    assert chosen.shape == (2, 3, 3)
    assert numpy.isnan(chosen[1]).all()
    assert not numpy.isnan(chosen[0]).any()


def test_a_side_that_resolves_nothing_is_refused() -> None:
    """An all-NaN backbone carries no structure and should not reach an encoder."""
    with pytest.raises(BackboneError, match="resolves none"):
        select([None, None], numpy.array([1, 1]), "antigen", "x_0P")


def test_verification_treats_nan_as_equal_to_nan() -> None:
    """Otherwise every unresolved residue reads as a difference."""
    entry = {
        "antibody_backbone": numpy.full((3, 3, 3), numpy.nan, dtype=numpy.float32),
        "antigen_backbone": numpy.ones((5, 3, 3), dtype=numpy.float32),
    }
    entry["antibody_backbone"][0] = 1.0

    verify(entry, entry, "x_0P", 1e-3)


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


class _NamedResidue(_Residue):
    """A residue with a name and an author id, as Biopython supplies."""

    def __init__(self, coords, name: str, number: int, icode: str = " ") -> None:
        super().__init__(coords)
        self._name = name
        self.id = (" ", number, icode)

    def get_resname(self) -> str:
        return self._name


class _NamedChain:
    def __init__(self, residues) -> None:
        self._residues = list(residues)

    def __iter__(self):
        return iter(self._residues)


def test_an_unresolved_residue_leaves_a_gap_rather_than_shifting_the_rest() -> None:
    """The defect the first version had: concatenating observed residues.

    Most entries resolve every residue and the difference never shows. One
    unresolved residue shifts every SEQRES position after it, which produces a
    backbone of exactly the right shape and entirely wrong coordinates.
    """
    align_to_seqres = _MODULE.align_to_seqres

    # SEQRES is ACDE; the structure resolves A, C and E, skipping D at number 3.
    chain = _NamedChain([
        _NamedResidue(_residue(0.0), "ALA", 1),
        _NamedResidue(_residue(1.0), "CYS", 2),
        _NamedResidue(_residue(3.0), "GLU", 4),
    ])

    slots = align_to_seqres(chain, "ACDE")

    assert len(slots) == 4
    assert slots[2] is None
    assert float(slots[3][0][0]) == 3.0


def test_a_fully_resolved_chain_aligns_position_for_position() -> None:
    align_to_seqres = _MODULE.align_to_seqres

    chain = _NamedChain([
        _NamedResidue(_residue(float(index)), name, index + 1)
        for index, name in enumerate(("ALA", "CYS", "ASP"))
    ])

    slots = align_to_seqres(chain, "ACD")

    assert [float(slot[0][0]) for slot in slots] == [0.0, 1.0, 2.0]


def test_the_alignment_must_reconstruct_the_observed_sequence() -> None:
    """A substitution would otherwise be read as two unresolved residues."""
    align_to_seqres = _MODULE.align_to_seqres

    # SEQRES says D at position 2; the structure resolves W there.
    chain = _NamedChain([
        _NamedResidue(_residue(0.0), "ALA", 1),
        _NamedResidue(_residue(1.0), "CYS", 2),
        _NamedResidue(_residue(2.0), "TRP", 3),
    ])

    with pytest.raises(BackboneError, match="does not reconstruct"):
        align_to_seqres(chain, "ACD")


def test_a_deletion_reconstructs_and_is_accepted() -> None:
    align_to_seqres = _MODULE.align_to_seqres

    chain = _NamedChain([
        _NamedResidue(_residue(0.0), "ALA", 1),
        _NamedResidue(_residue(1.0), "CYS", 2),
    ])

    slots = align_to_seqres(chain, "ACD")

    assert slots[2] is None
    assert [float(slot[0][0]) for slot in slots[:2]] == [0.0, 1.0]


def test_author_numbering_is_not_used_as_a_witness() -> None:
    """It looked like one and is not: scheme-numbered chains skip numbers.

    2adf_0P skips author number 95 while resolving every residue but the last,
    so numbering would place the gap fourteen positions from where the sequence
    puts it -- and the sequence is decisive, the observed chain being exactly
    seqres[:108].
    """
    assert not hasattr(_MODULE, "numbering_gaps")


def test_verification_records_a_marker_disagreement_instead_of_failing() -> None:
    """The stored file places some markers one position late; that is a correction."""
    rebuilt = {
        "antibody_backbone": numpy.ones((3, 3, 3), dtype=numpy.float32),
        "antigen_backbone": numpy.ones((4, 3, 3), dtype=numpy.float32),
    }
    stored = {
        "antibody_backbone": numpy.ones((3, 3, 3), dtype=numpy.float32),
        "antigen_backbone": numpy.ones((4, 3, 3), dtype=numpy.float32),
    }
    rebuilt["antigen_backbone"][1] = numpy.nan
    stored["antigen_backbone"][2] = numpy.nan
    differences: list = []

    verify(rebuilt, stored, "x_0P", 1e-3, differences)

    assert differences == [
        {"instance_id": "x_0P", "side": "antigen_backbone",
         "only_in_rebuild": [1], "only_in_stored": [2]}
    ]


def test_a_coordinate_difference_is_still_fatal() -> None:
    """Only the unresolved-marker disagreement is downgraded."""
    rebuilt = {
        "antibody_backbone": numpy.ones((3, 3, 3), dtype=numpy.float32),
        "antigen_backbone": numpy.ones((4, 3, 3), dtype=numpy.float32),
    }
    stored = {
        "antibody_backbone": numpy.full((3, 3, 3), 5.0, dtype=numpy.float32),
        "antigen_backbone": numpy.ones((4, 3, 3), dtype=numpy.float32),
    }

    with pytest.raises(BackboneError, match="differs from the stored one"):
        verify(rebuilt, stored, "x_0P", 1e-3, [])


def test_waters_and_ligands_are_not_part_of_the_sequence() -> None:
    """One chain offers 195 residues against a 120-residue SEQRES."""
    is_amino_acid = _MODULE.is_amino_acid

    class _Het:
        def __init__(self, name: str) -> None:
            self._name = name

        def get_resname(self) -> str:
            return self._name

    assert not is_amino_acid(_Het("HOH"))
    assert is_amino_acid(_Het("ALA"))


def test_a_modified_residue_stays_in_the_sequence() -> None:
    """Selenomethionine is a hetero record and a real methionine in the chain.

    Filtering on the hetero flag alone would delete it and misalign everything
    after it -- the same class of one-position error, from the opposite cause.
    """
    is_amino_acid = _MODULE.is_amino_acid
    one_letter = _MODULE.one_letter

    class _Het:
        def __init__(self, name: str) -> None:
            self._name = name

        def get_resname(self) -> str:
            return self._name

    assert is_amino_acid(_Het("MSE"))
    assert one_letter(_Het("MSE")) == "M"
