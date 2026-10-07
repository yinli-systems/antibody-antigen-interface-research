"""Reconstructing the carbonyl oxygen from an N, CA, C backbone.

ProteinMPNN featurises four backbone atoms and the store here holds three, so O
is constructed. It is the same construction for every arm, so it cannot create a
difference between them -- but it can be wrong in a way that makes every arm
worse, which would be invisible in a paired comparison and would understate the
encoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from interaction_first.representation_backends import place_backbone_oxygen  # noqa: E402


def chain(count: int):
    """A crude extended backbone: residues spaced along x."""
    torch = pytest.importorskip("torch")
    residues = []
    for index in range(count):
        base = 3.8 * index
        residues.append([[base, 0.0, 0.0], [base + 1.45, 0.0, 0.0], [base + 2.4, 1.0, 0.0]])
    return torch.tensor(residues, dtype=torch.float32)


def test_the_oxygen_is_added_as_a_fourth_atom():
    coordinates = chain(5)
    atoms = place_backbone_oxygen(coordinates)
    assert atoms.shape == (5, 4, 3)
    assert (atoms[:, :3] == coordinates).all(), "the given atoms must be untouched"


def test_the_carbon_oxygen_distance_is_the_bond_length():
    torch = pytest.importorskip("torch")
    atoms = place_backbone_oxygen(chain(6))
    distance = (atoms[:, 3] - atoms[:, 2]).norm(dim=-1)
    assert torch.allclose(distance, torch.full_like(distance, 1.231), atol=1e-4)


def test_the_oxygen_points_away_from_the_alpha_carbon():
    """A sign error would put O on top of CA and every arm would degrade."""
    atoms = place_backbone_oxygen(chain(4))
    to_oxygen = atoms[:, 3] - atoms[:, 2]
    to_alpha = atoms[:, 1] - atoms[:, 2]
    cosine = (to_oxygen * to_alpha).sum(-1) / (
        to_oxygen.norm(dim=-1) * to_alpha.norm(dim=-1)
    )
    assert (cosine < 0).all(), "O must lie on the far side of C from CA"


def test_the_oxygen_points_away_from_the_next_nitrogen():
    atoms = place_backbone_oxygen(chain(4))
    to_oxygen = atoms[:-1, 3] - atoms[:-1, 2]
    to_next = atoms[1:, 0] - atoms[:-1, 2]
    cosine = (to_oxygen * to_next).sum(-1) / (
        to_oxygen.norm(dim=-1) * to_next.norm(dim=-1)
    )
    assert (cosine < 0).all(), "O must lie on the far side of C from the next N"


def test_a_single_residue_still_gets_an_oxygen():
    """The construction needs a next residue; one residue has none."""
    atoms = place_backbone_oxygen(chain(1))
    assert atoms.shape == (1, 4, 3)
    assert float((atoms[0, 3] - atoms[0, 2]).norm()) == pytest.approx(1.231, abs=1e-4)


def test_the_construction_is_deterministic():
    torch = pytest.importorskip("torch")
    coordinates = chain(7)
    assert torch.equal(place_backbone_oxygen(coordinates),
                       place_backbone_oxygen(coordinates))


def test_translating_the_backbone_translates_the_oxygen_with_it():
    """The geometry is local; an absolute position must not leak into it."""
    torch = pytest.importorskip("torch")
    coordinates = chain(5)
    shift = torch.tensor([10.0, -3.0, 7.0])
    moved = place_backbone_oxygen(coordinates + shift)
    assert torch.allclose(moved, place_backbone_oxygen(coordinates) + shift, atol=1e-4)
