from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "remote_tools"
    / "abpairfm_distributed_contact_pretraining_v1"
    / "materialize_pinder_contact_labels_v1.py"
)
SPEC = importlib.util.spec_from_file_location("pinder_contact_labels", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _atom(serial: int, atom: str, residue: str, number: int, x: float, chain: str = "R") -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}  \n"
    )


def test_parse_and_contacts_use_resolved_sequence_axis() -> None:
    left = (
        _atom(1, "N", "ALA", 4, 0.0)
        + _atom(2, "CA", "ALA", 4, 1.0)
        + _atom(3, "N", "GLY", 9, 20.0)
    ).encode()
    right = (
        _atom(1, "N", "SER", 2, 4.4, "L")
        + _atom(2, "CA", "SER", 2, 5.0, "L")
        + _atom(3, "N", "THR", 7, 30.0, "L")
    ).encode()
    left_sequence, left_coordinates, left_indices = MODULE.parse_holo_endpoint(left)
    right_sequence, right_coordinates, right_indices = MODULE.parse_holo_endpoint(right)
    assert left_sequence == "AG"
    assert right_sequence == "ST"
    assert MODULE.residue_contacts(
        left_coordinates, left_indices, right_coordinates, right_indices, 4.5
    ) == [(0, 0)]


def test_parse_uses_first_model_for_multi_model_endpoint() -> None:
    payload = (
        "MODEL        1\n"
        + _atom(1, "N", "ALA", 21, 0.0)
        + _atom(2, "CA", "ALA", 21, 1.0)
        + "ENDMDL\n"
        + "MODEL        2\n"
        + _atom(1, "N", "GLY", 21, 0.0)
        + _atom(2, "CA", "GLY", 21, 1.0)
        + "ENDMDL\n"
    ).encode()
    sequence, coordinates, indices = MODULE.parse_holo_endpoint(payload)
    assert sequence == "A"
    assert coordinates.shape == (2, 3)
    assert indices.tolist() == [0, 0]


def test_parse_keeps_first_flattened_blank_altloc_residue_variant() -> None:
    payload = (
        _atom(1, "N", "ILE", 21, 0.0)
        + _atom(2, "CA", "ILE", 21, 1.0)
        + _atom(3, "CG1", "ILE", 21, 2.0)
        + _atom(4, "N", "VAL", 21, 0.0)
        + _atom(5, "CA", "VAL", 21, 1.0)
        + _atom(6, "CG1", "VAL", 21, 3.0)
    ).encode()
    sequence, coordinates, indices = MODULE.parse_holo_endpoint(payload)
    assert sequence == "I"
    assert coordinates.shape == (3, 3)
    assert indices.tolist() == [0, 0, 0]


def test_interface_crop_is_deterministic_and_keeps_primary_contacts() -> None:
    left = "A" * 900
    right = "G" * 500
    primary = [(700, 300), (701, 301), (702, 302)]
    auxiliary = [*primary, (699, 299)]
    first = MODULE.crop_contact_patch(
        left, right, primary, auxiliary, maximum_residues=1020
    )
    second = MODULE.crop_contact_patch(
        left, right, primary, auxiliary, maximum_residues=1020
    )
    assert first == second
    cropped_left, cropped_right, cropped_primary, cropped_auxiliary, crop = first
    assert len(cropped_left) + len(cropped_right) == 1020
    assert cropped_primary
    assert set(cropped_primary).issubset(cropped_auxiliary)
    assert crop["left_resolved_length"] == 900
    assert crop["right_resolved_length"] == 500


def test_contact_radius_rejects_invalid_value() -> None:
    coordinates = np.zeros((1, 3), dtype=np.float32)
    indices = np.zeros(1, dtype=np.int32)
    try:
        MODULE.residue_contacts(coordinates, indices, coordinates, indices, 0.0)
    except ValueError as error:
        assert "finite and positive" in str(error)
    else:
        raise AssertionError("invalid contact radius was accepted")
