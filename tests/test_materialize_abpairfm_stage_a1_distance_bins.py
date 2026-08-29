from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_abpairfm_stage_a1_distance_bins.py"
SPEC = importlib.util.spec_from_file_location("stage_a1_distance_bins", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _atom(
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain:1s}"
        f"{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}          {element:>2s}"
    )


def test_minimum_heavy_atom_distance_ignores_hydrogen() -> None:
    pdb = "\n".join(
        [
            _atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
            _atom(2, "H", "ALA", "A", 1, 2.9, 0.0, 0.0, "H"),
            _atom(3, "CA", "GLY", "B", 1, 3.0, 0.0, 0.0, "C"),
            _atom(4, "CA", "SER", "B", 2, 0.0, 4.0, 0.0, "C"),
        ]
    ).encode("ascii")
    sequence1, sequence2, distances = MODULE.minimum_heavy_atom_distance_matrix(
        pdb, "A", "B"
    )
    assert sequence1 == "A"
    assert sequence2 == "GS"
    np.testing.assert_allclose(distances, [[3.0, 4.0]])


def test_default_bins_preserve_4p5_and_6a_boundaries() -> None:
    distances = np.asarray([[2.0, 4.5, 6.0, 6.1, 24.1]])
    bins = np.searchsorted(
        np.asarray(MODULE.DEFAULT_DISTANCE_BIN_EDGES), distances, side="left"
    )
    assert bins.tolist() == [[0, 1, 2, 3, 7]]
