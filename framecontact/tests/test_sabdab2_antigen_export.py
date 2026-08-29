"""Tests for compact Foldseek antigen PDB output."""

from __future__ import annotations

import numpy as np

from interaction_first.sabdab2_antigen_export import _pdb_text


def test_pdb_export_skips_missing_frames_without_changing_residue_numbers() -> None:
    backbone = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[np.nan, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    text, valid = _pdb_text("AGY", backbone, np.asarray([0, 0, 1]))
    assert valid == 2
    assert text.count("ATOM") == 6
    assert " A   1" in text
    assert " B   1" in text
    assert text.endswith("END\n")
