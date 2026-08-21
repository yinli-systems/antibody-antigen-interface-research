"""Tests for OAS exact native-pair quality normalization."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_oas_stage_b_pairs.py"


def _module():
    specification = importlib.util.spec_from_file_location("oas_stage_b", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_normalize_row_requires_native_productive_pair_and_all_cdrs() -> None:
    module = _module()
    heavy = "A" * 80 + "CCC" + "D" * 20 + "EEE" + "F" * 20 + "GGG"
    light = "H" * 70 + "III" + "K" * 10 + "LLL" + "M" * 10 + "NNN"
    row = {
        "productive_heavy": "True",
        "productive_light": "True",
        "vj_in_frame_heavy": "True",
        "vj_in_frame_light": "True",
        "stop_codon_heavy": "False",
        "stop_codon_light": "False",
        "locus_heavy": "H",
        "locus_light": "K",
        "sequence_alignment_aa_heavy": heavy,
        "sequence_alignment_aa_light": light,
        "cdr1_aa_heavy": "CCC",
        "cdr2_aa_heavy": "EEE",
        "cdr3_aa_heavy": "GGG",
        "cdr1_aa_light": "III",
        "cdr2_aa_light": "LLL",
        "cdr3_aa_light": "NNN",
    }
    normalized = module.normalize_row(row)
    assert normalized is not None
    assert normalized[:2] == (heavy, light)
    row["locus_heavy"] = "IGH"
    row["locus_light"] = "IGL"
    assert module.normalize_row(row) is not None
    row["productive_light"] = "False"
    assert module.normalize_row(row) is None
