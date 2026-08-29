from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_ppiref50k_interfaces_v2.py"


def module():
    spec = importlib.util.spec_from_file_location("ppiref50k", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def atom(
    serial: int,
    name: str,
    residue: str,
    chain: str,
    number: int,
    x: float,
    y: float,
    z: float,
    *,
    altloc: str = " ",
    occupancy: float = 1.0,
) -> str:
    return f"ATOM  {serial:5d} {name:^4s}{altloc}{residue:>3s} {chain}{number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f} 20.00           C  "


def test_interface_contacts_and_patch_sequences() -> None:
    value = module()
    lines = [
        "REMARK   3  RESOLUTION: 2.0 A",
        "REMARK   3  STRUCTURE METHOD: x-ray diffraction",
        "REMARK   3  BSA: 700.0 A^2",
        atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
        atom(2, "CB", "ALA", "A", 1, 0.0, 1.0, 0.0),
        atom(3, "CA", "GLY", "A", 3, 20.0, 0.0, 0.0),
        atom(4, "CA", "SER", "B", 5, 5.9, 0.0, 0.0),
        atom(5, "CA", "TYR", "B", 6, 40.0, 0.0, 0.0),
    ]
    result = value.parse_interface_pdb(("\n".join(lines) + "\n").encode(), "A", "B", 6.0)
    assert result["chain1_patch_sequence"] == "AG"
    assert result["chain2_patch_sequence"] == "SY"
    assert result["contact_pairs"] == [(0, 0)]
    assert result["metadata"] == {
        "bsa": 700.0,
        "resolution": 2.0,
        "source_atom_chain1": "A",
        "source_atom_chain2": "B",
        "structure_method": "x-ray diffraction",
    }


def test_seqres_selects_only_target_protein_chains(tmp_path: Path) -> None:
    value = module()
    path = tmp_path / "seqres.gz"
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write(
            ">1ABC_A mol:protein length:4\nACDE\n"
            ">1ABC_B mol:dna length:4\nACGT\n"
            ">2xyz_C mol:protein length:3\nFGH\n"
        )
    selected, counts = value.target_full_sequences(path, {("1abc", "A"), ("1abc", "B")})
    assert selected == {("1abc", "A"): "ACDE"}
    assert counts == {
        "source_seqres_records": 3,
        "target_seqres_ambiguous": 0,
        "target_seqres_casefold_aliases": 0,
        "target_seqres_records": 1,
        "target_seqres_missing": 1,
    }


def test_seqres_uses_unique_casefold_alias_but_prefers_exact(tmp_path: Path) -> None:
    value = module()
    path = tmp_path / "seqres.gz"
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write(">1ABC_c mol:protein length:4\nACDE\n>2XYZ_A mol:protein length:4\nFGHI\n>2XYZ_a mol:protein length:4\nKLMN\n")
    selected, counts = value.target_full_sequences(path, {("1abc", "C"), ("2xyz", "A")})
    assert selected == {("1abc", "C"): "ACDE", ("2xyz", "A"): "FGHI"}
    assert counts["target_seqres_casefold_aliases"] == 1
    assert counts["target_seqres_ambiguous"] == 0


def test_canonical_id_is_order_preserving_and_pdb_normalized() -> None:
    assert module().canonical_id("3HCH_A_B") == ("3hch", "A", "B")


def test_interface_missing_chain_fails_closed() -> None:
    value = module()
    payload = (atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0) + "\n").encode()
    try:
        value.parse_interface_pdb(payload, "A", "B", 6.0)
    except ValueError as error:
        assert "cannot be resolved" in str(error)
    else:
        raise AssertionError("missing PPIRef chain was accepted")


def test_interface_chain_case_alias_is_resolved_without_coalescing() -> None:
    value = module()
    payload = (
        atom(1, "CA", "ALA", "c", 1, 0.0, 0.0, 0.0)
        + "\n"
        + atom(2, "CA", "SER", "i", 2, 5.0, 0.0, 0.0)
        + "\n"
    ).encode()
    result = value.parse_interface_pdb(payload, "C", "I", 6.0)
    assert result["contact_pairs"] == [(0, 0)]
    assert result["metadata"]["source_atom_chain1"] == "c"
    assert result["metadata"]["source_atom_chain2"] == "i"


def test_highest_occupancy_altloc_is_selected_and_b_only_is_valid() -> None:
    value = module()
    payload = (
        atom(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, altloc="A", occupancy=0.2)
        + "\n"
        + atom(2, "CA", "ALA", "A", 1, 100.0, 0.0, 0.0, altloc="B", occupancy=0.8)
        + "\n"
        + atom(3, "CA", "SER", "B", 2, 105.0, 0.0, 0.0, altloc="B", occupancy=0.4)
        + "\n"
    ).encode()
    result = value.parse_interface_pdb(payload, "A", "B", 6.0)
    assert result["contact_pairs"] == [(0, 0)]


def test_microheterogeneous_residue_uses_highest_occupancy_identity() -> None:
    value = module()
    payload = (
        atom(1, "CA", "TRP", "A", 11, 0.0, 0.0, 0.0, altloc="A", occupancy=0.77)
        + "\n"
        + atom(2, "CA", "TYR", "A", 11, 100.0, 0.0, 0.0, altloc="B", occupancy=0.23)
        + "\n"
        + atom(3, "CA", "SER", "B", 2, 5.0, 0.0, 0.0)
        + "\n"
    ).encode()
    result = value.parse_interface_pdb(payload, "A", "B", 6.0)
    assert result["chain1_patch_sequence"] == "W"
    assert result["contact_pairs"] == [(0, 0)]
