from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "interaction_first"
    / "chimera_external.py"
)
_SPEC = importlib.util.spec_from_file_location("chimera_external_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ChimeraExternalError = _MODULE.ChimeraExternalError
adapt_complex = _MODULE.adapt_complex
adapt_complex_native_axis = _MODULE.adapt_complex_native_axis
adapt_complex_pdb_anchored = _MODULE.adapt_complex_pdb_anchored
load_feature_archive = _MODULE.load_feature_archive
load_split = _MODULE.load_split
verify_frozen_authority = _MODULE.verify_frozen_authority


def _atom14(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.zeros((length, 14, 3), dtype=np.float32)
    mask = np.zeros((length, 14), dtype=np.bool_)
    for index in range(length):
        coords[index, 0] = (index, 0.0, 0.0)
        coords[index, 1] = (index, 1.0, 0.0)
        coords[index, 2] = (index, 1.0, 1.0)
        mask[index, :3] = True
    return coords, mask, coords[:, 1].copy()


def _feature() -> dict:
    heavy = _atom14(2)
    light = _atom14(1)
    antigen = _atom14(2)
    return {
        "complex_id": "1abc_H_L_A",
        "heavy_sequence": "AG",
        "heavy_atom14_coords": heavy[0],
        "heavy_atom14_mask": heavy[1],
        "heavy_ca_coords": heavy[2],
        "light_sequence": "S",
        "light_atom14_coords": light[0],
        "light_atom14_mask": light[1],
        "light_ca_coords": light[2],
        "antigen_sequence": "TV",
        "antigen_atom14_coords": antigen[0],
        "antigen_atom14_mask": antigen[1],
        "antigen_ca_coords": antigen[2],
        "epitope_residues": [("A", 20, "THR")],
        "paratope_residues": [("H", 10, "ALA")],
        "contact_pairs": [("H", 10, "ALA", "A", 20, "THR", np.float32(4.2))],
    }


def _native_axis_feature() -> dict:
    payload = _feature()
    placements = {
        "heavy": (0.0, 20.0),
        "light": (40.0,),
        "antigen": (4.2, 100.0),
    }
    for prefix, origins in placements.items():
        coords = np.zeros((len(origins), 14, 3), dtype=np.float32)
        mask = np.zeros((len(origins), 14), dtype=np.bool_)
        for index, origin in enumerate(origins):
            coords[index, 0] = (origin, 0.0, 0.0)
            coords[index, 1] = (origin, 1.0, 0.0)
            coords[index, 2] = (origin, 1.0, 1.0)
            mask[index, :3] = True
        payload[f"{prefix}_atom14_coords"] = coords
        payload[f"{prefix}_atom14_mask"] = mask
        payload[f"{prefix}_ca_coords"] = coords[:, 1].copy()
    return payload


def _write_feature(path: Path, payload: dict) -> None:
    root = path.stem
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}/data.pkl", pickle.dumps(payload, protocol=2))
        archive.writestr(f"{root}/version", b"3\n")


def _pdb_line(serial: int, residue: str, chain: str, number: int, xyz) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d}  CA  {residue:>3s} {chain:1s}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}           C  \n"
    )


def _pdb_atom_line(
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    number: int,
    xyz,
    *,
    insertion: str = "",
) -> str:
    x, y, z = xyz
    element = atom.lstrip("0123456789")[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain:1s}{number:4d}{insertion:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {element:>2s}  \n"
    )


def test_real_schema_round_trip_and_adapter(tmp_path: Path) -> None:
    feature_path = tmp_path / "1abc_H_L_A.pt"
    _write_feature(feature_path, _feature())
    feature = load_feature_archive(feature_path)
    pdb = tmp_path / "1abc.pdb"
    pdb.write_text(
        _pdb_line(1, "ALA", "H", 10, (0, 1, 0))
        + _pdb_line(2, "GLY", "H", 11, (1, 1, 0))
        + _pdb_line(3, "SER", "L", 15, (0, 1, 0))
        + _pdb_line(4, "THR", "A", 20, (0, 1, 0))
        + _pdb_line(5, "VAL", "A", 21, (1, 1, 0))
    )
    adapted = adapt_complex(feature, pdb)
    assert adapted["antibody_sequence"] == "AGS"
    assert adapted["antigen_sequence"] == "TV"
    np.testing.assert_array_equal(adapted["antibody_chain_ids"], [0, 0, 1])
    np.testing.assert_array_equal(adapted["antibody_labels"], [1, 0, 0])
    np.testing.assert_array_equal(adapted["antigen_labels"], [1, 0])


def test_native_axis_adapter_recomputes_labels_without_pdb() -> None:
    adapted = adapt_complex_native_axis(_native_axis_feature())
    assert adapted["antibody_sequence"] == "AGS"
    assert adapted["antigen_sequence"] == "TV"
    np.testing.assert_array_equal(adapted["antibody_chain_ids"], [0, 0, 1])
    np.testing.assert_array_equal(adapted["antibody_labels"], [1, 0, 0])
    np.testing.assert_array_equal(adapted["antigen_labels"], [1, 0])
    assert adapted["native_contact_pair_count"] == 1
    assert adapted["provided_contact_pair_count"] == 1


def test_native_axis_adapter_fails_when_contact_distance_cannot_be_reproduced() -> None:
    payload = _native_axis_feature()
    payload["contact_pairs"][0] = (*payload["contact_pairs"][0][:6], np.float32(3.1))
    with pytest.raises(ChimeraExternalError, match="distances disagree"):
        adapt_complex_native_axis(payload)


def test_native_axis_adapter_recovers_lossy_duplicate_residue_numbering() -> None:
    payload = _native_axis_feature()
    payload["heavy_sequence"] = "YY"
    payload["heavy_atom14_coords"][1, :3, 0] = 0.5
    payload["heavy_ca_coords"] = payload["heavy_atom14_coords"][:, 1].copy()
    payload["paratope_residues"] = [("H", 111, "TYR")]
    payload["contact_pairs"] = [
        ("H", 111, "TYR", "A", 20, "THR", np.float32(4.2))
    ]
    adapted = adapt_complex_native_axis(payload)
    np.testing.assert_array_equal(adapted["antibody_labels"], [1, 1, 0])
    assert adapted["native_contact_pair_count"] == 2
    assert adapted["provided_contact_pair_count"] == 1


def test_pdb_anchored_adapter_allows_feature_only_residue(tmp_path: Path) -> None:
    payload = _feature()
    payload["heavy_sequence"] = "AAG"
    heavy = _atom14(3)
    payload["heavy_atom14_coords"] = heavy[0]
    payload["heavy_atom14_mask"] = heavy[1]
    payload["heavy_ca_coords"] = heavy[2]
    pdb = tmp_path / "1abc.pdb"
    pdb.write_text(
        _pdb_line(1, "ALA", "H", 10, (0, 1, 0))
        + _pdb_line(2, "GLY", "H", 11, (2, 1, 0))
        + _pdb_line(3, "SER", "L", 15, (0, 1, 0))
        + _pdb_line(4, "THR", "A", 20, (0, 1, 0))
        + _pdb_line(5, "VAL", "A", 21, (1, 1, 0))
    )
    adapted = adapt_complex_pdb_anchored(payload, pdb)
    np.testing.assert_array_equal(adapted["antibody_labels"], [1, 0, 0, 0])
    assert adapted["feature_only_antibody_residues"] == 1
    assert adapted["unique_pair_distance_mismatch_count"] == 1


def test_pdb_anchored_adapter_uses_distance_only_for_ambiguous_tuple(
    tmp_path: Path,
) -> None:
    payload = _feature()
    payload["heavy_sequence"] = "YY"
    heavy = _atom14(2)
    heavy[0][0, :3, 0] = 0.0
    heavy[0][1, :3, 0] = 10.0
    payload["heavy_atom14_coords"] = heavy[0]
    payload["heavy_atom14_mask"] = heavy[1]
    payload["heavy_ca_coords"] = heavy[0][:, 1].copy()
    antigen = _atom14(2)
    antigen[0][0, :3, 0] = 4.2
    antigen[0][1, :3, 0] = 30.0
    payload["antigen_atom14_coords"] = antigen[0]
    payload["antigen_atom14_mask"] = antigen[1]
    payload["antigen_ca_coords"] = antigen[0][:, 1].copy()
    payload["paratope_residues"] = [("H", 111, "TYR")]
    payload["contact_pairs"] = [
        ("H", 111, "TYR", "A", 20, "THR", np.float32(4.2))
    ]
    pdb = tmp_path / "1abc.pdb"
    pdb.write_text(
        _pdb_atom_line(1, "CA", "TYR", "H", 111, (0, 1, 0), insertion="A")
        + _pdb_atom_line(2, "CA", "TYR", "H", 111, (10, 1, 0), insertion="B")
        + _pdb_atom_line(3, "CA", "SER", "L", 15, (0, 1, 0))
        + _pdb_atom_line(4, "CA", "THR", "A", 20, (4.2, 1, 0))
        + _pdb_atom_line(5, "CA", "VAL", "A", 21, (30, 1, 0))
    )
    adapted = adapt_complex_pdb_anchored(payload, pdb)
    np.testing.assert_array_equal(adapted["antibody_labels"], [1, 0, 0])
    assert adapted["distance_disambiguated_pair_count"] == 1


def test_pdb_anchored_adapter_fails_on_unresolved_ambiguous_tuple(
    tmp_path: Path,
) -> None:
    payload = _feature()
    payload["heavy_sequence"] = "YY"
    heavy = _atom14(2)
    heavy[0][0, :3, 0] = 0.0
    heavy[0][1, :3, 0] = 8.0
    payload["heavy_atom14_coords"] = heavy[0]
    payload["heavy_atom14_mask"] = heavy[1]
    payload["heavy_ca_coords"] = heavy[0][:, 1].copy()
    antigen = _atom14(2)
    antigen[0][0, :3, 0] = 4.0
    antigen[0][1, :3, 0] = 30.0
    payload["antigen_atom14_coords"] = antigen[0]
    payload["antigen_atom14_mask"] = antigen[1]
    payload["antigen_ca_coords"] = antigen[0][:, 1].copy()
    payload["paratope_residues"] = [("H", 111, "TYR")]
    payload["contact_pairs"] = [
        ("H", 111, "TYR", "A", 20, "THR", np.float32(4.0))
    ]
    pdb = tmp_path / "1abc.pdb"
    pdb.write_text(
        _pdb_atom_line(1, "CA", "TYR", "H", 111, (0, 1, 0), insertion="A")
        + _pdb_atom_line(2, "CA", "TYR", "H", 111, (8, 1, 0), insertion="B")
        + _pdb_atom_line(3, "CA", "SER", "L", 15, (0, 1, 0))
        + _pdb_atom_line(4, "CA", "THR", "A", 20, (4, 1, 0))
        + _pdb_atom_line(5, "CA", "VAL", "A", 21, (30, 1, 0))
    )
    with pytest.raises(ChimeraExternalError, match="not uniquely resolved"):
        adapt_complex_pdb_anchored(payload, pdb)


def test_feature_loader_blocks_arbitrary_pickle_global(tmp_path: Path) -> None:
    path = tmp_path / "malicious.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("malicious/data.pkl", pickle.dumps(Path("forbidden"), protocol=2))
    with pytest.raises(ChimeraExternalError, match="restricted schema"):
        load_feature_archive(path)


def test_contact_annotation_disagreement_fails_closed(tmp_path: Path) -> None:
    payload = _feature()
    payload["epitope_residues"] = [("A", 21, "VAL")]
    feature_path = tmp_path / "1abc_H_L_A.pt"
    _write_feature(feature_path, payload)
    feature = load_feature_archive(feature_path)
    pdb = tmp_path / "1abc.pdb"
    pdb.write_text(
        _pdb_line(1, "ALA", "H", 10, (0, 1, 0))
        + _pdb_line(2, "GLY", "H", 11, (1, 1, 0))
        + _pdb_line(3, "SER", "L", 15, (0, 1, 0))
        + _pdb_line(4, "THR", "A", 20, (0, 1, 0))
        + _pdb_line(5, "VAL", "A", 21, (1, 1, 0))
    )
    with pytest.raises(ChimeraExternalError, match="disagree"):
        adapt_complex(feature, pdb)


def test_split_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "temporal.json"
    path.write_text(json.dumps({"train": ["a"], "val": ["b"], "test": ["a"]}))
    with pytest.raises(ChimeraExternalError, match="overlap"):
        load_split(path)


def test_checked_remote_freeze_receipt() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = root / "evaluation" / "chimera_zero_shot_protocol_v1.json"
    receipt = root / "evaluation" / "AB03_FROZEN_PROTOCOL_RECEIPT.json"
    observed = verify_frozen_authority(protocol, receipt)
    assert observed["dataset"]["pinned_revision"] == (
        "54d5a2a1e59a329db7a6c09184652ed225188940"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        "174dd6231d934a17c0e1093f01aea784b7b1a5f6818f4fc748969327609560d8"
    )
