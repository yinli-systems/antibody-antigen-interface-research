from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "remote_tools"
    / "abpairfm_multimodal_pretraining_v1"
    / "materialize_predicted_backbone_features_v1.py"
)
SPEC = importlib.util.spec_from_file_location("backbone_features", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def atom_line(serial: int, atom: str, residue: str, index: int, xyz, plddt: float) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} A{index:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{plddt:6.2f}          {atom[0]:>2s}\n"
    )


def write_pdb(path: Path) -> bytes:
    residues = [("ALA", 90.0), ("CYS", 80.0), ("ASP", 70.0)]
    lines = []
    serial = 1
    for index, (residue, confidence) in enumerate(residues, 1):
        base = 3.8 * (index - 1)
        coordinates = {
            "N": (base, 0.2, 0.1),
            "CA": (base + 1.45, 0.7, 0.2),
            "C": (base + 2.90, 0.1, 0.5),
        }
        for atom in ("N", "CA", "C"):
            lines.append(atom_line(serial, atom, residue, index, coordinates[atom], confidence))
            serial += 1
    lines.append("END\n")
    payload = "".join(lines).encode("ascii")
    path.write_bytes(payload)
    return payload


def test_backbone_features_are_finite_and_axis_stable(tmp_path: Path) -> None:
    payload = write_pdb(tmp_path / "one.pdb")
    sequence, coordinates, plddt = MODULE.parse_backbone(tmp_path / "one.pdb")
    assert sequence == "ACD"
    features = MODULE.backbone_features(coordinates, plddt)
    assert features.shape == (3, 17)
    assert features.dtype == np.dtype("float32")
    assert np.isfinite(features).all()
    assert np.all(features[0, 0:2] == 0.0)
    assert np.all(features[-1, 8:16] == 0.0)
    np.testing.assert_allclose(features[:, 16], [0.9, 0.8, 0.7])
    assert hashlib.sha256(payload).hexdigest() == MODULE.sha256_file(tmp_path / "one.pdb")


def test_run_writes_flat_binary_and_index(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    pdb_root = root / "pdb"
    pdb_root.mkdir(parents=True)
    pdb_path = pdb_root / "af__one.pdb"
    payload = write_pdb(pdb_path)
    sequence = "ACD"
    sequence_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    manifest = root / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "member", "pdb_sha256", "sequence", "sequence_sha256", "residue_count"
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "member": pdb_path.name,
                "pdb_sha256": hashlib.sha256(payload).hexdigest(),
                "sequence": sequence,
                "sequence_sha256": sequence_sha,
                "residue_count": 3,
            }
        )
    args = SimpleNamespace(
        structure_root=root,
        structure_manifest=manifest,
        structure_manifest_sha256=MODULE.sha256_file(manifest),
        output_features=tmp_path / "features.f32",
        output_index=tmp_path / "index.tsv",
        output_result=tmp_path / "result.json",
    )
    result = MODULE.run(args)
    assert result["status"] == "PASS_AUTHENTICATED_PREDICTED_BACKBONE_FEATURES"
    assert args.output_features.stat().st_size == 3 * 17 * 4
    values = np.fromfile(args.output_features, dtype="<f4").reshape(3, 17)
    assert np.isfinite(values).all()
    assert json.loads(args.output_result.read_text())["test_access"]["bound_coordinates_read"] is False
