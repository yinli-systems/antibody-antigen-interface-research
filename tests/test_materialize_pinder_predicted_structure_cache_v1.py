"""Byte, sequence, and boundary tests for predicted structure caching."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "remote_tools"
    / "abpairfm_multimodal_pretraining_v1"
    / "materialize_pinder_predicted_structure_cache_v1.py"
)
SPEC = importlib.util.spec_from_file_location("predicted_cache", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def atom_line(serial: int, atom: str, residue: str, index: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} A{index:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{90.0:6.2f}          {atom[0]:>2s}\n"
    )


def pdb(sequence: str) -> bytes:
    names = {"A": "ALA", "C": "CYS"}
    lines = []
    serial = 1
    for index, residue in enumerate(sequence, 1):
        for offset, atom in enumerate(("N", "CA", "C")):
            lines.append(atom_line(serial, atom, names[residue], index, index + offset))
            serial += 1
    lines.append("END\n")
    return "".join(lines).encode("ascii")


def inventory(path: Path, sequence: str = "AC", repeated_sequence: str | None = None) -> None:
    fields = [
        "predicted_R_pdb",
        "predicted_L_pdb",
        "sequence_R",
        "sequence_L",
    ]
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=fields, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "predicted_R_pdb": "af__one.pdb",
                        "predicted_L_pdb": "af__two.pdb",
                        "sequence_R": sequence,
                        "sequence_L": sequence,
                    }
                )
                if repeated_sequence is not None:
                    writer.writerow(
                        {
                            "predicted_R_pdb": "af__one.pdb",
                            "predicted_L_pdb": "af__two.pdb",
                            "sequence_R": repeated_sequence,
                            "sequence_L": repeated_sequence,
                        }
                    )


def arguments(
    tmp_path: Path,
    *,
    expected_sequence: str = "AC",
    repeated_sequence: str | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    inventory_path = tmp_path / "inventory.tsv.gz"
    inventory(inventory_path, expected_sequence, repeated_sequence)
    archive = tmp_path / "pdbs.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("pdbs/af__one.pdb", pdb("AC"))
        handle.writestr("pdbs/af__two.pdb", pdb("AC"))
        handle.writestr("pdbs/not-allowlisted.pdb", pdb("AC"))
    return SimpleNamespace(
        inventory=inventory_path,
        inventory_sha256=MODULE.sha256_file(inventory_path),
        structure_archive=archive,
        structure_archive_sha256=MODULE.sha256_file(archive),
        structure_archive_size=archive.stat().st_size,
        expected_structure_members=2,
        output_root=tmp_path / "cache",
        output_result=tmp_path / "result.json",
    )


def test_extracts_only_allowlisted_full_predicted_monomers(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    result = MODULE.materialize(args)
    assert result["status"] == "PASS_AUTHENTICATED_PREDICTED_STRUCTURE_CACHE"
    assert result["counts"]["allowlisted_structure_members"] == 2
    assert sorted(path.name for path in (args.output_root / "pdb").iterdir()) == [
        "af__one.pdb",
        "af__two.pdb",
    ]
    assert not (args.output_root / "pdb" / "not-allowlisted.pdb").exists()
    assert result["outputs"]["holo_axis_alignment_status"] == "PENDING_SEPARATE_GATE"
    manifest = args.output_root / "PREDICTED_STRUCTURE_CACHE_MANIFEST_V1.tsv"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == result["outputs"][
        "manifest_sha256"
    ]


def test_reused_full_monomer_accepts_multiple_holo_fragments(tmp_path: Path) -> None:
    args = arguments(tmp_path, expected_sequence="A", repeated_sequence="C")
    result = MODULE.materialize(args)
    assert result["counts"]["allowlisted_structure_members"] == 2
    assert result["counts"]["unique_member_holo_sequence_associations"] == 4


def test_parse_rejects_nonconsecutive_predicted_residue_axis() -> None:
    malformed = pdb("AC").replace(b"A   2", b"A   3")
    with pytest.raises(ValueError, match="not consecutive"):
        MODULE.parse_predicted_pdb(malformed)
