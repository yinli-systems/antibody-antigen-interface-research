from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "remote_tools"
    / "abpairfm_multimodal_pretraining_v1"
    / "audit_foldseek_3di_v1.py"
)
SPEC = importlib.util.spec_from_file_location("foldseek_audit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _structures(path: Path) -> None:
    sequence = "AC"
    sequence_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["member", "sequence", "sequence_sha256"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {"member": "af__one.pdb", "sequence": sequence, "sequence_sha256": sequence_sha}
        )


def _arguments(tmp_path: Path):
    structures = tmp_path / "structures.tsv"
    foldseek_output = tmp_path / "foldseek.tsv"
    binary = tmp_path / "foldseek"
    _structures(structures)
    foldseek_output.write_text(
        "af__one.pdb_A\tAC\tDY\t" + ",".join(["0.0"] * 20) + "\n",
        encoding="ascii",
    )
    binary.write_bytes(b"binary")
    return SimpleNamespace(
        structure_manifest=structures,
        structure_manifest_sha256=MODULE.sha256_file(structures),
        foldseek_binary=binary,
        foldseek_binary_sha256=MODULE.sha256_file(binary),
        foldseek_output=foldseek_output,
        output_tokens=tmp_path / "tokens.tsv.gz",
        output_result=tmp_path / "result.json",
    )


def test_run_authenticates_foldseek_tokens(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    result = MODULE.run(args)
    assert result["status"] == "PASS_AUTHENTICATED_PREDICTED_STRUCTURE_3DI"
    assert result["counts"] == {"predicted_structure_members": 1}
    assert json.loads(args.output_result.read_text())["test_access"]["bound_coordinates_read"] is False


def test_parse_rejects_wrong_descriptor_axis(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    args.foldseek_output.write_text("af__one.pdb_A\tAC\tDY\t0.0\n", encoding="ascii")
    structures = MODULE.load_structures(args.structure_manifest)
    with pytest.raises(ValueError, match="dimension differs"):
        MODULE.parse_foldseek(args.foldseek_output, structures)


def test_parse_accepts_descriptor_field_above_csv_default_limit(tmp_path: Path) -> None:
    sequence = "A" * 7000
    sequence_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    structures_path = tmp_path / "structures.tsv"
    with structures_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["member", "sequence", "sequence_sha256"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "member": "af__long.pdb",
                "sequence": sequence,
                "sequence_sha256": sequence_sha,
            }
        )
    foldseek_output = tmp_path / "foldseek.tsv"
    foldseek_output.write_text(
        "af__long.pdb_A\t"
        + sequence
        + "\t"
        + ("D" * len(sequence))
        + "\t"
        + ",".join(["0.0"] * (10 * len(sequence)))
        + "\n",
        encoding="ascii",
    )

    rows = MODULE.parse_foldseek(
        foldseek_output, MODULE.load_structures(structures_path)
    )
    assert rows == [
        {
            "member": "af__long.pdb",
            "sequence_sha256": sequence_sha,
            "three_di_tokens": "D" * len(sequence),
        }
    ]
