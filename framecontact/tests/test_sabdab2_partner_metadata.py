"""Tests for label-free partner metadata and monomer SASA."""

from __future__ import annotations

import csv
import hashlib
from io import StringIO
from pathlib import Path

from Bio.PDB import PDBParser

from interaction_first.sabdab2_partner_metadata import (
    BASE_COLUMNS,
    PARTNER_METADATA_SPLIT_FORMAT,
    _antigen_surface_profile,
    bind_partner_metadata_to_split,
    project_split_bound_partner_metadata_to_base,
)
from interaction_first.sabdab2_structure import canonical_json_bytes


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_antigen_surface_profile_removes_non_antigen_chains() -> None:
    pdb = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.400   2.400   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.300   1.500   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.900   2.800   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.400   2.700   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.000   1.700   0.000  1.00  0.00           O
ATOM      9  CA  ALA B   1       1.450   0.000   1.000  1.00  0.00           C
END
"""
    # Convert the compact PDB to mmCIF-like model semantics by testing the core
    # implementation through a tiny monkey-patched parser payload.
    model = PDBParser(QUIET=True).get_structure("toy", StringIO(pdb))[0]

    class Parser:
        def get_structure(self, _identifier: str, _payload: StringIO):
            structure = model.get_parent().copy()
            return structure

    import interaction_first.sabdab2_partner_metadata as module

    original = module._bio_parser
    module._bio_parser = lambda: Parser()
    try:
        sequence, exposed, observed = _antigen_surface_profile(
            "toy",
            {"agchains": "A", "ag_index_list": "__FULL_CHAIN__"},
            b"ignored",
        )
    finally:
        module._bio_parser = original
    assert sequence == "AG"
    assert observed == 2
    assert exposed == 2


def test_bind_partner_metadata_to_split_is_exact(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    fieldnames = [
        "instance_id", "antigen_cluster_id", "antigen_sequence_sha256",
        "antigen_length", "surface_residue_count", "surface_observed_residue_count",
        "antigen_class", "antibody_cluster_id", "antibody_sequence_sha256",
        "antibody_length", "cdr_residue_count", "antibody_class",
        "documented_cognate_group_id",
    ]
    with base.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(3):
            writer.writerow(
                {
                    "instance_id": f"case_{index}",
                    "antigen_cluster_id": f"ag:{index}",
                    "antigen_sequence_sha256": "a" * 64,
                    "antigen_length": 100,
                    "surface_residue_count": 50,
                    "surface_observed_residue_count": 100,
                    "antigen_class": "protein",
                    "antibody_cluster_id": f"ab:{index}",
                    "antibody_sequence_sha256": "b" * 64,
                    "antibody_length": 220,
                    "cdr_residue_count": 50,
                    "antibody_class": "conventional_fv",
                    "documented_cognate_group_id": f"pdb_{index}",
                }
            )
    split = tmp_path / "split.csv"
    split.write_text(
        "instance_id,role\ncase_0,train\ncase_1,validation\ncase_2,test\n",
        encoding="utf-8",
    )
    result = bind_partner_metadata_to_split(
        base_path=base,
        base_sha256=_sha(base),
        split_path=split,
        split_sha256=_sha(split),
        output_root=tmp_path / "bound",
    )
    assert result["role_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert result["authority"]["task_labels_read"] is False


def test_project_bound_metadata_removes_only_role(tmp_path: Path) -> None:
    bound = tmp_path / "bound.csv"
    with bound.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instance_id", "role", *BASE_COLUMNS[1:]])
        writer.writeheader()
        for index, role in enumerate(("train", "validation", "test")):
            writer.writerow(
                {
                    "instance_id": f"case_{index}",
                    "role": role,
                    "antigen_cluster_id": f"ag:{index}",
                    "antigen_sequence_sha256": "a" * 64,
                    "antigen_length": 100,
                    "surface_residue_count": 50,
                    "surface_observed_residue_count": 100,
                    "antigen_class": "protein",
                    "antibody_cluster_id": f"ab:{index}",
                    "antibody_sequence_sha256": "b" * 64,
                    "antibody_length": 220,
                    "cdr_residue_count": 50,
                    "antibody_class": "conventional_fv",
                    "documented_cognate_group_id": f"pdb_{index}",
                }
            )
    receipt = tmp_path / "bound-receipt.json"
    receipt.write_bytes(
        canonical_json_bytes(
            {
                "format_version": PARTNER_METADATA_SPLIT_FORMAT,
                "metadata_sha256": _sha(bound),
                "authority": {"task_labels_read": False},
            }
        )
    )
    result = project_split_bound_partner_metadata_to_base(
        bound_path=bound,
        bound_sha256=_sha(bound),
        bound_receipt_path=receipt,
        bound_receipt_sha256=_sha(receipt),
        output_root=tmp_path / "base",
    )
    projected = tmp_path / "base" / "PARTNER_METADATA_BASE.csv"
    with projected.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == BASE_COLUMNS
        assert len(list(reader)) == 3
    assert result["authority"]["task_labels_read"] is False
    assert result["authority"]["predictions_read"] is False
    assert result["projection"]["removed_columns"] == ["role"]
