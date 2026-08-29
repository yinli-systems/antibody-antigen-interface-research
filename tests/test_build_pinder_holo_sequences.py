"""Unit contracts for strict PINDER mapping-sequence publication."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pinder_holo_sequences.py"


def _module():
    specification = importlib.util.spec_from_file_location("pinder_sequences", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_raw(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "mapping_id",
        "sequence",
        "mapping_rows",
        "residue_rows",
        "resolved_residue_rows",
        "invalid_residue_indices",
        "ambiguous_residue_indices",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_publish_sequences_counts_exact_diversity(tmp_path: Path) -> None:
    module = _module()
    raw = tmp_path / "raw.tsv"
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    _write_raw(
        raw,
        [
            {
                "mapping_id": "a-R",
                "sequence": sequence,
                "mapping_rows": "22",
                "residue_rows": "20",
                "resolved_residue_rows": "19",
                "invalid_residue_indices": "0",
                "ambiguous_residue_indices": "0",
            },
            {
                "mapping_id": "b-L",
                "sequence": sequence,
                "mapping_rows": "20",
                "residue_rows": "20",
                "resolved_residue_rows": "20",
                "invalid_residue_indices": "0",
                "ambiguous_residue_indices": "0",
            },
        ],
    )
    counts = module.publish_sequences(
        raw,
        tmp_path / "sequences.fasta",
        tmp_path / "sequences.tsv.gz",
        2,
    )
    assert counts == {
        "canonical_residues": 40,
        "collapsed_mapping_rows": 2,
        "mapping_annotation_rows": 42,
        "mapping_sequences": 2,
        "maximum_sequence_length": 20,
        "minimum_sequence_length": 20,
        "nonstandard_uppercase_residues": 0,
        "sequences_longer_than_1020": 0,
        "sequences_longer_than_2048": 0,
        "sequences_shorter_than_20": 0,
        "resolved_residues": 39,
        "unknown_x_residues": 0,
        "unique_exact_sequences": 1,
    }


def test_publish_sequences_rejects_invalid_or_duplicate_residue_rows(
    tmp_path: Path,
) -> None:
    module = _module()
    raw = tmp_path / "bad.tsv"
    _write_raw(
        raw,
        [
            {
                "mapping_id": "bad-R",
                "sequence": "A" * 20,
                "mapping_rows": "20",
                "residue_rows": "20",
                "resolved_residue_rows": "20",
                "invalid_residue_indices": "1",
                "ambiguous_residue_indices": "1",
            }
        ],
    )
    try:
        module.publish_sequences(
            raw,
            tmp_path / "bad.fasta",
            tmp_path / "bad.tsv.gz",
            1,
        )
    except RuntimeError as error:
        assert "Invalid canonical residue mapping" in str(error)
    else:
        raise AssertionError("Invalid PINDER residue mapping was accepted")
