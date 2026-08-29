"""Tests for case-safe Foldseek identifier normalization."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from interaction_first.foldseek_normalize import normalize_foldseek_alignments


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_normalize_maps_chain_suffixed_hashes(tmp_path: Path) -> None:
    ids = ["pdb_case_H_L", "pdb_case_h_l"]
    hashes = [hashlib.sha256(value.encode()).hexdigest() for value in ids]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["antigen_structure_id", "foldseek_structure_id"]
        )
        writer.writeheader()
        for identifier, digest in zip(ids, hashes):
            writer.writerow(
                {"antigen_structure_id": identifier, "foldseek_structure_id": digest}
            )
    raw = tmp_path / "raw.tsv"
    raw.write_text(
        f"{hashes[0]}.pdb_A\t{hashes[0]}.pdb_A\t1.0\t1.0\n"
        f"{hashes[0]}.pdb_A\t{hashes[1]}.pdb_B\t0.6\t0.7\n"
        f"{hashes[1]}.pdb_B\t{hashes[0]}.pdb_A\t0.7\t0.6\n"
        f"{hashes[1]}.pdb_B\t{hashes[1]}.pdb_B\t1.0\t1.0\n",
        encoding="utf-8",
    )
    output, receipt = tmp_path / "normalized.tsv", tmp_path / "receipt.json"
    result = normalize_foldseek_alignments(
        raw_path=raw,
        raw_sha256=_sha(raw),
        export_manifest_path=manifest,
        export_manifest_sha256=_sha(manifest),
        output_path=output,
        receipt_path=receipt,
        foldseek_version="test",
        command=["foldseek", "easy-search"],
    )
    assert output.read_text().splitlines()[2].split("\t")[:2] == ids
    assert result["structure_count"] == 2
    assert result["exhaustive_search"] is True


def test_normalize_does_not_publish_partial_output(tmp_path: Path) -> None:
    identifier = "pdb_case_H_L"
    digest = hashlib.sha256(identifier.encode()).hexdigest()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "antigen_structure_id,foldseek_structure_id\n"
        f"{identifier},{digest}\n",
        encoding="utf-8",
    )
    raw = tmp_path / "raw.tsv"
    raw.write_text(
        f"{digest}.pdb_A\t{digest}.pdb_A\t1.0\t1.0\nmalformed\n",
        encoding="utf-8",
    )
    output, receipt = tmp_path / "normalized.tsv", tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="row 2"):
        normalize_foldseek_alignments(
            raw_path=raw,
            raw_sha256=_sha(raw),
            export_manifest_path=manifest,
            export_manifest_sha256=_sha(manifest),
            output_path=output,
            receipt_path=receipt,
            foldseek_version="test",
            command=["foldseek", "easy-search"],
        )
    assert not output.exists()
    assert not receipt.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_normalize_requires_complete_unique_cartesian_grid(tmp_path: Path) -> None:
    identifiers = ["case_a", "case_b"]
    digests = [hashlib.sha256(value.encode()).hexdigest() for value in identifiers]
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "antigen_structure_id,foldseek_structure_id\n"
        + "".join(f"{identifier},{digest}\n" for identifier, digest in zip(identifiers, digests)),
        encoding="utf-8",
    )
    raw = tmp_path / "raw.tsv"
    raw.write_text(
        f"{digests[0]}\t{digests[0]}\t1\t1\n"
        f"{digests[0]}\t{digests[1]}\t0.5\t0.5\n"
        f"{digests[1]}\t{digests[0]}\t0.5\t0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete ordered Cartesian grid"):
        normalize_foldseek_alignments(
            raw_path=raw,
            raw_sha256=_sha(raw),
            export_manifest_path=manifest,
            export_manifest_sha256=_sha(manifest),
            output_path=tmp_path / "normalized.tsv",
            receipt_path=tmp_path / "receipt.json",
            foldseek_version="test",
            command=["foldseek"],
        )
    assert not (tmp_path / "normalized.tsv").exists()
