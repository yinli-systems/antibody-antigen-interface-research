"""Focused archive-auditor tests with a tiny synthetic PPLM layout."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_pplm_dataset_archive.py"


def _module():
    spec = importlib.util.spec_from_file_location("pplm_archive_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _tiny_archive(path: Path, *, missing_sequence: bool = False) -> None:
    manifests = {
        "trainset.PDB.txt": b"1abc-assembly1_A_B\n",
        "trainset.STRING.txt": b"9606.X-1-9606.Y-2\n",
        "validset.PDB.txt": b"2def-assembly1_C_D\n",
        "validset.PDB_hetero.txt": b"2def-assembly1_C_D\n",
        "validset.PDB_homo.txt": b"",
        "validset.STRING.txt": b"9606.Z-9606.W\n",
    }
    sequences = {
        "PDB_seqs/1abc-assembly1_A.fa": b">a\nACDE\n",
        "PDB_seqs/1abc-assembly1_B.fa": b">b\nFGHI\n",
        "PDB_seqs/2def-assembly1_C.fa": b">c\nKLMN\n",
        "PDB_seqs/2def-assembly1_D.fa": b">d\nPQRS\n",
        "STRING_seqs/9606.X-1.fa": b">x\nTVWY\n",
        "STRING_seqs/9606.Y-2.fa": b">y\nACDX\n",
        "STRING_seqs/9606.Z.fa": b">z\nEFGH\n",
        "STRING_seqs/9606.W.fa": b">w\nIKLM\n",
    }
    if missing_sequence:
        del sequences["STRING_seqs/9606.W.fa"]
    with tarfile.open(path, "w:gz") as archive:
        _write_member(archive, "PPLM_Dataset/README.txt", b"fixture\n")
        for name, payload in manifests.items():
            _write_member(archive, f"PPLM_Dataset/{name}", payload)
        for name, payload in sequences.items():
            _write_member(archive, f"PPLM_Dataset/{name}", payload)


def test_auditor_counts_pairs_sequences_and_refuses_training(tmp_path: Path) -> None:
    path = tmp_path / "pplm.tar.gz"
    _tiny_archive(path)
    payload = _module().audit_archive(
        path,
        expected_size=path.stat().st_size,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert payload["status"] == "MATERIALIZED_NOT_ADMITTED"
    assert payload["authorization"]["training"] is False
    assert payload["pair_totals"]["training_exact_pair_count"] == 2
    assert payload["pair_totals"][
        "admissible_training_unique_exact_pair_count_after_missing_endpoint_exclusion_and_order_invariant_deduplication"
    ] == 2
    assert payload["pair_totals"]["validation_primary_pair_count"] == 2
    assert payload["sequences"]["sequence_file_count"] == 8
    assert payload["sequences"]["ambiguous_residue_sequence_count"] == 1
    assert payload["test_access"]["sabdab2_heldout_accessed"] is False


def test_auditor_reports_missing_pair_endpoint_and_stays_blocked(tmp_path: Path) -> None:
    path = tmp_path / "pplm-missing.tar.gz"
    _tiny_archive(path, missing_sequence=True)
    payload = _module().audit_archive(
        path,
        expected_size=path.stat().st_size,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert payload["status"] == "MATERIALIZED_WITH_SCHEMA_DEFECTS_NOT_ADMITTED"
    assert payload["schema_defects"]["missing_endpoint_count"] == 1
    assert payload["schema_defects"]["pair_with_missing_endpoint_count"] == 1
    assert payload["authorization"]["training"] is False


def test_cli_output_is_create_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "pplm.tar.gz"
    output = tmp_path / "receipt.json"
    _tiny_archive(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--archive",
            str(path),
            "--expected-size",
            str(path.stat().st_size),
            "--expected-sha256",
            digest,
            "--output",
            str(output),
        ],
    )
    _module().main()
    assert json.loads(output.read_text())["archive"]["sha256"] == digest
    with pytest.raises(FileExistsError):
        _module().main()
