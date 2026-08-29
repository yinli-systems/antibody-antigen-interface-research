from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_ppiref50k_aux_materialization.py"


def module():
    spec = importlib.util.spec_from_file_location("ppiref_aux", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_final_http_headers_are_selected(tmp_path: Path) -> None:
    path = tmp_path / "headers"
    path.write_text(
        "HTTP/1.1 301 Moved\r\nLocation: elsewhere\r\n\r\n"
        "HTTP/2 200\r\nETag: frozen\r\nContent-Length: 12\r\n\r\n",
        encoding="ascii",
    )
    assert module().parse_final_headers(path) == {"etag": "frozen", "content-length": "12"}


def test_non_200_headers_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "headers"
    path.write_text("HTTP/2 404\r\n\r\n", encoding="ascii")
    with pytest.raises(ValueError, match="not 200"):
        module().parse_final_headers(path)


def test_seqres_validation_counts_records_and_residues(tmp_path: Path) -> None:
    path = tmp_path / "seqres.gz"
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write(">1abc_A mol:protein\nACDE\n>1abc_B mol:protein\nFGH\n")
    assert module().validate_seqres(path) == {"records": 2, "residues": 7}


def test_seqres_validation_rejects_sequence_before_header(tmp_path: Path) -> None:
    path = tmp_path / "seqres.gz"
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write("ACDE\n")
    with pytest.raises(ValueError, match="Malformed"):
        module().validate_seqres(path)


def test_recovery_transport_is_explicit_in_cli() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "authenticated_local_https_scp_recovery" in source
    assert "superseded_failed_job_id" in source
