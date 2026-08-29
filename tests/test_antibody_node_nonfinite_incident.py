from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import freeze_antibody_node_nonfinite_incident as incident


def _write_manifest(root: Path, name: str, payload: bytes) -> None:
    path = root / name
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (root / "artifact_sha256.txt").write_text(
        f"{digest}  {path}\n", encoding="ascii"
    )


def test_verify_manifest_and_environment_fields(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "evidence.log", b"finite\n")
    assert incident.verify_manifest(tmp_path)["artifact_count"] == 1
    environment = tmp_path / "environment.txt"
    environment.write_text("job_id=17\nnode=gpu-a\n", encoding="utf-8")
    assert incident.environment_fields(environment) == {
        "job_id": "17",
        "node": "gpu-a",
    }


def test_verify_manifest_rejects_drift(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "evidence.log", b"finite\n")
    (tmp_path / "evidence.log").write_bytes(b"drift\n")
    with pytest.raises(RuntimeError, match="hash differs"):
        incident.verify_manifest(tmp_path)


def test_require_text_checks_positive_and_negative_signatures(tmp_path: Path) -> None:
    log = tmp_path / "diagnostic.log"
    log.write_text("two updates without reproduction\n", encoding="utf-8")
    incident.require_text(log, ("without reproduction",), ("NONFINITE",))
    with pytest.raises(RuntimeError, match="Diagnostic signature differs"):
        incident.require_text(log, ("missing",))
