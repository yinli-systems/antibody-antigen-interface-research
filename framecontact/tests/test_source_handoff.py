from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_source_handoff.py"
MODULE_SPEC = importlib.util.spec_from_file_location("build_source_handoff", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
HANDOFF_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(HANDOFF_MODULE)
ARCHIVE_ROOT = HANDOFF_MODULE.ARCHIVE_ROOT
build_archive = HANDOFF_MODULE.build_archive


def test_source_handoff_is_deterministic_and_manifested(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("auditable\n", encoding="utf-8")
    script = source / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"not-source")
    evaluation = source / "evaluation"
    evaluation.mkdir()
    (evaluation / "benchmark-result.json").write_text("{}\n", encoding="utf-8")

    archive_a, receipt_a = build_archive(source, tmp_path / "out-a", "a.tar.gz")
    archive_b, receipt_b = build_archive(source, tmp_path / "out-b", "b.tar.gz")

    assert archive_a.read_bytes() == archive_b.read_bytes()
    receipt = json.loads(receipt_a.read_text(encoding="utf-8"))
    assert receipt["archive_sha256"] == hashlib.sha256(
        archive_a.read_bytes()
    ).hexdigest()
    assert receipt["file_count"] == 2
    assert json.loads(receipt_b.read_text(encoding="utf-8"))[
        "archive_sha256"
    ] == receipt["archive_sha256"]

    with tarfile.open(archive_a, "r:gz") as archive:
        names = archive.getnames()
        assert f"{ARCHIVE_ROOT}/README.md" in names
        assert f"{ARCHIVE_ROOT}/run.sh" in names
        assert f"{ARCHIVE_ROOT}/SOURCE_MANIFEST.json" in names
        assert not any("__pycache__" in name for name in names)
        assert not any("evaluation/" in name for name in names)
        manifest_member = archive.extractfile(
            f"{ARCHIVE_ROOT}/SOURCE_MANIFEST.json"
        )
        assert manifest_member is not None
        manifest = json.load(manifest_member)
        assert [entry["path"] for entry in manifest["files"]] == [
            "README.md",
            "run.sh",
        ]


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        (".env", b"TOKEN=not-a-real-secret\n"),
        ("notes.txt", b"-----BEGIN OPENSSH PRIVATE KEY-----\n"),
    ],
)
def test_source_handoff_refuses_secret_material(
    tmp_path: Path, name: str, payload: bytes
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / name).write_bytes(payload)

    with pytest.raises(ValueError, match="Refusing"):
        build_archive(source, tmp_path / "out", "source.tar.gz")
