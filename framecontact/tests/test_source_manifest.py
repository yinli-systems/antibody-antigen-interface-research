from __future__ import annotations

from pathlib import Path

import pytest

from interaction_first.sabdab2_structure import canonical_json_bytes, sha256_file
from interaction_first.source_manifest import build_source_manifest, verify_source_manifest


def test_source_manifest_detects_mutation_and_extra_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for directory in ("scripts", "src", "protocols"):
        (source / directory).mkdir(parents=True)
        (source / directory / "one.txt").write_text(directory, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            build_source_manifest(source, claim_boundary="test only")
        )
    )
    digest = sha256_file(manifest)
    assert verify_source_manifest(source, manifest, digest)["record_count"] == 3

    (source / "src" / "one.txt").write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="size differs|SHA-256 differs"):
        verify_source_manifest(source, manifest, digest)

    (source / "src" / "one.txt").write_text("src", encoding="utf-8")
    (source / "scripts" / "extra.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differs"):
        verify_source_manifest(source, manifest, digest)
