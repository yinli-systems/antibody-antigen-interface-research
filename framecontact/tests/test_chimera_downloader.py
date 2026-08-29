from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "download_chimera_external_subset.py"
)
_SPEC = importlib.util.spec_from_file_location("chimera_downloader_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_git_blob_sha1_matches_git_object_definition(tmp_path: Path) -> None:
    payload = b"pinned external artifact\n"
    path = tmp_path / "artifact"
    path.write_bytes(payload)
    expected = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    assert _MODULE.git_blob_sha1(path) == expected


def test_expected_digest_prefers_lfs_sha256() -> None:
    assert _MODULE.expected_digest(
        {"rfilename": "x", "blobId": "0" * 40, "lfs": {"sha256": "a" * 64}}
    ) == ("sha256", "a" * 64)
