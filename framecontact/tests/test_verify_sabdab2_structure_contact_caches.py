from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_sabdab2_structure_contact_caches.py"
SPEC = importlib.util.spec_from_file_location("verify_sabdab2_caches", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_inventory_rejects_extra_artifact(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    artifact = root / "artifacts" / "aa" / "expected.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"expected")
    extra = artifact.with_name("extra.pt")
    extra.write_bytes(b"extra")
    records = [{"artifact_path": "artifacts/aa/expected.pt"}]

    try:
        MODULE._inventory(root, records, ".pt")
    except ValueError as error:
        assert "inventory differs" in str(error)
    else:
        raise AssertionError("extra artifact should fail closed")


def test_inventory_rejects_appledouble_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    artifact = root / "artifacts" / "aa" / "expected.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"expected")
    artifact.with_name("._expected.npz").write_bytes(b"sidecar")
    records = [{"artifact_path": "artifacts/aa/expected.npz"}]

    try:
        MODULE._inventory(root, records, ".npz")
    except ValueError as error:
        assert "AppleDouble" in str(error)
    else:
        raise AssertionError("AppleDouble sidecar should fail closed")
