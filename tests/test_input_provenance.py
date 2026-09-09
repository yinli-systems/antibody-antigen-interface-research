"""The run receipts must say which data the run read.

The receipts named every hyperparameter and the split manifest digest but no
input directory. Recovering the inputs of a finished ten-seed replication
therefore meant matching the split file by digest across the filesystem and the
embedding store by tensor width -- and the store the width identified was not
the one the directory name suggested, so a plausible guess produced a run that
died on a shape check rather than a wrong number.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "sabdab2_interaction_first_under_test",
    REPOSITORY_ROOT / "scripts" / "sabdab2_interaction_first.py",
)
assert _SPEC is not None and _SPEC.loader is not None
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

directory_fingerprint = _MODULE.directory_fingerprint
input_provenance = _MODULE.input_provenance


def _arguments(root: Path) -> argparse.Namespace:
    prepare = root / "prepare"
    contacts = root / "contacts"
    embeddings = root / "embeddings"
    for directory in (prepare, contacts, embeddings):
        directory.mkdir()
    (prepare / "instances.parquet").write_bytes(b"not really a parquet")
    (contacts / "a.npz").write_bytes(b"a")
    (contacts / "b.npz").write_bytes(b"b")
    (embeddings / "00").mkdir()
    (embeddings / "00" / "deadbeef.pt").write_bytes(b"tensor")
    manifest = root / "split.json"
    manifest.write_text(json.dumps({"train": []}), encoding="utf-8")
    return argparse.Namespace(
        prepare_dir=prepare,
        contact_dir=contacts,
        embedding_dir=embeddings,
        split_manifest=manifest,
    )


def test_every_input_directory_is_named(tmp_path: Path) -> None:
    """The failure this exists for: a receipt that does not say what it read."""
    provenance = input_provenance(_arguments(tmp_path))

    for key in ("prepare_dir", "contact_dir", "embedding_dir", "split_manifest"):
        assert key in provenance
        assert Path(provenance[key]["path"]).is_absolute()


def test_paths_are_resolved_not_relative(tmp_path: Path, monkeypatch) -> None:
    """A relative path in a receipt identifies nothing once the shell is gone."""
    arguments = _arguments(tmp_path)
    monkeypatch.chdir(tmp_path)
    arguments.contact_dir = Path("contacts")

    provenance = input_provenance(arguments)

    assert provenance["contact_dir"]["path"] == str((tmp_path / "contacts").resolve())


def test_the_split_manifest_is_identified_by_content(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    before = input_provenance(arguments)["split_manifest"]["sha256"]

    arguments.split_manifest.write_text(json.dumps({"train": ["x"]}), encoding="utf-8")
    after = input_provenance(arguments)["split_manifest"]["sha256"]

    assert before != after
    assert len(before) == 64


def test_the_instance_table_is_identified_by_content(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    before = input_provenance(arguments)["prepare_dir"]["instances_parquet_sha256"]

    (arguments.prepare_dir / "instances.parquet").write_bytes(b"different bytes")
    after = input_provenance(arguments)["prepare_dir"]["instances_parquet_sha256"]

    assert before != after


def test_a_missing_instance_table_records_none_rather_than_raising(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    (arguments.prepare_dir / "instances.parquet").unlink()

    provenance = input_provenance(arguments)

    assert provenance["prepare_dir"]["instances_parquet_sha256"] is None


def test_shard_directories_are_counted_recursively(tmp_path: Path) -> None:
    """Embedding stores shard by digest prefix, so a flat count would read zero."""
    arguments = _arguments(tmp_path)

    fingerprint = directory_fingerprint(arguments.embedding_dir)

    assert fingerprint["exists"] is True
    assert fingerprint["file_count"] == 1


def test_two_stores_of_different_size_are_distinguishable(tmp_path: Path) -> None:
    """The two candidate embedding stores differed in width, and in file count."""
    arguments = _arguments(tmp_path)
    other = tmp_path / "other_embeddings"
    (other / "00").mkdir(parents=True)
    (other / "00" / "deadbeef.pt").write_bytes(b"tensor")
    (other / "00" / "cafe.pt").write_bytes(b"tensor")

    assert (
        directory_fingerprint(arguments.embedding_dir)["file_count"]
        != directory_fingerprint(other)["file_count"]
    )


def test_a_missing_directory_is_recorded_not_fatal(tmp_path: Path) -> None:
    fingerprint = directory_fingerprint(tmp_path / "absent")

    assert fingerprint["exists"] is False
    assert "file_count" not in fingerprint


def test_the_receipt_states_what_it_does_not_cover(tmp_path: Path) -> None:
    """File counts are not content digests, and the receipt must not imply they are."""
    provenance = input_provenance(_arguments(tmp_path))

    assert "not by content" in provenance["coverage"]


def test_the_provenance_block_is_json_serializable(tmp_path: Path) -> None:
    """It is written into receipts, so a Path leaking through would break them."""
    provenance = input_provenance(_arguments(tmp_path))

    assert json.loads(json.dumps(provenance)) == provenance


@pytest.mark.parametrize("key", ["contact_dir", "embedding_dir"])
def test_large_stores_are_not_hashed(tmp_path: Path, key: str) -> None:
    """A content digest of a gigabyte store on every run would not be affordable."""
    provenance = input_provenance(_arguments(tmp_path))

    assert "sha256" not in provenance[key]
