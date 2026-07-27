"""Tests for authenticated and create-only public-source materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import interaction_first.source_materializer as materializer
from interaction_first.data_registry import load_source_registry
from interaction_first.source_materializer import (
    DownloadPlan,
    SourceMaterializationError,
    derive_download_artifacts,
    load_download_plan,
    materialize_registered_source,
    validate_plan_against_registry,
    verify_materialized_registered_source,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _git_blob_sha1(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


def _synthetic_huggingface_metadata(revision: str) -> tuple[bytes, dict[str, bytes]]:
    payloads = {"README.md": b"hello", "weights.bin": b"data"}
    metadata = {
        "sha": revision,
        "siblings": [
            {
                "rfilename": "README.md",
                "size": len(payloads["README.md"]),
                "blobId": _git_blob_sha1(payloads["README.md"]),
                "lfs": None,
            },
            {
                "rfilename": "weights.bin",
                "size": len(payloads["weights.bin"]),
                "blobId": "0" * 40,
                "lfs": {
                    "sha256": hashlib.sha256(payloads["weights.bin"]).hexdigest(),
                    "size": len(payloads["weights.bin"]),
                },
            },
        ],
    }
    return _canonical_bytes(metadata), payloads


def _synthetic_plan(
    *,
    metadata_bytes: bytes,
    source_id: str = "synthetic_source",
    revision: str = "a" * 40,
) -> DownloadPlan:
    return DownloadPlan(
        path=Path("synthetic.json"),
        source_id=source_id,
        provider="huggingface_dataset",
        pinned_revision=revision,
        metadata_url="https://example.org/metadata",
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        selectors=("*.bin", "README.md"),
        expected_selected_file_count=2,
        expected_selected_bytes=9,
        locator={"dataset_repo_id": "owner/dataset"},
        file_sha256="f" * 64,
    )


def test_checked_download_plans_match_registered_revisions() -> None:
    root = _repository_root()
    plans = root / "data_registry" / "download_plans"
    sources = root / "data_registry" / "sources"
    observed = []
    for plan_path in sorted(plans.glob("*.json")):
        plan = load_download_plan(plan_path)
        registry = load_source_registry(sources / plan.source_id)
        validate_plan_against_registry(plan, registry)
        observed.append(plan.source_id)
    assert observed == [
        "avida_hil6_2024_11",
        "chimera_bench_2026_05",
        "ppiref_zenodo_14845086",
        "sabdab2_ml_zenodo_20083995",
    ]


def test_huggingface_derivation_authenticates_git_and_lfs_files() -> None:
    revision = "a" * 40
    metadata_bytes, _ = _synthetic_huggingface_metadata(revision)
    plan = _synthetic_plan(metadata_bytes=metadata_bytes, revision=revision)
    artifacts = derive_download_artifacts(plan, metadata_bytes)
    assert [value.relative_path for value in artifacts] == [
        "README.md",
        "weights.bin",
    ]
    assert artifacts[0].upstream_algorithm == "git_blob_sha1"
    assert artifacts[1].upstream_algorithm == "sha256"
    with pytest.raises(SourceMaterializationError, match="metadata SHA-256 differs"):
        derive_download_artifacts(plan, metadata_bytes + b" ")


def test_metadata_fetch_is_bounded_and_authenticates_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    metadata_bytes, _ = _synthetic_huggingface_metadata(revision)
    plan = _synthetic_plan(metadata_bytes=metadata_bytes, revision=revision)
    observed_command: list[str] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        stdout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        assert check
        assert stdout == subprocess.PIPE
        observed_command.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout=metadata_bytes)

    monkeypatch.setattr(materializer.subprocess, "run", fake_run)
    assert materializer.fetch_provider_metadata(plan) == metadata_bytes
    assert observed_command[observed_command.index("--connect-timeout") + 1] == "30"
    assert observed_command[observed_command.index("--max-time") + 1] == "300"


def test_zenodo_derivation_rejects_selection_drift() -> None:
    metadata = {
        "id": 123,
        "files": [
            {
                "key": "archive.zip",
                "size": 11,
                "checksum": "md5:" + hashlib.md5(
                    b"archive.zip", usedforsecurity=False
                ).hexdigest(),
                "links": {"self": "https://example.org/archive.zip"},
            }
        ],
    }
    metadata_bytes = _canonical_bytes(metadata)
    plan = DownloadPlan(
        path=Path("zenodo.json"),
        source_id="zenodo_source",
        provider="zenodo_record",
        pinned_revision="zenodo-record-123",
        metadata_url="https://example.org/metadata",
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        selectors=("archive.zip",),
        expected_selected_file_count=1,
        expected_selected_bytes=10,
        locator={"record_id": 123},
        file_sha256="e" * 64,
    )
    with pytest.raises(SourceMaterializationError, match="byte count differs"):
        derive_download_artifacts(plan, metadata_bytes)


def test_create_only_materialization_verifies_bytes_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "cd783d573e015f83364e42287810ad597850aeed"
    metadata_bytes, payloads = _synthetic_huggingface_metadata(revision)
    source_id = "avida_hil6_2024_11"
    plan_payload = {
        "expected_selected_bytes": 9,
        "expected_selected_file_count": 2,
        "format_version": "pairscale_download_plan_v1",
        "locator": {"dataset_repo_id": "owner/dataset"},
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "metadata_url": "https://example.org/metadata",
        "pinned_revision": revision,
        "provider": "huggingface_dataset",
        "selectors": ["*.bin", "README.md"],
        "source_id": source_id,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(_canonical_bytes(plan_payload))
    registry_root = tmp_path / "sources"
    registered_source = registry_root / source_id
    registered_source.mkdir(parents=True)
    source_fixture = (
        _repository_root() / "data_registry" / "sources" / source_id
    )
    for path in source_fixture.iterdir():
        (registered_source / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(
        materializer,
        "fetch_provider_metadata",
        lambda plan, curl_binary: metadata_bytes,
    )

    def fake_curl(binary: str, url: str, partial: Path) -> None:
        del binary
        filename = url.split("?", maxsplit=1)[0].rsplit("/", maxsplit=1)[-1]
        partial.write_bytes(payloads[filename])

    monkeypatch.setattr(materializer, "_run_curl", fake_curl)
    destination_root = tmp_path / "raw"
    inventory = materialize_registered_source(
        plan_path=plan_path,
        registry_root=registry_root,
        destination_root=destination_root,
        workers=2,
    )
    assert [value["path"] for value in inventory["files"]] == [
        ".authority/provider_metadata.json",
        "README.md",
        "weights.bin",
    ]
    source_root = destination_root / source_id
    assert json.loads((source_root / "raw_sha256.json").read_text()) == inventory
    seeded_destination = tmp_path / "seeded-raw"

    def reject_download(binary: str, url: str, partial: Path) -> None:
        del binary, url, partial
        pytest.fail("fully verified seed materialization must not download")

    monkeypatch.setattr(materializer, "_run_curl", reject_download)
    seeded_inventory = materialize_registered_source(
        plan_path=plan_path,
        registry_root=registry_root,
        destination_root=seeded_destination,
        workers=2,
        verified_seed_root=source_root,
    )
    assert [value["sha256"] for value in seeded_inventory["files"]] == [
        value["sha256"] for value in inventory["files"]
    ]
    assert (
        seeded_destination / source_id / "weights.bin"
    ).stat().st_ino != (source_root / "weights.bin").stat().st_ino
    monkeypatch.setattr(materializer, "_run_curl", fake_curl)
    verification = verify_materialized_registered_source(
        plan_path=plan_path,
        registry_root=registry_root,
        source_root=source_root,
    )
    assert verification["data_file_count"] == 2
    assert verification["total_data_bytes"] == 9
    (source_root / "unexpected.txt").write_text("unexpected\n")
    with pytest.raises(SourceMaterializationError, match="differ from raw inventory"):
        verify_materialized_registered_source(
            plan_path=plan_path,
            registry_root=registry_root,
            source_root=source_root,
        )
    (source_root / "unexpected.txt").unlink()
    (source_root / "weights.bin").write_bytes(b"evil")
    downloaded: list[str] = []

    def record_download(binary: str, url: str, partial: Path) -> None:
        filename = url.split("?", maxsplit=1)[0].rsplit("/", maxsplit=1)[-1]
        downloaded.append(filename)
        fake_curl(binary, url, partial)

    monkeypatch.setattr(materializer, "_run_curl", record_download)
    recovered_destination = tmp_path / "recovered-raw"
    recovered_inventory = materialize_registered_source(
        plan_path=plan_path,
        registry_root=registry_root,
        destination_root=recovered_destination,
        workers=2,
        verified_seed_root=source_root,
    )
    assert downloaded == ["weights.bin"]
    assert [value["sha256"] for value in recovered_inventory["files"]] == [
        value["sha256"] for value in inventory["files"]
    ]
    with pytest.raises(SourceMaterializationError, match="digest differs"):
        materialize_registered_source(
            plan_path=plan_path,
            registry_root=registry_root,
            destination_root=destination_root,
        )
