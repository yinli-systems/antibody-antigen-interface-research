"""Adversarial tests for the train-only engineering-cache controller."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Any

import pytest


torch = pytest.importorskip("torch")


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_frontier_engineering_cache.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_frontier_engineering_cache",
    SCRIPT,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MODULE.canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_sequence(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _artifact_path(root: Path, key: str) -> Path:
    path = root / key[:2] / f"{key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_antibody_artifact(root: Path, vh: str, vl: str) -> Path:
    key = _sha256_sequence(vh + vl)
    embedding = torch.arange(
        (len(vh) + len(vl)) * MODULE.EXPECTED_ANTIBODY_DIM,
        dtype=torch.float32,
    ).reshape(len(vh) + len(vl), MODULE.EXPECTED_ANTIBODY_DIM)
    path = _artifact_path(root, key)
    torch.save(
        {
            "format_version": "sabdab2_antiberty_embedding_v1",
            "sequence_key": key,
            "source_sequence_sha256": key,
            "embedding": embedding,
            "mask": torch.ones(len(vh) + len(vl), dtype=torch.bool),
            "vh_length": len(vh),
            "vl_length": len(vl),
            "embedding_dim": MODULE.EXPECTED_ANTIBODY_DIM,
            "model_name": "AntiBERTy_md_smooth",
            "model_weights_sha256": MODULE.EXPECTED_ANTIBERTY_WEIGHTS_SHA256,
        },
        path,
    )
    return path


def _write_antigen_artifact(root: Path, sequence: str) -> Path:
    key = _sha256_sequence(sequence)
    split = max(1, len(sequence) - 1)
    chunks = [
        torch.ones((split, MODULE.EXPECTED_ANTIGEN_DIM), dtype=torch.float32),
        torch.full(
            (len(sequence) - split + 1, MODULE.EXPECTED_ANTIGEN_DIM),
            3.0,
            dtype=torch.float32,
        ),
    ]
    path = _artifact_path(root, key)
    torch.save(
        {
            "format_version": "sppider_seq_embedding_v1",
            "sequence_sha256": key,
            "sequence_length": len(sequence),
            "model_name": "facebook/esm2_t33_650M_UR50D",
            "max_tokens": 1024,
            "stride": 512,
            "starts": [0, split - 1],
            "ends": [split, len(sequence)],
            "chunks": chunks,
            "model_weight_sha256": MODULE.EXPECTED_ESM_WEIGHTS_SHA256,
        },
        path,
    )
    return path


def _store_manifest(
    *,
    side: str,
    fold_id: str,
    authority_sha256: str,
    canonical_sha256: str,
    paths: list[Path],
) -> dict[str, Any]:
    records = []
    for path in paths:
        sequence_key = path.stem
        records.append(
            {
                "sequence_key": sequence_key,
                "relative_path": f"{sequence_key[:2]}/{sequence_key}.pt",
                "artifact_sha256": _sha256_file(path),
                "artifact_size_bytes": path.stat().st_size,
            }
        )
    return {
        "format_version": MODULE.STORE_INVENTORY_FORMAT,
        "data_access_scope": MODULE.DATA_ACCESS_SCOPE,
        "split_role": "train",
        "fold_id": fold_id,
        "side": side,
        "authority_manifest_sha256": authority_sha256,
        "canonical_manifest_sha256": canonical_sha256,
        "encoder_name": (
            "AntiBERTy_md_smooth"
            if side == "antibody"
            else "facebook/esm2_t33_650M_UR50D"
        ),
        "encoder_weights_sha256": (
            MODULE.EXPECTED_ANTIBERTY_WEIGHTS_SHA256
            if side == "antibody"
            else MODULE.EXPECTED_ESM_WEIGHTS_SHA256
        ),
        "embedding_dim": (
            MODULE.EXPECTED_ANTIBODY_DIM
            if side == "antibody"
            else MODULE.EXPECTED_ANTIGEN_DIM
        ),
        "artifacts": sorted(records, key=lambda value: value["sequence_key"]),
    }


def _fixture(tmp_path: Path) -> argparse.Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    authority = tmp_path / "train_authority.json"
    canonical = tmp_path / "canonical.csv"
    antibody_root = tmp_path / "antibody"
    antigen_root = tmp_path / "antigen"
    antibody_inventory = tmp_path / "antibody_store.json"
    antigen_inventory = tmp_path / "antigen_store.json"
    fold_id = "engineering-fold-0"
    header = ["INSTANCE", "split", "vh_seq", "vl_seq", "ag_seq"]
    rows = [
        ["validation-bad", "validation", "INVALID!", "", "INVALID!"],
        ["train-a", "train", "ACD", "EF", "GHIK"],
        ["test-bad", "test", "INVALID!", "", "INVALID!"],
        ["train-b", "train", "ACD", "EF", "LMNP"],
        ["train-c", "train", "QRST", "", "GHIK"],
    ]
    with canonical.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    canonical_sha256 = _sha256_file(canonical)
    _write_json(
        authority,
        {
            "format_version": MODULE.SOURCE_AUTHORITY_FORMAT,
            "data_access_scope": MODULE.DATA_ACCESS_SCOPE,
            "split_role": "train",
            "fold_id": fold_id,
            "canonical_manifest_sha256": canonical_sha256,
            "source_assignment_sha256": hashlib.sha256(
                b"independent-source-assignment"
            ).hexdigest(),
            "dataset_admission_receipt_sha256": hashlib.sha256(
                b"independent-data-admission"
            ).hexdigest(),
            "authorized_instance_ids": ["train-a", "train-b", "train-c"],
        },
    )
    authority_sha256 = _sha256_file(authority)
    antibody_paths = [
        _write_antibody_artifact(antibody_root, "ACD", "EF"),
        _write_antibody_artifact(antibody_root, "QRST", ""),
    ]
    antigen_paths = [
        _write_antigen_artifact(antigen_root, "GHIK"),
        _write_antigen_artifact(antigen_root, "LMNP"),
    ]
    _write_json(
        antibody_inventory,
        _store_manifest(
            side="antibody",
            fold_id=fold_id,
            authority_sha256=authority_sha256,
            canonical_sha256=canonical_sha256,
            paths=antibody_paths,
        ),
    )
    _write_json(
        antigen_inventory,
        _store_manifest(
            side="antigen",
            fold_id=fold_id,
            authority_sha256=authority_sha256,
            canonical_sha256=canonical_sha256,
            paths=antigen_paths,
        ),
    )
    return argparse.Namespace(
        authority_manifest=authority,
        canonical_manifest=canonical,
        antibody_embedding_root=antibody_root,
        antigen_embedding_root=antigen_root,
        antibody_store_manifest=antibody_inventory,
        antigen_store_manifest=antigen_inventory,
        output_root=tmp_path / "output",
        expected_builder_sha256=_sha256_file(SCRIPT),
        expected_authority_sha256=authority_sha256,
        expected_canonical_manifest_sha256=canonical_sha256,
        expected_antibody_store_manifest_sha256=_sha256_file(
            antibody_inventory
        ),
        expected_antigen_store_manifest_sha256=_sha256_file(antigen_inventory),
    )


def _replace(args: argparse.Namespace, **changes: Any) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(args), **changes})


def _assert_no_staging(tmp_path: Path) -> None:
    assert not list(tmp_path.glob(".*.building.*"))


def test_builds_exact_authorization_equal_cache_without_parsing_sealed_rows(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    receipt = MODULE.build_cache(args)
    authorization_path = args.output_root / MODULE.AUTHORIZATION_FILENAME
    cache_manifest_path = (
        args.output_root / "fold_cache" / MODULE.CACHE_MANIFEST_FILENAME
    )
    authorization = json.loads(authorization_path.read_text())
    manifest = json.loads(cache_manifest_path.read_text())

    assert receipt["status"] == "complete"
    assert receipt["selection"]["authorized_instance_count"] == 3
    assert receipt["selection"]["selected_instance_count"] == 3
    assert receipt["selection"]["authorization_equal"] is True
    assert manifest["whole_pool_materialized"] is False
    assert manifest["positive_pairs"] == authorization["positive_pairs"]
    assert [value["instance_id"] for value in manifest["instances"]] == [
        "train-a",
        "train-b",
        "train-c",
    ]
    assert all(not value["contact_supervision"] for value in manifest["instances"])
    assert receipt["claim_boundary"] == {
        "engineering_data_path_verified": True,
        "scientific_training_evidence": False,
        "model_selection_evidence": False,
        "validation_or_test_evidence": False,
        "biological_performance_evidence": False,
        "state_of_the_art_evidence": False,
    }
    assert receipt["embedding_sources"]["hash_before_deserialization"] is True
    assert receipt["embedding_sources"]["antibody"]["artifact_count"] == 2
    assert receipt["embedding_sources"]["antigen"]["artifact_count"] == 2
    assert receipt["artifacts"]["example_artifact_count"] == 3
    assert receipt["immutability_policy"]["expected_directory_count"] == 3
    assert receipt["immutability_policy"]["expected_regular_file_count"] == 6
    assert not any(
        token in json.dumps(receipt)
        for token in (str(tmp_path), "validation-bad", "test-bad")
    )
    for path in [args.output_root, *args.output_root.rglob("*")]:
        expected_mode = 0o555 if path.is_dir() else 0o444
        assert stat.S_IMODE(path.lstat().st_mode) == expected_mode
        assert not path.is_symlink()
    _assert_no_staging(tmp_path)

    from interaction_first.frontier_data import load_fold_local_cache_contract

    contract = load_fold_local_cache_contract(
        args.output_root / "fold_cache",
        authorization_path,
        expected_authorization_manifest_sha256=(
            receipt["artifacts"]["authorization_manifest_sha256"]
        ),
        expected_cache_manifest_sha256=(
            receipt["artifacts"]["cache_manifest_sha256"]
        ),
    )
    assert contract.instance_count == 3
    assert contract.authorization_external_anchor_verified
    assert contract.cache_external_anchor_verified


def test_every_external_anchor_is_required_and_fail_closed(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    for field in (
        "expected_builder_sha256",
        "expected_authority_sha256",
        "expected_canonical_manifest_sha256",
        "expected_antibody_store_manifest_sha256",
        "expected_antigen_store_manifest_sha256",
    ):
        changed = _replace(
            args,
            output_root=tmp_path / f"output-{field}",
            **{field: "0" * 64},
        )
        with pytest.raises(MODULE.CacheBuildError, match="SHA-256"):
            MODULE.build_cache(changed)
        assert not changed.output_root.exists()


def test_authority_rejects_validation_or_test_membership_fields(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    authority = json.loads(args.authority_manifest.read_text())
    authority["validation_instance_ids"] = ["validation-bad"]
    _write_json(args.authority_manifest, authority)
    changed = _replace(
        args,
        expected_authority_sha256=_sha256_file(args.authority_manifest),
    )
    with pytest.raises(MODULE.CacheBuildError, match="fields differ"):
        MODULE.build_cache(changed)
    assert not args.output_root.exists()
    _assert_no_staging(tmp_path)


def test_unselected_sequence_bytes_are_never_decoded_or_csv_parsed() -> None:
    raw = (
        b"INSTANCE,split,vh_seq,vl_seq,ag_seq\n"
        b"validation-hidden,validation,\xff\xfe,broken,broken\n"
        b"train-a,train,ACD,EF,GHIK\n"
        b"test-hidden,test,\x80\x81,broken,broken\n"
    )
    rows = MODULE.load_train_rows(raw, ("train-a",))
    assert len(rows) == 1
    assert rows[0].instance_id == "train-a"
    assert rows[0].vh_sequence == "ACD"


def test_unselected_multiline_csv_cannot_forge_an_authorized_physical_row() -> None:
    raw = (
        b"INSTANCE,split,vh_seq,vl_seq,ag_seq\n"
        b'validation-hidden,validation,"AAAA\n'
        b"train-a,train,ACD,EF,GHIK\n"
        b'CCCC",,LMNP\n'
    )
    with pytest.raises(MODULE.CacheBuildError, match="span physical lines"):
        MODULE.load_train_rows(raw, ("train-a",))


def test_duplicate_json_keys_and_noncanonical_inventory_fail_closed(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    args.authority_manifest.write_text(
        '{"format_version":"x","format_version":"y"}\n',
        encoding="utf-8",
    )
    changed = _replace(
        args,
        expected_authority_sha256=_sha256_file(args.authority_manifest),
    )
    with pytest.raises(MODULE.CacheBuildError, match="Duplicate JSON key"):
        MODULE.build_cache(changed)

    args = _fixture(tmp_path / "extra")
    inventory = json.loads(args.antibody_store_manifest.read_text())
    extra = deepcopy(inventory["artifacts"][0])
    extra["sequence_key"] = "f" * 64
    extra["relative_path"] = f"ff/{'f' * 64}.pt"
    inventory["artifacts"].append(extra)
    inventory["artifacts"].sort(key=lambda value: value["sequence_key"])
    _write_json(args.antibody_store_manifest, inventory)
    changed = _replace(
        args,
        expected_antibody_store_manifest_sha256=_sha256_file(
            args.antibody_store_manifest
        ),
    )
    with pytest.raises(MODULE.CacheBuildError, match="allowlist-equal"):
        MODULE.build_cache(changed)


def test_embedding_hash_is_checked_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture(tmp_path)
    inventory = json.loads(args.antibody_store_manifest.read_text())
    selected = next(
        value
        for value in inventory["artifacts"]
        if value["sequence_key"] == _sha256_sequence("ACDEF")
    )
    artifact = (
        args.antibody_embedding_root
        / Path(*PurePosixPath(selected["relative_path"]).parts)
    )
    original = artifact.read_bytes()
    artifact.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    called = False

    def forbidden_deserialize(*values: object, **keywords: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("Unauthenticated bytes reached deserialization")

    monkeypatch.setattr(MODULE, "load_antibody_embedding", forbidden_deserialize)
    with pytest.raises(MODULE.CacheBuildError, match="before deserialization"):
        MODULE.build_cache(args)
    assert not called
    assert not args.output_root.exists()
    _assert_no_staging(tmp_path)


def test_symlink_and_special_file_inputs_are_rejected(tmp_path: Path) -> None:
    args = _fixture(tmp_path / "authority-symlink")
    authority_target = args.authority_manifest.with_suffix(".real")
    args.authority_manifest.replace(authority_target)
    args.authority_manifest.symlink_to(authority_target)
    with pytest.raises((MODULE.CacheBuildError, OSError)):
        MODULE.build_cache(args)
    assert not args.output_root.exists()

    args = _fixture(tmp_path / "symlink-root")
    real_root = args.antibody_embedding_root
    linked_root = real_root.parent / "antibody-link"
    linked_root.symlink_to(real_root, target_is_directory=True)
    changed = _replace(args, antibody_embedding_root=linked_root)
    with pytest.raises((MODULE.CacheBuildError, OSError)):
        MODULE.build_cache(changed)
    assert not changed.output_root.exists()

    args = _fixture(tmp_path / "symlink-artifact")
    inventory = json.loads(args.antibody_store_manifest.read_text())
    selected = inventory["artifacts"][0]
    artifact = (
        args.antibody_embedding_root
        / Path(*PurePosixPath(selected["relative_path"]).parts)
    )
    target = artifact.with_suffix(".real")
    artifact.replace(target)
    artifact.symlink_to(target)
    with pytest.raises((MODULE.CacheBuildError, OSError)):
        MODULE.build_cache(args)

    if hasattr(os, "mkfifo"):
        args = _fixture(tmp_path / "special")
        inventory = json.loads(args.antigen_store_manifest.read_text())
        selected = inventory["artifacts"][0]
        artifact = (
            args.antigen_embedding_root
            / Path(*PurePosixPath(selected["relative_path"]).parts)
        )
        artifact.unlink()
        os.mkfifo(artifact)
        with pytest.raises(MODULE.CacheBuildError, match="regular file"):
            MODULE.build_cache(args)


def test_output_parent_symlink_is_rejected_before_staging(tmp_path: Path) -> None:
    args = _fixture(tmp_path / "output-parent")
    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    changed = _replace(args, output_root=linked_parent / "output")
    with pytest.raises((MODULE.CacheBuildError, OSError)):
        MODULE.build_cache(changed)
    assert not list(real_parent.iterdir())


def test_selected_row_failure_cleans_private_staging(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    lines = args.canonical_manifest.read_text(encoding="utf-8").splitlines()
    lines = [
        line.replace(
            "train-a,train,ACD,EF,GHIK",
            "train-a,train,INVALID!,EF,GHIK",
        )
        for line in lines
    ]
    args.canonical_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    canonical_sha256 = _sha256_file(args.canonical_manifest)
    authority = json.loads(args.authority_manifest.read_text())
    authority["canonical_manifest_sha256"] = canonical_sha256
    _write_json(args.authority_manifest, authority)
    changed = _replace(
        args,
        expected_canonical_manifest_sha256=canonical_sha256,
        expected_authority_sha256=_sha256_file(args.authority_manifest),
    )
    with pytest.raises(MODULE.CacheBuildError, match="invalid residues"):
        MODULE.build_cache(changed)
    assert not args.output_root.exists()
    _assert_no_staging(tmp_path)


def test_publication_is_create_only_and_never_reuses_existing_target(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    MODULE.build_cache(args)
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        MODULE.build_cache(args)

    args = _fixture(tmp_path / "preexisting")
    args.output_root.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        MODULE.build_cache(args)
    assert args.output_root.read_text(encoding="utf-8") == "sentinel\n"


def test_kernel_noreplace_primitive_cannot_overwrite_a_racing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    descriptor = MODULE._open_directory_nofollow(tmp_path)
    try:
        with pytest.raises(OSError) as error:
            MODULE._rename_noreplace(descriptor, source.name, target.name)
    finally:
        os.close(descriptor)
    assert error.value.errno in {errno.EEXIST, errno.ENOTEMPTY}
    assert source.read_bytes() == b"source"
    assert target.read_bytes() == b"target"


def test_cli_has_no_selection_or_implicit_anchor_bypass() -> None:
    with pytest.raises(SystemExit):
        MODULE.parse_args([])
    parsed = MODULE.parse_args(
        [
            "--authority-manifest",
            "authority.json",
            "--canonical-manifest",
            "canonical.csv",
            "--antibody-embedding-root",
            "antibody",
            "--antigen-embedding-root",
            "antigen",
            "--antibody-store-manifest",
            "antibody.json",
            "--antigen-store-manifest",
            "antigen.json",
            "--output-root",
            "output",
            "--expected-builder-sha256",
            "1" * 64,
            "--expected-authority-sha256",
            "2" * 64,
            "--expected-canonical-manifest-sha256",
            "3" * 64,
            "--expected-antibody-store-manifest-sha256",
            "4" * 64,
            "--expected-antigen-store-manifest-sha256",
            "5" * 64,
        ]
    )
    assert not hasattr(parsed, "max_unique_pairs")
    assert not hasattr(parsed, "selection_seed")
