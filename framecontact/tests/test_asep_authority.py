"""Fail-closed tests for the prospective AsEP external authority."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import zipfile

import pytest


REVISION = "2b653d7bb5aea57c156b90297e3adf6afa290452"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _digests(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), hashlib.md5(payload).hexdigest()


def _materialize(archive: Path, output: Path, **overrides: object) -> dict:
    from interaction_first.asep_authority import materialize_asep_authority

    sha256, md5 = _digests(archive)
    arguments = {
        "archive_path": archive,
        "archive_sha256": sha256,
        "archive_md5": md5,
        "expected_size_bytes": archive.stat().st_size,
        "source_code_revision": REVISION,
        "output_root": output,
        "maximum_members": 100,
        "maximum_uncompressed_bytes": 1_000_000,
    }
    arguments.update(overrides)
    return materialize_asep_authority(**arguments)


def _write_inner_tar(
    path: Path,
    root: str,
    extension: str,
    identifiers: tuple[str, ...],
    *,
    add_symlink: bool = False,
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        directory = tarfile.TarInfo(root)
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for identifier in identifiers:
            payload = b"opaque"
            member = tarfile.TarInfo(f"{root}/{identifier}{extension}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        if add_symlink:
            link = tarfile.TarInfo(f"{root}/link{extension}")
            link.type = tarfile.SYMTYPE
            link.linkname = "../outside"
            archive.addfile(link)


def _inner_authority(
    root: Path,
    *,
    identifiers: tuple[str, ...] = ("1abc_0P", "2def_1P"),
    graph_ids: tuple[str, ...] | None = None,
    structure_ids: tuple[str, ...] | None = None,
    graph_symlink: bool = False,
) -> Path:
    if graph_ids is None:
        graph_ids = identifiers
    if structure_ids is None:
        structure_ids = identifiers
    payload = root / "payload/asepv1.2.0"
    payload.mkdir(parents=True)
    (payload / "asepv1-AbDb-IDs.txt").write_text(
        "".join(f"{identifier}\n" for identifier in identifiers), encoding="ascii"
    )
    (payload / "split_dict.pt").write_bytes(b"opaque-split")
    _write_inner_tar(
        payload / "asepv1_interim_graphs.tar.gz",
        "asepv1_interim_graphs",
        ".pt",
        graph_ids,
        add_symlink=graph_symlink,
    )
    _write_inner_tar(
        payload / "structures.tar.gz",
        "structures",
        ".pdb",
        structure_ids,
    )
    receipt = root / "ASEP_AUTHORITY_RECEIPT.json"
    receipt.write_text(
        json.dumps(
            {
                "format_version": "framecontact_asep_external_authority_v1",
                "authority": {"task_predictions_read": False},
                "safe_extraction_checks": {
                    "dataset_lock_member_set_verified": True
                },
            }
        ),
        encoding="utf-8",
    )
    return receipt


def test_materialize_asep_authority_is_atomic_and_receipted(tmp_path: Path) -> None:
    archive = tmp_path / "asep.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("asep-dataset/", b"")
        handle.writestr("asep-dataset/split/test.txt", b"example\n")
    output = tmp_path / "authority"
    result = _materialize(archive, output)
    assert (output / "payload/asep-dataset/split/test.txt").read_text() == "example\n"
    receipt = json.loads((output / "ASEP_AUTHORITY_RECEIPT.json").read_text())
    inventory = json.loads((output / "ZIP_MEMBER_MANIFEST.json").read_text())
    assert result["format_version"] == "framecontact_asep_external_authority_v1"
    assert receipt["authority"]["task_predictions_read"] is False
    assert receipt["authority"]["official_source_code_revision"] == REVISION
    assert receipt["member_count"] == 2
    assert [entry["path"] for entry in inventory] == [
        "asep-dataset",
        "asep-dataset/split/test.txt",
    ]
    with pytest.raises(FileExistsError):
        _materialize(archive, output)


def test_repository_asep_lock_is_authenticated_and_pre_access() -> None:
    from interaction_first.asep_authority import load_asep_dataset_lock

    lock_path = REPOSITORY_ROOT / "baselines/ASEP_DATASET_LOCK.json"
    lock = load_asep_dataset_lock(lock_path, _digests(lock_path)[0])
    assert lock["dataset"] == {
        "name": "AsEP",
        "version": "v1.2.0",
        "record_doi": "10.5281/zenodo.11495514",
        "record_url": "https://zenodo.org/records/11495514",
        "download_url": "https://zenodo.org/records/11495514/files/asep-dataset.zip?download=1",
        "archive_size_bytes": 2638740834,
        "publisher_md5": "ad0055009ed45bdca7453f1fe58d953b",
        "archive_sha256": "07889a0eab3dd5efbdf4ad141497d2287cbd9350bd95212fc9bb0a2028c41cff",
    }
    assert lock["evidence_boundary_at_freeze"]["task_labels_read"] is False
    assert len(lock["expected_archive_members"]) == 5


def test_materialize_asep_authority_enforces_locked_member_set(tmp_path: Path) -> None:
    archive = tmp_path / "asep.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("payload.txt", b"content")
    output = tmp_path / "authority"
    with pytest.raises(ValueError, match="members differ"):
        _materialize(
            archive,
            output,
            expected_member_paths=("different.txt",),
        )
    assert not output.exists()


def test_audit_asep_inner_archives_proves_equal_label_free_axes(
    tmp_path: Path,
) -> None:
    from interaction_first.asep_authority import audit_asep_inner_archives

    authority = tmp_path / "authority"
    receipt = _inner_authority(authority)
    output = tmp_path / "inner-audit.json"
    result = audit_asep_inner_archives(
        authority_root=authority,
        authority_receipt_sha256=_digests(receipt)[0],
        output_path=output,
    )
    assert result["instance_count"] == 2
    assert result["graph_archive"]["file_count"] == 2
    assert result["structure_archive"]["file_count"] == 2
    assert result["checks"]["split_payload_opened"] is False
    with pytest.raises(FileExistsError):
        audit_asep_inner_archives(
            authority_root=authority,
            authority_receipt_sha256=_digests(receipt)[0],
            output_path=output,
        )


def test_audit_asep_inner_archives_rejects_axis_mismatch(tmp_path: Path) -> None:
    from interaction_first.asep_authority import audit_asep_inner_archives

    authority = tmp_path / "authority"
    receipt = _inner_authority(
        authority,
        structure_ids=("1abc_0P", "3ghi_0P"),
    )
    with pytest.raises(ValueError, match="axes differ"):
        audit_asep_inner_archives(
            authority_root=authority,
            authority_receipt_sha256=_digests(receipt)[0],
            output_path=tmp_path / "audit.json",
        )


def test_audit_asep_inner_archives_rejects_links(tmp_path: Path) -> None:
    from interaction_first.asep_authority import audit_asep_inner_archives

    authority = tmp_path / "authority"
    receipt = _inner_authority(authority, graph_symlink=True)
    with pytest.raises(ValueError, match="non-file member"):
        audit_asep_inner_archives(
            authority_root=authority,
            authority_receipt_sha256=_digests(receipt)[0],
            output_path=tmp_path / "audit.json",
        )


def _write_split_dictionary(
    path: Path,
    *,
    train: tuple[int, ...] = (0, 1),
    val: tuple[int, ...] = (2, 3),
    test: tuple[int, ...] = (4, 5),
) -> None:
    torch = pytest.importorskip("torch")
    method = {
        "train": torch.tensor(train, dtype=torch.int64),
        "val": torch.tensor(val, dtype=torch.int64),
        "test": torch.tensor(test, dtype=torch.int64),
    }
    torch.save(
        {
            "epitope_group": method,
            "epitope_ratio": {
                role: values.clone() for role, values in method.items()
            },
        },
        path,
    )


def test_materialize_asep_split_manifests_binds_indices_atomically(
    tmp_path: Path,
) -> None:
    from interaction_first.asep_authority import (
        audit_asep_inner_archives,
        materialize_asep_split_manifests,
    )

    identifiers = tuple(f"{index + 1}abc_{index}P" for index in range(6))
    authority = tmp_path / "authority"
    receipt_path = _inner_authority(authority, identifiers=identifiers)
    _write_split_dictionary(authority / "payload/asepv1.2.0/split_dict.pt")
    inner_audit_path = tmp_path / "inner-audit.json"
    audit_asep_inner_archives(
        authority_root=authority,
        authority_receipt_sha256=_digests(receipt_path)[0],
        output_path=inner_audit_path,
    )
    output = tmp_path / "split-authority"
    result = materialize_asep_split_manifests(
        authority_root=authority,
        inner_audit_path=inner_audit_path,
        inner_audit_sha256=_digests(inner_audit_path)[0],
        output_root=output,
    )

    with (output / "epitope_group.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["instance_id"] for row in rows] == list(identifiers)
    assert [row["role"] for row in rows] == [
        "train",
        "train",
        "val",
        "val",
        "test",
        "test",
    ]
    assert result["manifests"]["epitope_group"]["role_counts"] == {
        "train": 2,
        "val": 2,
        "test": 2,
    }
    assert result["authority"]["torch_load_weights_only"] is True
    assert result["authority"]["task_labels_read"] is False
    with pytest.raises(FileExistsError):
        materialize_asep_split_manifests(
            authority_root=authority,
            inner_audit_path=inner_audit_path,
            inner_audit_sha256=_digests(inner_audit_path)[0],
            output_root=output,
        )


@pytest.mark.parametrize(
    ("train", "val", "test", "message"),
    [
        ((0, 1), (1, 2), (3, 4, 5), "overlap or lie outside"),
        ((0, 1), (2, 3), (4, 6), "overlap or lie outside"),
        ((0, 1), (2,), (3, 4), "complete ID axis"),
    ],
)
def test_materialize_asep_split_manifests_rejects_invalid_partitions(
    tmp_path: Path,
    train: tuple[int, ...],
    val: tuple[int, ...],
    test: tuple[int, ...],
    message: str,
) -> None:
    from interaction_first.asep_authority import (
        audit_asep_inner_archives,
        materialize_asep_split_manifests,
    )

    identifiers = tuple(f"{index + 1}abc_{index}P" for index in range(6))
    authority = tmp_path / "authority"
    receipt_path = _inner_authority(authority, identifiers=identifiers)
    _write_split_dictionary(
        authority / "payload/asepv1.2.0/split_dict.pt",
        train=train,
        val=val,
        test=test,
    )
    inner_audit_path = tmp_path / "inner-audit.json"
    audit_asep_inner_archives(
        authority_root=authority,
        authority_receipt_sha256=_digests(receipt_path)[0],
        output_path=inner_audit_path,
    )
    output = tmp_path / "split-authority"
    with pytest.raises(ValueError, match=message):
        materialize_asep_split_manifests(
            authority_root=authority,
            inner_audit_path=inner_audit_path,
            inner_audit_sha256=_digests(inner_audit_path)[0],
            output_root=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("unsafe_name", ["../escape", "/absolute", "a\\evil"])
def test_materialize_asep_authority_rejects_unsafe_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(unsafe_name, b"blocked")
    output = tmp_path / "authority"
    with pytest.raises(ValueError, match="[Uu]nsafe|backslash"):
        _materialize(archive, output)
    assert not output.exists()
    assert not (tmp_path / "escape").exists()


def test_materialize_asep_authority_rejects_symbolic_links(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, b"../outside")
    output = tmp_path / "authority"
    with pytest.raises(ValueError, match="symbolic links"):
        _materialize(archive, output)
    assert not output.exists()


def test_materialize_asep_authority_rejects_duplicate_members(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("same.txt", b"first")
            handle.writestr("same.txt", b"second")
    output = tmp_path / "authority"
    with pytest.raises(ValueError, match="Duplicate"):
        _materialize(archive, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"archive_sha256": "0" * 64}, "SHA-256 differs"),
        ({"archive_md5": "0" * 32}, "MD5 differs"),
        ({"expected_size_bytes": 1}, "size differs"),
        ({"maximum_members": 0}, "member count"),
        ({"maximum_uncompressed_bytes": 1}, "uncompressed size"),
    ],
)
def test_materialize_asep_authority_rejects_wrong_authority_or_limits(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    archive = tmp_path / "asep.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("payload.txt", b"content")
    output = tmp_path / "authority"
    with pytest.raises(ValueError, match=message):
        _materialize(archive, output, **override)
    assert not output.exists()
