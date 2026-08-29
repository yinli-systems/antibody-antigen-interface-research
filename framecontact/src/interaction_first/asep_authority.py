"""Safe, receipt-bearing materialization of the prospective AsEP authority."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
from typing import Any
import zipfile

from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file


FORMAT = "framecontact_asep_external_authority_v1"
LOCK_FORMAT = "framecontact_asep_dataset_lock_v1"
INNER_AUDIT_FORMAT = "framecontact_asep_inner_archive_audit_v1"
SPLIT_MANIFEST_FORMAT = "framecontact_asep_split_manifest_authority_v1"


def load_asep_dataset_lock(
    lock_path: Path,
    lock_sha256: str,
) -> dict[str, Any]:
    """Authenticate the pre-access Zenodo and source-code authority lock."""
    authenticate_file(lock_path, lock_sha256, "AsEP dataset lock")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot parse the AsEP dataset lock") from error
    if payload.get("format_version") != LOCK_FORMAT:
        raise ValueError("Unexpected AsEP dataset lock format")
    dataset = payload.get("dataset", {})
    source = payload.get("official_source", {})
    boundary = payload.get("evidence_boundary_at_freeze", {})
    if (
        dataset.get("name") != "AsEP"
        or dataset.get("version") != "v1.2.0"
        or dataset.get("record_doi") != "10.5281/zenodo.11495514"
        or not isinstance(dataset.get("archive_size_bytes"), int)
        or dataset["archive_size_bytes"] <= 0
        or not isinstance(dataset.get("publisher_md5"), str)
        or len(dataset["publisher_md5"]) != 32
        or not isinstance(dataset.get("archive_sha256"), str)
        or len(dataset["archive_sha256"]) != 64
        or not isinstance(source.get("revision"), str)
        or len(source["revision"]) != 40
    ):
        raise ValueError("AsEP dataset lock authority is incomplete")
    if (
        boundary.get("task_labels_read") is not False
        or boundary.get("model_predictions_or_scores_read") is not False
        or boundary.get("external_model_selection_performed") is not False
    ):
        raise ValueError("AsEP dataset lock is not eligible for pre-access use")
    members = payload.get("expected_archive_members")
    if not isinstance(members, list) or not members:
        raise ValueError("AsEP dataset lock has no expected archive members")
    normalized = [_safe_member_path(str(value)).as_posix() for value in members]
    if len(set(normalized)) != len(normalized):
        raise ValueError("AsEP dataset lock contains duplicate archive members")
    payload["expected_archive_members"] = normalized
    return payload


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name:
        raise ValueError("Zip member name is empty or contains a backslash")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe zip member path: {name!r}")
    return path


def materialize_asep_authority(
    *,
    archive_path: Path,
    archive_sha256: str,
    archive_md5: str,
    expected_size_bytes: int,
    source_code_revision: str,
    output_root: Path,
    maximum_members: int = 100_000,
    maximum_uncompressed_bytes: int = 64 * 1024**3,
    expected_member_paths: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Safely extract a fully authenticated AsEP zip into one atomic authority."""
    authenticated_archive = authenticate_file(
        archive_path, archive_sha256, "AsEP archive"
    )
    if archive_path.stat().st_size != expected_size_bytes:
        raise ValueError("AsEP archive size differs from the frozen authority")
    if _md5_file(archive_path) != archive_md5.lower():
        raise ValueError("AsEP archive MD5 differs from the publisher checksum")
    if len(source_code_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_code_revision
    ):
        raise ValueError("source_code_revision must be a lowercase Git commit")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite AsEP authority: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        payload_root = temporary / "payload"
        payload_root.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members or len(members) > maximum_members:
                raise ValueError("AsEP zip member count is empty or exceeds the limit")
            total_uncompressed = sum(member.file_size for member in members)
            if total_uncompressed <= 0 or total_uncompressed > maximum_uncompressed_bytes:
                raise ValueError("AsEP zip uncompressed size is outside the allowed range")
            inventory = []
            seen: set[PurePosixPath] = set()
            validated_members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            for member in members:
                relative = _safe_member_path(member.filename)
                if relative in seen:
                    raise ValueError(f"Duplicate AsEP zip member: {member.filename!r}")
                seen.add(relative)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("AsEP zip may not contain symbolic links")
                validated_members.append((member, relative))
                inventory.append(
                    {
                        "path": relative.as_posix(),
                        "is_directory": member.is_dir(),
                        "uncompressed_size": member.file_size,
                        "compressed_size": member.compress_size,
                        "crc32": f"{member.CRC:08x}",
                    }
                )
            if expected_member_paths is not None:
                expected = {
                    _safe_member_path(value).as_posix()
                    for value in expected_member_paths
                }
                observed = {relative.as_posix() for _, relative in validated_members}
                if observed != expected:
                    raise ValueError("AsEP zip members differ from the dataset lock")
            for member, relative in validated_members:
                destination = payload_root.joinpath(*relative.parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        inventory.sort(key=lambda value: value["path"])
        inventory_path = temporary / "ZIP_MEMBER_MANIFEST.json"
        inventory_path.write_bytes(canonical_json_bytes(inventory))
        receipt = {
            "format_version": FORMAT,
            "authority": {
                "record_doi": "10.5281/zenodo.11495514",
                "dataset_version": "v1.2.0",
                "archive_sha256": authenticated_archive,
                "publisher_md5": archive_md5.lower(),
                "archive_size_bytes": expected_size_bytes,
                "official_source_code_revision": source_code_revision,
                "task_predictions_read": False,
            },
            "member_count": len(inventory),
            "uncompressed_size_bytes": total_uncompressed,
            "member_manifest_sha256": sha256_file(inventory_path),
            "safe_extraction_checks": {
                "absolute_or_parent_paths_rejected": True,
                "backslash_paths_rejected": True,
                "duplicate_paths_rejected": True,
                "symbolic_links_rejected": True,
                "zip_crc_verified_during_streaming_read": True,
                "member_and_uncompressed_size_limits_enforced": True,
                "dataset_lock_member_set_verified": expected_member_paths is not None,
            },
            "claim_boundary": (
                "Authenticated external data materialization only. No split was opened "
                "for model selection and no external performance claim is authorized."
            ),
        }
        (temporary / "ASEP_AUTHORITY_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _tar_identifier_set(
    archive_path: Path,
    *,
    root_name: str,
    extension: str,
) -> tuple[set[str], dict[str, Any]]:
    identifiers: set[str] = set()
    member_count = 0
    directory_count = 0
    try:
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                member_count += 1
                relative = _safe_member_path(member.name)
                if member.issym() or member.islnk() or not (
                    member.isdir() or member.isfile()
                ):
                    raise ValueError("AsEP inner tar contains a non-file member")
                if member.isdir():
                    directory_count += 1
                    if relative.as_posix() != root_name:
                        raise ValueError("AsEP inner tar contains an unexpected directory")
                    continue
                if len(relative.parts) != 2 or relative.parts[0] != root_name:
                    raise ValueError("AsEP inner tar file lies outside its expected root")
                name = relative.parts[1]
                if not name.endswith(extension):
                    raise ValueError("AsEP inner tar contains an unexpected file type")
                identifier = name[: -len(extension)]
                if not identifier or identifier in identifiers:
                    raise ValueError("AsEP inner tar contains a duplicate identifier")
                identifiers.add(identifier)
    except (OSError, tarfile.TarError) as error:
        raise ValueError("Cannot stream the AsEP inner tar") from error
    if directory_count != 1 or not identifiers:
        raise ValueError("AsEP inner tar has an incomplete member layout")
    return identifiers, {
        "member_count": member_count,
        "directory_count": directory_count,
        "file_count": len(identifiers),
        "sha256": sha256_file(archive_path),
    }


def audit_asep_inner_archives(
    *,
    authority_root: Path,
    authority_receipt_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Audit the label-free ID axes and safe types of both AsEP inner tars."""
    if authority_root.is_symlink() or not authority_root.is_dir():
        raise ValueError("AsEP authority root must be a real directory")
    receipt_path = authority_root / "ASEP_AUTHORITY_RECEIPT.json"
    receipt_sha256 = authenticate_file(
        receipt_path,
        authority_receipt_sha256,
        "AsEP authority receipt",
    )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot parse the AsEP authority receipt") from error
    if (
        receipt.get("format_version") != FORMAT
        or receipt.get("authority", {}).get("task_predictions_read") is not False
        or receipt.get("safe_extraction_checks", {}).get(
            "dataset_lock_member_set_verified"
        )
        is not True
    ):
        raise ValueError("AsEP authority is not eligible for inner audit")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite AsEP inner audit: {output_path}")
    payload = authority_root / "payload/asepv1.2.0"
    identifiers_path = payload / "asepv1-AbDb-IDs.txt"
    split_path = payload / "split_dict.pt"
    graph_path = payload / "asepv1_interim_graphs.tar.gz"
    structure_path = payload / "structures.tar.gz"
    for path, name in (
        (identifiers_path, "identifier list"),
        (split_path, "split dictionary"),
        (graph_path, "graph archive"),
        (structure_path, "structure archive"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"AsEP {name} must be a regular non-symlink file")
    try:
        identifiers = tuple(
            line.strip()
            for line in identifiers_path.read_text(encoding="ascii").splitlines()
        )
    except (OSError, UnicodeError) as error:
        raise ValueError("Cannot read the AsEP identifier list") from error
    if not identifiers or any(
        not value or "/" in value or "\\" in value for value in identifiers
    ):
        raise ValueError("AsEP identifier list contains an unsafe identifier")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("AsEP identifier list contains duplicates")
    expected = set(identifiers)
    graph_identifiers, graph_audit = _tar_identifier_set(
        graph_path,
        root_name="asepv1_interim_graphs",
        extension=".pt",
    )
    structure_identifiers, structure_audit = _tar_identifier_set(
        structure_path,
        root_name="structures",
        extension=".pdb",
    )
    if graph_identifiers != expected or structure_identifiers != expected:
        raise ValueError("AsEP identifier, graph, and structure axes differ")
    payload_receipt = {
        "format_version": INNER_AUDIT_FORMAT,
        "authority": {
            "authority_receipt_sha256": receipt_sha256,
            "identifier_list_sha256": sha256_file(identifiers_path),
            "split_dict_sha256": sha256_file(split_path),
            "task_labels_read": False,
            "model_predictions_or_scores_read": False,
        },
        "instance_count": len(expected),
        "graph_archive": graph_audit,
        "structure_archive": structure_audit,
        "checks": {
            "identifier_sets_equal": True,
            "path_traversal_rejected": True,
            "links_and_special_members_rejected": True,
            "gzip_streams_read_to_end": True,
            "split_payload_opened": False,
        },
        "claim_boundary": (
            "Label-free external axis and archive-safety audit only. The split "
            "payload, graph tensors, contacts, and task labels were not loaded."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(payload_receipt))
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {**payload_receipt, "output_path": str(output_path)}


def materialize_asep_split_manifests(
    *,
    authority_root: Path,
    inner_audit_path: Path,
    inner_audit_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Safely open only official split indices and bind them to AsEP IDs."""
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("PyTorch is required to read the AsEP split index") from error
    authenticate_file(inner_audit_path, inner_audit_sha256, "AsEP inner audit")
    try:
        inner_audit = json.loads(inner_audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot parse the AsEP inner audit") from error
    if (
        inner_audit.get("format_version") != INNER_AUDIT_FORMAT
        or inner_audit.get("authority", {}).get("task_labels_read") is not False
        or inner_audit.get("checks", {}).get("split_payload_opened") is not False
    ):
        raise ValueError("AsEP inner audit is not eligible for split materialization")
    if authority_root.is_symlink() or not authority_root.is_dir():
        raise ValueError("AsEP authority root must be a real directory")
    payload_root = authority_root / "payload/asepv1.2.0"
    identifiers_path = payload_root / "asepv1-AbDb-IDs.txt"
    split_path = payload_root / "split_dict.pt"
    authenticate_file(
        identifiers_path,
        inner_audit["authority"]["identifier_list_sha256"],
        "AsEP identifiers",
    )
    authenticate_file(
        split_path,
        inner_audit["authority"]["split_dict_sha256"],
        "AsEP split dictionary",
    )
    identifiers = tuple(
        line.strip()
        for line in identifiers_path.read_text(encoding="ascii").splitlines()
    )
    if len(identifiers) != int(inner_audit["instance_count"]):
        raise ValueError("AsEP identifier count differs from the inner audit")
    split_payload = torch.load(split_path, map_location="cpu", weights_only=True)
    expected_methods = {"epitope_group", "epitope_ratio"}
    expected_roles = {"train", "val", "test"}
    if not isinstance(split_payload, dict) or set(split_payload) != expected_methods:
        raise ValueError("AsEP split methods differ from the official schema")
    validated: dict[str, dict[str, tuple[int, ...]]] = {}
    role_counts: dict[str, dict[str, int]] = {}
    for method in sorted(expected_methods):
        split = split_payload[method]
        if not isinstance(split, dict) or set(split) != expected_roles:
            raise ValueError("AsEP split roles differ from the official schema")
        by_role: dict[str, tuple[int, ...]] = {}
        seen: set[int] = set()
        for role in ("train", "val", "test"):
            tensor = split[role]
            if (
                not torch.is_tensor(tensor)
                or tensor.ndim != 1
                or tensor.dtype == torch.bool
                or torch.is_floating_point(tensor)
            ):
                raise ValueError("AsEP split indices must be one-dimensional integers")
            values = tuple(int(value) for value in tensor.tolist())
            if (
                not values
                or len(set(values)) != len(values)
                or any(value < 0 or value >= len(identifiers) for value in values)
                or seen.intersection(values)
            ):
                raise ValueError("AsEP split indices overlap or lie outside the ID axis")
            seen.update(values)
            by_role[role] = values
        if seen != set(range(len(identifiers))):
            raise ValueError("AsEP split indices do not cover the complete ID axis")
        validated[method] = by_role
        role_counts[method] = {
            role: len(by_role[role]) for role in ("train", "val", "test")
        }
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite AsEP splits: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        manifests: dict[str, dict[str, Any]] = {}
        for method, by_role in validated.items():
            role_by_index = {
                index: role for role, values in by_role.items() for index in values
            }
            path = temporary / f"{method}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("dataset_index", "instance_id", "role"),
                    lineterminator="\n",
                )
                writer.writeheader()
                for index, identifier in enumerate(identifiers):
                    writer.writerow(
                        {
                            "dataset_index": index,
                            "instance_id": identifier,
                            "role": role_by_index[index],
                        }
                    )
            manifests[method] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "role_counts": role_counts[method],
            }
        receipt = {
            "format_version": SPLIT_MANIFEST_FORMAT,
            "authority": {
                "inner_audit_sha256": inner_audit_sha256,
                "identifier_list_sha256": inner_audit["authority"][
                    "identifier_list_sha256"
                ],
                "split_dict_sha256": inner_audit["authority"]["split_dict_sha256"],
                "torch_load_weights_only": True,
                "task_labels_read": False,
                "model_predictions_or_scores_read": False,
            },
            "instance_count": len(identifiers),
            "manifests": manifests,
            "checks": {
                "roles_pairwise_disjoint": True,
                "complete_index_coverage": True,
                "indices_in_bounds": True,
            },
            "claim_boundary": (
                "Official split membership only. No graph tensor, contact label, "
                "epitope label, prediction, or performance metric was read."
            ),
        }
        (temporary / "ASEP_SPLIT_MANIFEST_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "FORMAT",
    "INNER_AUDIT_FORMAT",
    "LOCK_FORMAT",
    "SPLIT_MANIFEST_FORMAT",
    "audit_asep_inner_archives",
    "load_asep_dataset_lock",
    "materialize_asep_split_manifests",
    "materialize_asep_authority",
]
