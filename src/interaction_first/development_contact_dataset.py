"""Authenticated train-only dataset for PAIRScale contact development."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping

from .contacts import FORMAT_VERSION as CONTACT_FORMAT_VERSION
from .contacts import SparseContactMap, load_contact_map_bytes
from .development_authority import CLAIM_BOUNDARY, PERMISSION_NAME, canonical_sha256
from .development_contact_cache import CACHE_FORMAT, RECEIPT_FORMAT
from .frontier_data import EXAMPLE_ARTIFACT_FORMAT


MANIFEST_FILENAME = "PAIRSCALE_CONTACT_CACHE_MANIFEST.json"
RECEIPT_FILENAME = "CONTACT_CACHE_BUILD_RECEIPT.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONTROL_FILES = frozenset(
    {
        "AUTHORIZED_FOLD_INVENTORY.json",
        "PAIRSCALE_DEVELOPMENT_SUBSET.json",
        "DEVELOPMENT_AUTHORITY_DECISION.json",
        MANIFEST_FILENAME,
        RECEIPT_FILENAME,
    }
)
_MANIFEST_FIELDS = {
    "format_version",
    "permission_name",
    "claim_boundary",
    "source_split",
    "whole_pool_materialized",
    "policy_sha256",
    "decision_sha256",
    "subset_manifest_sha256",
    "authorization_manifest_sha256",
    "parent_cache_manifest_sha256",
    "contact_summary_sha256",
    "antibody_input_dim",
    "antigen_input_dim",
    "instance_count",
    "total_embedding_bytes",
    "total_contact_bytes",
    "candidate_pair_count",
    "positive_pair_count",
    "instances",
}
_INSTANCE_FIELDS = {
    "instance_id",
    "antibody_id",
    "antigen_id",
    "antibody_length",
    "antigen_length",
    "embedding_artifact_path",
    "embedding_artifact_sha256",
    "embedding_artifact_size_bytes",
    "contact_artifact_path",
    "contact_artifact_sha256",
    "contact_artifact_size_bytes",
    "contact_format_version",
    "antibody_candidate_residue_count",
    "antigen_candidate_residue_count",
    "candidate_pair_count",
    "positive_pair_count",
}
_RECEIPT_FIELDS = {
    "format_version",
    "permission_name",
    "claim_boundary",
    "source_split",
    "whole_pool_materialized",
    "manifest_sha256",
    "policy_sha256",
    "decision_sha256",
    "subset_manifest_sha256",
    "authorization_manifest_sha256",
    "parent_cache_manifest_sha256",
    "contact_summary_sha256",
    "instance_count",
    "candidate_pair_count",
    "positive_pair_count",
    "implementation_source_sha256",
    "tensor_artifacts_deserialized_by_materializer",
    "validation_or_test_data_accessed",
    "confirmatory_authority_granted",
    "receipt_sha256",
}
_EMBEDDING_FIELDS = {
    "format_version",
    "instance_id",
    "antibody_id",
    "antigen_id",
    "antibody_embeddings",
    "antigen_embeddings",
    "antibody_chain_ids",
    "contact_target",
    "contact_mask",
}


class DevelopmentContactDatasetError(ValueError):
    """Raised when authenticated development data fails closed."""


@dataclass(frozen=True)
class ContactCacheRecord:
    """Immutable manifest metadata for one development complex."""

    instance_id: str
    antibody_id: str
    antigen_id: str
    antibody_length: int
    antigen_length: int
    embedding_path: PurePosixPath
    embedding_sha256: str
    embedding_size_bytes: int
    contact_path: PurePosixPath
    contact_sha256: str
    contact_size_bytes: int
    antibody_candidate_count: int
    antigen_candidate_count: int
    candidate_pair_count: int
    positive_pair_count: int


@dataclass(frozen=True)
class DevelopmentContactExample:
    """One authenticated complex in antibody-by-antigen orientation."""

    instance_id: str
    antibody_id: str
    antigen_id: str
    antibody_embeddings: Any
    antigen_embeddings: Any
    contact_targets: Any
    contact_mask: Any
    positive_pair_count: int


def _require_torch() -> Any:
    """Import PyTorch only when authenticated tensors are deserialized."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for PAIRScale training") from error
    return torch


def _reject_constant(value: str) -> None:
    raise DevelopmentContactDatasetError(f"Non-finite JSON constant: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DevelopmentContactDatasetError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentContactDatasetError(f"Cannot parse {name}") from error
    if type(value) is not dict:
        raise DevelopmentContactDatasetError(f"{name} must be a JSON object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise DevelopmentContactDatasetError(
            f"{name} fields differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise DevelopmentContactDatasetError(f"{name} must be a SHA-256 digest")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DevelopmentContactDatasetError(f"{name} must be a positive integer")
    return int(value)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DevelopmentContactDatasetError(f"{name} is not canonical")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise DevelopmentContactDatasetError(f"{name} contains a path character")
    return value


def _relative_path(value: Any, expected: str, name: str) -> PurePosixPath:
    if not isinstance(value, str) or value != expected:
        raise DevelopmentContactDatasetError(f"{name} is not canonical")
    result = PurePosixPath(value)
    if result.is_absolute() or ".." in result.parts:
        raise DevelopmentContactDatasetError(f"{name} escapes the cache root")
    return result


def _read_regular_bytes(path: Path, name: str) -> bytes:
    """Read a non-symlink regular file and detect concurrent mutation."""
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DevelopmentContactDatasetError(f"Cannot open {name}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DevelopmentContactDatasetError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise DevelopmentContactDatasetError(f"{name} ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise DevelopmentContactDatasetError(f"{name} grew during reading")
        after = os.fstat(descriptor)
        fingerprint = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after):
            raise DevelopmentContactDatasetError(f"{name} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _safe_torch_load(raw: bytes) -> Any:
    torch = _require_torch()
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("PyTorch with weights_only=True is required") from error


def _cpu_dense_contiguous(value: Any) -> bool:
    torch = _require_torch()
    return bool(
        torch.is_tensor(value)
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.is_contiguous()
    )


class DevelopmentContactDataset:
    """Random-access loader bound to an isolated score-free train-only cache."""

    def __init__(
        self,
        cache_root: Path,
        *,
        expected_manifest_sha256: str,
        expected_receipt_file_sha256: str,
        expected_policy_sha256: str,
        expected_decision_sha256: str,
        expected_subset_manifest_sha256: str,
        expected_authorization_manifest_sha256: str,
    ) -> None:
        self.cache_root = Path(cache_root).resolve(strict=True)
        if not self.cache_root.is_dir() or self.cache_root.is_symlink():
            raise DevelopmentContactDatasetError("cache_root must be a real directory")
        external = {
            "manifest": _sha256(expected_manifest_sha256, "manifest SHA-256"),
            "receipt": _sha256(expected_receipt_file_sha256, "receipt SHA-256"),
            "policy": _sha256(expected_policy_sha256, "policy SHA-256"),
            "decision": _sha256(expected_decision_sha256, "decision SHA-256"),
            "subset": _sha256(expected_subset_manifest_sha256, "subset SHA-256"),
            "authorization": _sha256(
                expected_authorization_manifest_sha256,
                "authorization SHA-256",
            ),
        }
        manifest_bytes = _read_regular_bytes(
            self.cache_root / MANIFEST_FILENAME,
            "contact cache manifest",
        )
        receipt_bytes = _read_regular_bytes(
            self.cache_root / RECEIPT_FILENAME,
            "contact cache receipt",
        )
        if hashlib.sha256(manifest_bytes).hexdigest() != external["manifest"]:
            raise DevelopmentContactDatasetError("manifest external anchor mismatch")
        if hashlib.sha256(receipt_bytes).hexdigest() != external["receipt"]:
            raise DevelopmentContactDatasetError("receipt external anchor mismatch")
        manifest = _strict_json(manifest_bytes, "contact cache manifest")
        receipt = _strict_json(receipt_bytes, "contact cache receipt")
        _exact_fields(manifest, _MANIFEST_FIELDS, "contact cache manifest")
        _exact_fields(receipt, _RECEIPT_FIELDS, "contact cache receipt")
        self._validate_top_level(manifest, receipt, external)
        self.antibody_input_dim = _positive_integer(
            manifest["antibody_input_dim"],
            "antibody_input_dim",
        )
        self.antigen_input_dim = _positive_integer(
            manifest["antigen_input_dim"],
            "antigen_input_dim",
        )
        raw_records = manifest["instances"]
        if not isinstance(raw_records, list) or not raw_records:
            raise DevelopmentContactDatasetError("instances must be nonempty")
        records = tuple(
            self._parse_record(value, index)
            for index, value in enumerate(raw_records)
        )
        if tuple(record.instance_id for record in records) != tuple(
            sorted(record.instance_id for record in records)
        ) or len({record.instance_id for record in records}) != len(records):
            raise DevelopmentContactDatasetError("instance IDs must be sorted and unique")
        if len(records) != _positive_integer(manifest["instance_count"], "instance_count"):
            raise DevelopmentContactDatasetError("instance count mismatch")
        self.records = records
        self.manifest_sha256 = external["manifest"]
        self.receipt_file_sha256 = external["receipt"]
        self._validate_aggregates(manifest)
        self._validate_file_inventory()

    @staticmethod
    def _validate_top_level(
        manifest: Mapping[str, Any],
        receipt: Mapping[str, Any],
        external: Mapping[str, str],
    ) -> None:
        locked = {
            "format_version": CACHE_FORMAT,
            "permission_name": PERMISSION_NAME,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_split": "train",
            "whole_pool_materialized": False,
            "policy_sha256": external["policy"],
            "decision_sha256": external["decision"],
            "subset_manifest_sha256": external["subset"],
            "authorization_manifest_sha256": external["authorization"],
        }
        for name, expected in locked.items():
            if manifest.get(name) != expected:
                raise DevelopmentContactDatasetError(f"manifest {name} mismatch")
        receipt_locked = dict(locked)
        receipt_locked["format_version"] = RECEIPT_FORMAT
        for name, expected in receipt_locked.items():
            if receipt.get(name) != expected:
                raise DevelopmentContactDatasetError(f"receipt {name} mismatch")
        if receipt["manifest_sha256"] != external["manifest"]:
            raise DevelopmentContactDatasetError("receipt does not bind the manifest")
        for name in (
            "parent_cache_manifest_sha256",
            "contact_summary_sha256",
            "instance_count",
            "candidate_pair_count",
            "positive_pair_count",
        ):
            if receipt[name] != manifest[name]:
                raise DevelopmentContactDatasetError(f"receipt {name} mismatch")
        if receipt["tensor_artifacts_deserialized_by_materializer"] is not False:
            raise DevelopmentContactDatasetError("materializer deserialized tensors")
        if receipt["validation_or_test_data_accessed"] is not False:
            raise DevelopmentContactDatasetError("validation or test access is forbidden")
        if receipt["confirmatory_authority_granted"] is not False:
            raise DevelopmentContactDatasetError("confirmatory authority is forbidden")
        receipt_payload = dict(receipt)
        declared_receipt_sha256 = _sha256(
            receipt_payload.pop("receipt_sha256"),
            "receipt internal SHA-256",
        )
        if canonical_sha256(receipt_payload) != declared_receipt_sha256:
            raise DevelopmentContactDatasetError("receipt internal digest mismatch")

    @staticmethod
    def _parse_record(value: Any, index: int) -> ContactCacheRecord:
        if type(value) is not dict:
            raise DevelopmentContactDatasetError(f"instances[{index}] is not an object")
        _exact_fields(value, _INSTANCE_FIELDS, f"instances[{index}]")
        instance_id = _identifier(value["instance_id"], "instance_id")
        if value["contact_format_version"] != CONTACT_FORMAT_VERSION:
            raise DevelopmentContactDatasetError("unsupported contact format")
        return ContactCacheRecord(
            instance_id=instance_id,
            antibody_id=_identifier(value["antibody_id"], "antibody_id"),
            antigen_id=_identifier(value["antigen_id"], "antigen_id"),
            antibody_length=_positive_integer(value["antibody_length"], "antibody_length"),
            antigen_length=_positive_integer(value["antigen_length"], "antigen_length"),
            embedding_path=_relative_path(
                value["embedding_artifact_path"],
                f"embeddings/{instance_id}.pt",
                "embedding_artifact_path",
            ),
            embedding_sha256=_sha256(value["embedding_artifact_sha256"], "embedding SHA-256"),
            embedding_size_bytes=_positive_integer(value["embedding_artifact_size_bytes"], "embedding size"),
            contact_path=_relative_path(
                value["contact_artifact_path"],
                f"contacts/{instance_id}.npz",
                "contact_artifact_path",
            ),
            contact_sha256=_sha256(value["contact_artifact_sha256"], "contact SHA-256"),
            contact_size_bytes=_positive_integer(value["contact_artifact_size_bytes"], "contact size"),
            antibody_candidate_count=_positive_integer(value["antibody_candidate_residue_count"], "antibody candidate count"),
            antigen_candidate_count=_positive_integer(value["antigen_candidate_residue_count"], "antigen candidate count"),
            candidate_pair_count=_positive_integer(value["candidate_pair_count"], "candidate pair count"),
            positive_pair_count=_positive_integer(value["positive_pair_count"], "positive pair count"),
        )

    def _validate_aggregates(self, manifest: Mapping[str, Any]) -> None:
        expected = {
            "total_embedding_bytes": sum(value.embedding_size_bytes for value in self.records),
            "total_contact_bytes": sum(value.contact_size_bytes for value in self.records),
            "candidate_pair_count": sum(value.candidate_pair_count for value in self.records),
            "positive_pair_count": sum(value.positive_pair_count for value in self.records),
        }
        for name, value in expected.items():
            if manifest[name] != value:
                raise DevelopmentContactDatasetError(f"aggregate {name} mismatch")

    def _validate_file_inventory(self) -> None:
        expected = set(_CONTROL_FILES)
        expected.update(str(value.embedding_path) for value in self.records)
        expected.update(str(value.contact_path) for value in self.records)
        observed: set[str] = set()
        for path in self.cache_root.rglob("*"):
            if path.is_symlink():
                raise DevelopmentContactDatasetError("cache cannot contain symlinks")
            if path.is_file():
                observed.add(path.relative_to(self.cache_root).as_posix())
            elif path.is_dir() and path.relative_to(self.cache_root).as_posix() not in {
                "embeddings",
                "contacts",
            }:
                raise DevelopmentContactDatasetError("cache contains an unexpected directory")
        if observed != expected:
            raise DevelopmentContactDatasetError(
                f"cache inventory differs; missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def identity(self, index: int) -> ContactCacheRecord:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        return self.records[index]

    def __getitem__(self, index: int) -> DevelopmentContactExample:
        record = self.identity(index)
        embedding_bytes = _read_regular_bytes(
            self.cache_root.joinpath(*record.embedding_path.parts),
            f"embedding {record.instance_id}",
        )
        contact_bytes = _read_regular_bytes(
            self.cache_root.joinpath(*record.contact_path.parts),
            f"contact {record.instance_id}",
        )
        for raw, expected_size, expected_sha256, name in (
            (embedding_bytes, record.embedding_size_bytes, record.embedding_sha256, "embedding"),
            (contact_bytes, record.contact_size_bytes, record.contact_sha256, "contact"),
        ):
            if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
                raise DevelopmentContactDatasetError(f"{name} artifact anchor mismatch")
        payload = _safe_torch_load(embedding_bytes)
        if type(payload) is not dict:
            raise DevelopmentContactDatasetError("embedding artifact must be a mapping")
        _exact_fields(payload, _EMBEDDING_FIELDS, "embedding artifact")
        if payload["format_version"] != EXAMPLE_ARTIFACT_FORMAT:
            raise DevelopmentContactDatasetError("unsupported embedding format")
        for name in ("instance_id", "antibody_id", "antigen_id"):
            if payload[name] != getattr(record, name):
                raise DevelopmentContactDatasetError(f"embedding {name} mismatch")
        if payload["contact_target"] is not None or payload["contact_mask"] is not None:
            raise DevelopmentContactDatasetError("unbound embedded contact labels are forbidden")
        torch = _require_torch()
        antibody = payload["antibody_embeddings"]
        antigen = payload["antigen_embeddings"]
        chains = payload["antibody_chain_ids"]
        if not all(_cpu_dense_contiguous(value) for value in (antibody, antigen, chains)):
            raise DevelopmentContactDatasetError("embedding tensors must be contiguous CPU tensors")
        if antibody.ndim != 2 or tuple(antibody.shape) != (
            record.antibody_length,
            self.antibody_input_dim,
        ):
            raise DevelopmentContactDatasetError("antibody embedding shape mismatch")
        if antigen.ndim != 2 or tuple(antigen.shape) != (
            record.antigen_length,
            self.antigen_input_dim,
        ):
            raise DevelopmentContactDatasetError("antigen embedding shape mismatch")
        if chains.ndim != 1 or int(chains.shape[0]) != record.antibody_length:
            raise DevelopmentContactDatasetError("antibody chain tensor shape mismatch")
        if not torch.is_floating_point(antibody) or not torch.is_floating_point(antigen):
            raise DevelopmentContactDatasetError("embeddings must be floating point")
        if not bool(torch.isfinite(antibody).all()) or not bool(torch.isfinite(antigen).all()):
            raise DevelopmentContactDatasetError("embeddings must be finite")
        if antibody.requires_grad or antigen.requires_grad:
            raise DevelopmentContactDatasetError("input embeddings cannot require gradients")
        contact = load_contact_map_bytes(
            contact_bytes,
            expected_instance_id=record.instance_id,
            source_name=f"authenticated contact {record.instance_id}",
        )
        self._validate_contact(contact, record)
        labels, mask = contact.dense()
        antibody_indices = torch.from_numpy(contact.antibody_indices.copy()).long()
        antigen_indices = torch.from_numpy(contact.antigen_indices.copy()).long()
        return DevelopmentContactExample(
            instance_id=record.instance_id,
            antibody_id=record.antibody_id,
            antigen_id=record.antigen_id,
            antibody_embeddings=antibody.index_select(0, antibody_indices).float().contiguous(),
            antigen_embeddings=antigen.index_select(0, antigen_indices).float().contiguous(),
            contact_targets=torch.from_numpy(labels.T.copy()).float(),
            contact_mask=torch.from_numpy(mask.T.copy()).bool(),
            positive_pair_count=record.positive_pair_count,
        )

    @staticmethod
    def _validate_contact(contact: SparseContactMap, record: ContactCacheRecord) -> None:
        observed = (
            contact.antibody_length,
            contact.antigen_length,
            len(contact.antibody_indices),
            len(contact.antigen_indices),
            contact.candidate_count,
            contact.positive_count,
        )
        expected = (
            record.antibody_length,
            record.antigen_length,
            record.antibody_candidate_count,
            record.antigen_candidate_count,
            record.candidate_pair_count,
            record.positive_pair_count,
        )
        if observed != expected:
            raise DevelopmentContactDatasetError("contact metadata mismatch")
