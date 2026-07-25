"""Strict fold-local data access for real-data frontier training.

This module intentionally does not accept a canonical benchmark manifest or a
shared embedding store. A caller must provide an independently authorized
inventory and an isolated cache whose complete file inventory is bound to that
authorization. The cache is rejected before tensor deserialization when those
inventories differ.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .frontier_batching import FrontierExample, canonical_identifier
from .frontier_checkpoint import sha256_file


AUTHORIZATION_FILENAME = "AUTHORIZED_FOLD_INVENTORY.json"
CACHE_MANIFEST_FILENAME = "FOLD_LOCAL_CACHE_MANIFEST.json"
AUTHORIZATION_FORMAT = "frontier_authorized_fold_inventory_v1"
CACHE_MANIFEST_FORMAT = "frontier_fold_local_cache_manifest_v1"
EXAMPLE_ARTIFACT_FORMAT = "frontier_fold_local_example_v1"
TRAIN_ONLY_SCOPE = "train_only_fold_local"
TRAIN_ROLE = "train"
RESERVED_PADDING_PREFIX = "__frontier_padding__"
COLLECTIVE_VERIFICATION_FORMAT = "frontier_collective_cache_verification_v2"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PATH_TOKENS = {
    "all_data",
    "all_embeddings",
    "canonical_manifest",
    "global_cache",
    "whole_pool",
}


def _require_torch() -> Any:
    """Import PyTorch only when tensor artifacts are requested."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for frontier real-data loading") from error
    return torch


def _reject_constant(value: str) -> None:
    """Reject non-finite JSON constants."""
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _strict_json_object(path: Path) -> dict[str, Any]:
    """Read one finite JSON object with duplicate-key detection."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink JSON file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read strict JSON from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON metadata deterministically."""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def collective_verification_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Return the deterministic digest of one collective verification receipt."""
    if not isinstance(receipt, Mapping):
        raise ValueError("collective verification receipt must be a mapping")
    try:
        payload = _canonical_json_bytes(dict(receipt))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "collective verification receipt must be finite JSON metadata"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _observed_distributed_context() -> tuple[int, int]:
    """Return the initialized default-group rank and world size, or one process."""
    torch = _require_torch()
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return 0, 1
    return int(distributed.get_rank()), int(distributed.get_world_size())


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    """Require an exact object schema."""
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(
            f"{name} fields do not match the locked schema; "
            f"missing={missing}, extra={extra}"
        )


def _positive_integer(value: Any, name: str) -> int:
    """Return a strictly positive non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    """Return a non-negative non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _sha256(value: Any, name: str) -> str:
    """Validate one lowercase SHA-256 digest."""
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_pair_list(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    """Validate a sorted unique antibody-antigen edge list."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two identifiers")
        pairs.append(
            (
                canonical_identifier(pair[0], f"{name}[{index}].antibody_id"),
                canonical_identifier(pair[1], f"{name}[{index}].antigen_id"),
            )
        )
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{name} cannot contain duplicate edges")
    if pairs != sorted(pairs):
        raise ValueError(f"{name} must use deterministic lexicographic order")
    return tuple(pairs)


def _validate_scope_fields(value: Mapping[str, Any], name: str) -> tuple[str, str]:
    """Validate the train-only scope shared by both manifests."""
    if value["data_access_scope"] != TRAIN_ONLY_SCOPE:
        raise ValueError(f"{name} must declare data_access_scope={TRAIN_ONLY_SCOPE!r}")
    if value["split_role"] != TRAIN_ROLE:
        raise ValueError(f"{name} must declare split_role={TRAIN_ROLE!r}")
    fold_id = canonical_identifier(value["fold_id"], f"{name}.fold_id")
    return TRAIN_ONLY_SCOPE, fold_id


def _validate_real_identifier(value: Any, name: str) -> str:
    """Validate an identifier and reserve the synthetic padding namespace."""
    result = canonical_identifier(value, name)
    if result.startswith(RESERVED_PADDING_PREFIX):
        raise ValueError(f"{name} uses the reserved padding namespace")
    return result


def _relative_artifact_path(value: Any, name: str) -> PurePosixPath:
    """Validate a two-component path inside the isolated examples directory."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != "examples"
        or path.suffix != ".pt"
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise ValueError(
            f"{name} must have the exact form examples/<artifact>.pt"
        )
    folded_stem = path.stem.casefold()
    if any(token in folded_stem for token in _FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"{name} contains a forbidden whole-pool path token")
    return path


@dataclass(frozen=True)
class AuthorizedInstance:
    """One authorized matched instance identity."""

    instance_id: str
    antibody_id: str
    antigen_id: str


@dataclass(frozen=True)
class CacheArtifactRecord(AuthorizedInstance):
    """One authorization-bound tensor artifact."""

    artifact_path: PurePosixPath
    artifact_sha256: str
    artifact_size_bytes: int
    antibody_length: int
    antigen_length: int
    contact_supervision: bool


@dataclass(frozen=True)
class FoldLocalCacheContract:
    """The fully verified metadata contract for one train-only cache."""

    cache_root: Path
    fold_id: str
    antibody_input_dim: int
    antigen_input_dim: int
    records: tuple[CacheArtifactRecord, ...]
    positive_pairs: tuple[tuple[str, str], ...]
    excluded_pairs: tuple[tuple[str, str], ...]
    manifest_sha256: str
    authorization_manifest_sha256: str
    cache_external_anchor_verified: bool
    authorization_external_anchor_verified: bool

    @property
    def instance_count(self) -> int:
        """Return the exact number of authorized source instances."""
        return len(self.records)


def _parse_authorized_instances(value: Any) -> tuple[AuthorizedInstance, ...]:
    """Parse a deterministic exact instance inventory."""
    if not isinstance(value, list) or not value:
        raise ValueError("authorization.instances must be a nonempty JSON array")
    records: list[AuthorizedInstance] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"authorization.instances[{index}] must be an object")
        _exact_fields(
            raw,
            {"instance_id", "antibody_id", "antigen_id"},
            f"authorization.instances[{index}]",
        )
        records.append(
            AuthorizedInstance(
                instance_id=_validate_real_identifier(
                    raw["instance_id"],
                    f"authorization.instances[{index}].instance_id",
                ),
                antibody_id=_validate_real_identifier(
                    raw["antibody_id"],
                    f"authorization.instances[{index}].antibody_id",
                ),
                antigen_id=_validate_real_identifier(
                    raw["antigen_id"],
                    f"authorization.instances[{index}].antigen_id",
                ),
            )
        )
    instance_ids = [record.instance_id for record in records]
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("authorization.instances contains duplicate instance_id values")
    if instance_ids != sorted(instance_ids):
        raise ValueError("authorization.instances must be sorted by instance_id")
    return tuple(records)


def _parse_cache_records(value: Any) -> tuple[CacheArtifactRecord, ...]:
    """Parse the cache records without opening any tensor artifact."""
    if not isinstance(value, list) or not value:
        raise ValueError("cache.instances must be a nonempty JSON array")
    records: list[CacheArtifactRecord] = []
    expected_fields = {
        "instance_id",
        "antibody_id",
        "antigen_id",
        "artifact_path",
        "artifact_sha256",
        "artifact_size_bytes",
        "antibody_length",
        "antigen_length",
        "contact_supervision",
    }
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"cache.instances[{index}] must be an object")
        _exact_fields(raw, expected_fields, f"cache.instances[{index}]")
        if not isinstance(raw["contact_supervision"], bool):
            raise ValueError(
                f"cache.instances[{index}].contact_supervision must be boolean"
            )
        records.append(
            CacheArtifactRecord(
                instance_id=_validate_real_identifier(
                    raw["instance_id"],
                    f"cache.instances[{index}].instance_id",
                ),
                antibody_id=_validate_real_identifier(
                    raw["antibody_id"],
                    f"cache.instances[{index}].antibody_id",
                ),
                antigen_id=_validate_real_identifier(
                    raw["antigen_id"],
                    f"cache.instances[{index}].antigen_id",
                ),
                artifact_path=_relative_artifact_path(
                    raw["artifact_path"],
                    f"cache.instances[{index}].artifact_path",
                ),
                artifact_sha256=_sha256(
                    raw["artifact_sha256"],
                    f"cache.instances[{index}].artifact_sha256",
                ),
                artifact_size_bytes=_positive_integer(
                    raw["artifact_size_bytes"],
                    f"cache.instances[{index}].artifact_size_bytes",
                ),
                antibody_length=_positive_integer(
                    raw["antibody_length"],
                    f"cache.instances[{index}].antibody_length",
                ),
                antigen_length=_positive_integer(
                    raw["antigen_length"],
                    f"cache.instances[{index}].antigen_length",
                ),
                contact_supervision=raw["contact_supervision"],
            )
        )
    instance_ids = [record.instance_id for record in records]
    artifact_paths = [record.artifact_path for record in records]
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("cache.instances contains duplicate instance_id values")
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("cache.instances contains duplicate artifact paths")
    if instance_ids != sorted(instance_ids):
        raise ValueError("cache.instances must be sorted by instance_id")
    return tuple(records)


def _validate_graph_inventory(
    records: Sequence[AuthorizedInstance],
    positive_pairs: Sequence[tuple[str, str]],
    excluded_pairs: Sequence[tuple[str, str]],
    name: str,
) -> None:
    """Require all graph edges and matched rows to stay within the inventory."""
    antibody_ids = {record.antibody_id for record in records}
    antigen_ids = {record.antigen_id for record in records}
    graph_edges = set(positive_pairs)
    if graph_edges.intersection(excluded_pairs):
        raise ValueError(f"{name} positive_pairs and excluded_pairs must be disjoint")
    for graph_name, graph in (
        ("positive_pairs", positive_pairs),
        ("excluded_pairs", excluded_pairs),
    ):
        outside = [
            pair
            for pair in graph
            if pair[0] not in antibody_ids or pair[1] not in antigen_ids
        ]
        if outside:
            raise ValueError(f"{name}.{graph_name} references an entity outside the fold")
    missing = sorted(
        {
            (record.antibody_id, record.antigen_id)
            for record in records
        }.difference(graph_edges)
    )
    if missing:
        raise ValueError(
            f"{name}.positive_pairs omits matched instance edges: {missing[:5]}"
        )


def _validate_exact_cache_tree(
    cache_root: Path,
    records: Sequence[CacheArtifactRecord],
) -> None:
    """Prove that the mounted cache contains no unlisted file or directory."""
    expected_files = {
        PurePosixPath(CACHE_MANIFEST_FILENAME),
        *(record.artifact_path for record in records),
    }
    expected_directories: set[PurePosixPath] = set()
    for file_path in expected_files:
        parent = file_path.parent
        while str(parent) != ".":
            expected_directories.add(parent)
            parent = parent.parent

    actual_files: set[PurePosixPath] = set()
    actual_directories: set[PurePosixPath] = set()
    for path in cache_root.rglob("*"):
        relative = PurePosixPath(path.relative_to(cache_root).as_posix())
        if path.is_symlink():
            raise ValueError(f"Fold-local cache cannot contain symlinks: {relative}")
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError(f"Fold-local cache contains a special file: {relative}")
    if actual_files != expected_files:
        raise ValueError(
            "Fold-local cache file inventory differs from its manifest; "
            f"missing={sorted(map(str, expected_files - actual_files))}, "
            f"extra={sorted(map(str, actual_files - expected_files))}"
        )
    if actual_directories != expected_directories:
        raise ValueError(
            "Fold-local cache directory inventory differs from its manifest; "
            f"missing={sorted(map(str, expected_directories - actual_directories))}, "
            f"extra={sorted(map(str, actual_directories - expected_directories))}"
        )


def load_fold_local_cache_contract(
    cache_root: Path,
    authorization_manifest: Path,
    *,
    verify_artifact_hashes: bool = True,
    expected_authorization_manifest_sha256: str | None = None,
    expected_cache_manifest_sha256: str | None = None,
) -> FoldLocalCacheContract:
    """Verify authorization, manifest equality, tree isolation, and artifact hashes."""
    if not isinstance(verify_artifact_hashes, bool):
        raise TypeError("verify_artifact_hashes must be boolean")
    cache_root_argument = Path(cache_root).expanduser()
    authorization_argument = Path(authorization_manifest).expanduser()
    if cache_root_argument.is_symlink():
        raise ValueError("cache_root cannot be a symlink")
    cache_root = cache_root_argument.resolve()
    authorization_path = authorization_argument.resolve()
    if not cache_root.is_dir():
        raise ValueError(f"cache_root must be a directory: {cache_root}")
    if authorization_argument.name != AUTHORIZATION_FILENAME:
        raise ValueError(
            f"authorization manifest must be named {AUTHORIZATION_FILENAME}"
        )
    if authorization_argument.is_symlink() or not authorization_path.is_file():
        raise ValueError("authorization manifest must be a regular non-symlink file")
    try:
        authorization_path.relative_to(cache_root)
    except ValueError:
        pass
    else:
        raise ValueError("authorization manifest must remain outside the cache root")

    cache_manifest_path = cache_root / CACHE_MANIFEST_FILENAME
    authorization_sha256 = sha256_file(authorization_path)
    cache_manifest_sha256 = sha256_file(cache_manifest_path)
    if expected_authorization_manifest_sha256 is not None and (
        _sha256(
            expected_authorization_manifest_sha256,
            "expected_authorization_manifest_sha256",
        )
        != authorization_sha256
    ):
        raise ValueError("Authorization manifest differs from its external anchor")
    if expected_cache_manifest_sha256 is not None and (
        _sha256(
            expected_cache_manifest_sha256,
            "expected_cache_manifest_sha256",
        )
        != cache_manifest_sha256
    ):
        raise ValueError("Cache manifest differs from its external anchor")
    authorization = _strict_json_object(authorization_path)
    cache_manifest = _strict_json_object(cache_manifest_path)
    _exact_fields(
        authorization,
        {
            "format_version",
            "data_access_scope",
            "split_role",
            "fold_id",
            "instances",
            "positive_pairs",
            "excluded_pairs",
        },
        "authorization manifest",
    )
    _exact_fields(
        cache_manifest,
        {
            "format_version",
            "data_access_scope",
            "split_role",
            "fold_id",
            "whole_pool_materialized",
            "authorization_manifest_sha256",
            "antibody_input_dim",
            "antigen_input_dim",
            "instances",
            "positive_pairs",
            "excluded_pairs",
        },
        "cache manifest",
    )
    if authorization["format_version"] != AUTHORIZATION_FORMAT:
        raise ValueError("Unsupported authorization manifest format")
    if cache_manifest["format_version"] != CACHE_MANIFEST_FORMAT:
        raise ValueError("Unsupported fold-local cache manifest format")
    _, authorized_fold = _validate_scope_fields(authorization, "authorization")
    _, cached_fold = _validate_scope_fields(cache_manifest, "cache")
    if authorized_fold != cached_fold:
        raise ValueError("Authorization and cache fold_id values differ")
    if cache_manifest["whole_pool_materialized"] is not False:
        raise ValueError("Cache must prove whole_pool_materialized=false")

    if (
        _sha256(
            cache_manifest["authorization_manifest_sha256"],
            "cache.authorization_manifest_sha256",
        )
        != authorization_sha256
    ):
        raise ValueError("Cache is bound to a different authorization manifest")

    authorized_records = _parse_authorized_instances(authorization["instances"])
    cache_records = _parse_cache_records(cache_manifest["instances"])
    cached_identities = tuple(
        AuthorizedInstance(
            instance_id=record.instance_id,
            antibody_id=record.antibody_id,
            antigen_id=record.antigen_id,
        )
        for record in cache_records
    )
    if cached_identities != authorized_records:
        raise ValueError("Cache instance inventory is not exactly authorization-equal")

    authorized_positive = _canonical_pair_list(
        authorization["positive_pairs"],
        "authorization.positive_pairs",
    )
    authorized_excluded = _canonical_pair_list(
        authorization["excluded_pairs"],
        "authorization.excluded_pairs",
    )
    cached_positive = _canonical_pair_list(
        cache_manifest["positive_pairs"],
        "cache.positive_pairs",
    )
    cached_excluded = _canonical_pair_list(
        cache_manifest["excluded_pairs"],
        "cache.excluded_pairs",
    )
    if cached_positive != authorized_positive or cached_excluded != authorized_excluded:
        raise ValueError("Cache graph inventory is not exactly authorization-equal")
    _validate_graph_inventory(
        authorized_records,
        authorized_positive,
        authorized_excluded,
        "authorization",
    )
    _validate_graph_inventory(
        cache_records,
        cached_positive,
        cached_excluded,
        "cache",
    )

    _validate_exact_cache_tree(cache_root, cache_records)
    for record in cache_records:
        artifact = cache_root.joinpath(*record.artifact_path.parts)
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"Missing regular cache artifact: {record.artifact_path}")
        if artifact.stat().st_size != record.artifact_size_bytes:
            raise ValueError(
                f"Artifact size mismatch for {record.artifact_path}"
            )
        if verify_artifact_hashes and sha256_file(artifact) != record.artifact_sha256:
            raise ValueError(
                f"Artifact SHA-256 mismatch for {record.artifact_path}"
            )

    return FoldLocalCacheContract(
        cache_root=cache_root,
        fold_id=authorized_fold,
        antibody_input_dim=_positive_integer(
            cache_manifest["antibody_input_dim"],
            "cache.antibody_input_dim",
        ),
        antigen_input_dim=_positive_integer(
            cache_manifest["antigen_input_dim"],
            "cache.antigen_input_dim",
        ),
        records=cache_records,
        positive_pairs=cached_positive,
        excluded_pairs=cached_excluded,
        manifest_sha256=cache_manifest_sha256,
        authorization_manifest_sha256=authorization_sha256,
        cache_external_anchor_verified=expected_cache_manifest_sha256 is not None,
        authorization_external_anchor_verified=(
            expected_authorization_manifest_sha256 is not None
        ),
    )


def _safe_torch_load(path: Path) -> Any:
    """Load a trusted tensor-only artifact with restricted unpickling."""
    torch = _require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "Fold-local loading requires a PyTorch release with weights_only=True"
        ) from error


def _tensor_is_cpu_dense_contiguous(value: Any) -> bool:
    """Return whether a value is a dense contiguous CPU tensor."""
    torch = _require_torch()
    return bool(
        torch.is_tensor(value)
        and value.layout == torch.strided
        and value.device.type == "cpu"
        and value.is_contiguous()
    )


class FoldLocalFrontierDataset:
    """Random-access dataset backed only by a verified fold-local cache."""

    def __init__(
        self,
        cache_root: Path,
        authorization_manifest: Path,
        *,
        verify_artifact_hashes: bool = True,
        memory_cache_size: int = 0,
        expected_authorization_manifest_sha256: str | None = None,
        expected_cache_manifest_sha256: str | None = None,
    ) -> None:
        if not isinstance(verify_artifact_hashes, bool):
            raise TypeError("verify_artifact_hashes must be boolean")
        if (
            isinstance(memory_cache_size, bool)
            or not isinstance(memory_cache_size, int)
            or memory_cache_size < 0
        ):
            raise ValueError("memory_cache_size must be a non-negative integer")
        self.contract = load_fold_local_cache_contract(
            cache_root,
            authorization_manifest,
            verify_artifact_hashes=False,
            expected_authorization_manifest_sha256=(
                expected_authorization_manifest_sha256
            ),
            expected_cache_manifest_sha256=expected_cache_manifest_sha256,
        )
        self.memory_cache_size = int(memory_cache_size)
        self._memory_cache: OrderedDict[int, FrontierExample] = OrderedDict()
        self._deserialized_artifact_indices: set[int] = set()
        self._verified_artifact_indices: set[int] = set()
        self._artifact_stat_fingerprints = tuple(
            self._stat_fingerprint(
                self.contract.cache_root.joinpath(*record.artifact_path.parts)
            )
            for record in self.contract.records
        )
        if verify_artifact_hashes:
            for index, record in enumerate(self.contract.records):
                path = self._assert_artifact_unchanged(index)
                if sha256_file(path) != record.artifact_sha256:
                    raise ValueError(
                        f"Artifact SHA-256 mismatch for {record.artifact_path}"
                    )
                self._assert_artifact_unchanged(index)
                self._verified_artifact_indices.add(index)

    @staticmethod
    def _stat_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
        """Capture metadata that changes on replacement or in-place mutation."""
        status = path.stat()
        return (
            int(status.st_dev),
            int(status.st_ino),
            int(status.st_size),
            int(status.st_mtime_ns),
            int(status.st_ctime_ns),
        )

    def __len__(self) -> int:
        """Return the authorization-equal instance count."""
        return self.contract.instance_count

    def identity(self, index: int) -> AuthorizedInstance:
        """Return identity metadata without opening a tensor artifact."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError("dataset index is outside the authorized inventory")
        record = self.contract.records[index]
        return AuthorizedInstance(
            instance_id=record.instance_id,
            antibody_id=record.antibody_id,
            antigen_id=record.antigen_id,
        )

    def _assert_artifact_unchanged(self, index: int) -> Path:
        """Return one artifact only while its verified filesystem identity is stable."""
        record = self.contract.records[index]
        path = self.contract.cache_root.joinpath(*record.artifact_path.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Cache artifact changed type: {record.artifact_path}")
        if self._stat_fingerprint(path) != self._artifact_stat_fingerprints[index]:
            raise ValueError(
                f"Cache artifact changed after verification: {record.artifact_path}"
            )
        if path.stat().st_size != record.artifact_size_bytes:
            raise ValueError(f"Cache artifact changed size: {record.artifact_path}")
        return path

    @staticmethod
    def _clone_example(example: FrontierExample) -> FrontierExample:
        """Return a tensor-independent copy so callers cannot mutate the cache."""
        return FrontierExample(
            instance_id=example.instance_id,
            antibody_id=example.antibody_id,
            antigen_id=example.antigen_id,
            antibody_embeddings=example.antibody_embeddings.clone(),
            antigen_embeddings=example.antigen_embeddings.clone(),
            antibody_chain_ids=example.antibody_chain_ids.clone(),
            contact_target=(
                None
                if example.contact_target is None
                else example.contact_target.clone()
            ),
            contact_mask=(
                None if example.contact_mask is None else example.contact_mask.clone()
            ),
        )

    def _load_example(self, index: int) -> FrontierExample:
        """Load and validate one manifest-bound tensor artifact."""
        torch = _require_torch()
        record = self.contract.records[index]
        path = self._assert_artifact_unchanged(index)
        if index not in self._verified_artifact_indices:
            if sha256_file(path) != record.artifact_sha256:
                raise ValueError(f"Cache artifact changed hash: {record.artifact_path}")
            self._verified_artifact_indices.add(index)
        payload = _safe_torch_load(path)
        self._assert_artifact_unchanged(index)
        if not isinstance(payload, dict):
            raise ValueError("Example artifact must contain a mapping")
        _exact_fields(
            payload,
            {
                "format_version",
                "instance_id",
                "antibody_id",
                "antigen_id",
                "antibody_embeddings",
                "antigen_embeddings",
                "antibody_chain_ids",
                "contact_target",
                "contact_mask",
            },
            f"example artifact {record.artifact_path}",
        )
        if payload["format_version"] != EXAMPLE_ARTIFACT_FORMAT:
            raise ValueError("Unsupported example artifact format")
        for field in ("instance_id", "antibody_id", "antigen_id"):
            if payload[field] != getattr(record, field):
                raise ValueError(
                    f"Example artifact {record.artifact_path} has a mismatched {field}"
                )
        tensor_fields = (
            "antibody_embeddings",
            "antigen_embeddings",
            "antibody_chain_ids",
        )
        if any(
            not _tensor_is_cpu_dense_contiguous(payload[field])
            for field in tensor_fields
        ):
            raise ValueError(
                "Embedding and chain tensors must be dense contiguous CPU tensors"
            )
        if int(payload["antibody_embeddings"].shape[0]) != record.antibody_length:
            raise ValueError("Artifact antibody length differs from its manifest")
        if int(payload["antigen_embeddings"].shape[0]) != record.antigen_length:
            raise ValueError("Artifact antigen length differs from its manifest")
        if (
            payload["antibody_embeddings"].ndim != 2
            or int(payload["antibody_embeddings"].shape[1])
            != self.contract.antibody_input_dim
        ):
            raise ValueError("Artifact antibody embedding dimension is incompatible")
        if (
            payload["antigen_embeddings"].ndim != 2
            or int(payload["antigen_embeddings"].shape[1])
            != self.contract.antigen_input_dim
        ):
            raise ValueError("Artifact antigen embedding dimension is incompatible")
        has_contacts = (
            payload["contact_target"] is not None
            and payload["contact_mask"] is not None
        )
        if (payload["contact_target"] is None) != (payload["contact_mask"] is None):
            raise ValueError("Artifact contact target and mask must be provided together")
        if has_contacts != record.contact_supervision:
            raise ValueError("Artifact contact supervision differs from its manifest")
        if has_contacts and (
            not _tensor_is_cpu_dense_contiguous(payload["contact_target"])
            or not _tensor_is_cpu_dense_contiguous(payload["contact_mask"])
        ):
            raise ValueError("Contact tensors must be dense contiguous CPU tensors")
        example = FrontierExample(
            instance_id=record.instance_id,
            antibody_id=record.antibody_id,
            antigen_id=record.antigen_id,
            antibody_embeddings=payload["antibody_embeddings"],
            antigen_embeddings=payload["antigen_embeddings"],
            antibody_chain_ids=payload["antibody_chain_ids"],
            contact_target=payload["contact_target"],
            contact_mask=payload["contact_mask"],
        )
        if not all(
            not value.requires_grad
            for value in (
                example.antibody_embeddings,
                example.antigen_embeddings,
            )
        ):
            raise ValueError("Frozen input embeddings cannot require gradients")
        self._deserialized_artifact_indices.add(index)
        return example

    def __getitem__(self, index: int) -> FrontierExample:
        """Return one fully validated authorization-bound example."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError("dataset index is outside the authorized inventory")
        cached = self._memory_cache.pop(index, None)
        if cached is not None:
            self._assert_artifact_unchanged(index)
            self._memory_cache[index] = cached
            return self._clone_example(cached)
        example = self._load_example(index)
        if self.memory_cache_size > 0:
            self._memory_cache[index] = example
            while len(self._memory_cache) > self.memory_cache_size:
                self._memory_cache.popitem(last=False)
            return self._clone_example(example)
        return example

    def verify_all_examples(self) -> dict[str, int]:
        """Deserialize every authorized artifact without exposing any score."""
        contact_count = 0
        for index in range(len(self)):
            if self[index].contact_target is not None:
                contact_count += 1
        return {
            "instance_count": len(self),
            "contact_supervised_instance_count": contact_count,
            "antibody_count": len(
                {record.antibody_id for record in self.contract.records}
            ),
            "antigen_count": len(
                {record.antigen_id for record in self.contract.records}
            ),
            "positive_pair_count": len(self.contract.positive_pairs),
            "excluded_pair_count": len(self.contract.excluded_pairs),
        }

    def _artifact_inventory_sha256(self) -> str:
        """Hash the manifest-bound artifact identities used for collective proof."""
        inventory = [
            {
                "artifact_path": record.artifact_path.as_posix(),
                "artifact_sha256": record.artifact_sha256,
                "artifact_size_bytes": record.artifact_size_bytes,
            }
            for record in self.contract.records
        ]
        payload = json.dumps(
            inventory,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def collective_verification_receipt(self) -> dict[str, Any]:
        """Create a rank-zero proof after every artifact was hashed and loaded."""
        producer_rank, world_size = _observed_distributed_context()
        if producer_rank != 0:
            raise ValueError("Only observed distributed rank zero may issue the receipt")
        expected = set(range(len(self)))
        if self._verified_artifact_indices != expected:
            raise ValueError("Collective proof requires every artifact hash")
        if self._deserialized_artifact_indices != expected:
            raise ValueError("Collective proof requires every artifact deserialization")
        return {
            "format_version": COLLECTIVE_VERIFICATION_FORMAT,
            "manifest_sha256": self.contract.manifest_sha256,
            "authorization_manifest_sha256": (
                self.contract.authorization_manifest_sha256
            ),
            "artifact_inventory_sha256": self._artifact_inventory_sha256(),
            "instance_count": len(self),
            "producer_rank_observed": producer_rank,
            "world_size_observed": world_size,
            "cache_external_anchor_verified": (
                self.contract.cache_external_anchor_verified
            ),
            "authorization_external_anchor_verified": (
                self.contract.authorization_external_anchor_verified
            ),
        }

    def accept_collective_verification_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        expected_receipt_sha256: str | None = None,
    ) -> int:
        """Trust rank-zero hashes only while every local stat fingerprint is stable."""
        _, observed_world_size = _observed_distributed_context()
        expected_fields = {
            "format_version",
            "manifest_sha256",
            "authorization_manifest_sha256",
            "artifact_inventory_sha256",
            "instance_count",
            "producer_rank_observed",
            "world_size_observed",
            "cache_external_anchor_verified",
            "authorization_external_anchor_verified",
        }
        if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
            raise ValueError("Collective verification receipt fields differ")
        expected = {
            "format_version": COLLECTIVE_VERIFICATION_FORMAT,
            "manifest_sha256": self.contract.manifest_sha256,
            "authorization_manifest_sha256": (
                self.contract.authorization_manifest_sha256
            ),
            "artifact_inventory_sha256": self._artifact_inventory_sha256(),
            "instance_count": len(self),
            "producer_rank_observed": 0,
            "world_size_observed": observed_world_size,
            "cache_external_anchor_verified": (
                self.contract.cache_external_anchor_verified
            ),
            "authorization_external_anchor_verified": (
                self.contract.authorization_external_anchor_verified
            ),
        }
        if dict(receipt) != expected:
            raise ValueError("Collective verification receipt differs from local metadata")
        actual_receipt_sha256 = collective_verification_receipt_sha256(receipt)
        if expected_receipt_sha256 is not None and (
            _sha256(
                expected_receipt_sha256,
                "expected_receipt_sha256",
            )
            != actual_receipt_sha256
        ):
            raise ValueError(
                "Collective verification receipt differs from its external anchor"
            )
        for index, record in enumerate(self.contract.records):
            path = self.contract.cache_root.joinpath(*record.artifact_path.parts)
            if self._stat_fingerprint(path) != self._artifact_stat_fingerprints[index]:
                raise ValueError(
                    "Cache artifact changed during collective verification: "
                    f"{record.artifact_path}"
                )
        self._verified_artifact_indices = set(range(len(self)))
        return len(self._verified_artifact_indices)

    def clear_memory_cache(self) -> int:
        """Clear loaded examples so every data-loader worker starts symmetrically."""
        removed = len(self._memory_cache)
        self._memory_cache.clear()
        return removed
