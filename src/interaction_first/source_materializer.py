"""Materialize registered public sources from pinned provider metadata."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from interaction_first.data_registry import (
    SourceRegistry,
    load_source_registry,
)


DOWNLOAD_PLAN_FORMAT = "pairscale_download_plan_v1"
RAW_INVENTORY_FORMAT = "pairscale_raw_sha256_inventory_v1"
PROVIDERS = frozenset({"huggingface_dataset", "zenodo_record"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")


class SourceMaterializationError(RuntimeError):
    """Raised when a source cannot be authenticated or materialized safely."""


@dataclass(frozen=True)
class DownloadPlan:
    """One validated, provider-pinned download plan."""

    path: Path
    source_id: str
    provider: str
    pinned_revision: str
    metadata_url: str
    metadata_sha256: str
    selectors: tuple[str, ...]
    expected_selected_file_count: int
    expected_selected_bytes: int
    locator: Mapping[str, Any]
    file_sha256: str


@dataclass(frozen=True)
class DownloadArtifact:
    """One provider-authenticated file selected for materialization."""

    relative_path: str
    url: str
    size_bytes: int
    upstream_algorithm: str
    upstream_digest: str


def load_download_plan(path: Path) -> DownloadPlan:
    """Load one strict download plan and validate all non-provider fields."""

    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise SourceMaterializationError("download plan must be a non-symlink file")
    raw = path.read_bytes()
    payload = _strict_json(raw, "download plan")
    _exact_fields(
        payload,
        {
            "format_version",
            "source_id",
            "provider",
            "pinned_revision",
            "metadata_url",
            "metadata_sha256",
            "selectors",
            "expected_selected_file_count",
            "expected_selected_bytes",
            "locator",
        },
        "download plan",
    )
    if payload["format_version"] != DOWNLOAD_PLAN_FORMAT:
        raise SourceMaterializationError("unsupported download plan format")
    source_id = _identifier(payload["source_id"], "source_id")
    provider = payload["provider"]
    if provider not in PROVIDERS:
        raise SourceMaterializationError("unsupported source provider")
    pinned_revision = _nonempty_string(payload["pinned_revision"], "pinned_revision")
    metadata_url = _https_url(payload["metadata_url"], "metadata_url")
    metadata_sha256 = _digest(payload["metadata_sha256"], _SHA256, "metadata SHA-256")
    selectors_value = payload["selectors"]
    if (
        type(selectors_value) is not list
        or not selectors_value
        or selectors_value != sorted(set(selectors_value))
    ):
        raise SourceMaterializationError("selectors must be a sorted unique list")
    selectors = tuple(_selector(value) for value in selectors_value)
    expected_count = _positive_int(
        payload["expected_selected_file_count"], "expected file count"
    )
    expected_bytes = _positive_int(
        payload["expected_selected_bytes"], "expected selected bytes"
    )
    locator = _object(payload["locator"], "locator")
    if provider == "huggingface_dataset":
        _exact_fields(locator, {"dataset_repo_id"}, "Hugging Face locator")
        repo_id = _nonempty_string(locator["dataset_repo_id"], "dataset_repo_id")
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
            raise SourceMaterializationError("dataset_repo_id is not canonical")
    else:
        _exact_fields(locator, {"record_id"}, "Zenodo locator")
        _positive_int(locator["record_id"], "Zenodo record ID")
    return DownloadPlan(
        path=path,
        source_id=source_id,
        provider=provider,
        pinned_revision=pinned_revision,
        metadata_url=metadata_url,
        metadata_sha256=metadata_sha256,
        selectors=selectors,
        expected_selected_file_count=expected_count,
        expected_selected_bytes=expected_bytes,
        locator=dict(locator),
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def validate_plan_against_registry(
    plan: DownloadPlan,
    registry: SourceRegistry,
) -> None:
    """Require a download plan to match its reviewed source registry exactly."""

    if plan.source_id != registry.source_id:
        raise SourceMaterializationError("plan source differs from registry")
    if plan.pinned_revision != registry.source["pinned_revision"]:
        raise SourceMaterializationError("plan revision differs from registry")


def derive_download_artifacts(
    plan: DownloadPlan,
    metadata_bytes: bytes,
) -> tuple[DownloadArtifact, ...]:
    """Derive an authenticated canonical file list from provider metadata bytes."""

    observed_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    if observed_sha256 != plan.metadata_sha256:
        raise SourceMaterializationError("provider metadata SHA-256 differs")
    metadata = _strict_json(metadata_bytes, "provider metadata")
    if plan.provider == "huggingface_dataset":
        artifacts = _derive_huggingface_artifacts(plan, metadata)
    else:
        artifacts = _derive_zenodo_artifacts(plan, metadata)
    _validate_selection(plan, artifacts)
    return artifacts


def fetch_provider_metadata(
    plan: DownloadPlan,
    *,
    curl_binary: str = "curl",
) -> bytes:
    """Fetch and authenticate the small provider metadata authority object."""

    command = [
        curl_binary,
        "--fail",
        "--location",
        "--retry",
        "5",
        "--retry-all-errors",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--max-time",
        "300",
        "--user-agent",
        "PAIRScale/1.0",
        plan.metadata_url,
    ]
    try:
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceMaterializationError("provider metadata request failed") from exc
    metadata_bytes = completed.stdout
    if hashlib.sha256(metadata_bytes).hexdigest() != plan.metadata_sha256:
        raise SourceMaterializationError("provider metadata SHA-256 differs")
    return metadata_bytes


def materialize_registered_source(
    *,
    plan_path: Path,
    registry_root: Path,
    destination_root: Path,
    curl_binary: str = "curl",
    workers: int = 1,
) -> Mapping[str, Any]:
    """Download one registered source and emit a create-only raw inventory."""

    if type(workers) is not int or not 1 <= workers <= 32:
        raise SourceMaterializationError("workers must be an integer from 1 to 32")
    plan = load_download_plan(plan_path)
    registry = load_source_registry(Path(registry_root) / plan.source_id)
    validate_plan_against_registry(plan, registry)
    metadata_bytes = fetch_provider_metadata(plan, curl_binary=curl_binary)
    artifacts = derive_download_artifacts(plan, metadata_bytes)
    source_root = _prepare_source_root(destination_root, plan.source_id)
    authority_path = source_root / ".authority" / "provider_metadata.json"
    _ensure_safe_directory(source_root, PurePosixPath(".authority"))
    _write_create_only_or_verify(authority_path, metadata_bytes)
    inventory_files: list[dict[str, Any]] = [
        {
            "path": ".authority/provider_metadata.json",
            "size_bytes": len(metadata_bytes),
            "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        }
    ]
    for artifact in artifacts:
        destination = _safe_destination(source_root, artifact.relative_path)
        relative_parent = PurePosixPath(artifact.relative_path).parent
        _ensure_safe_directory(source_root, relative_parent)
        if destination.exists() and (
            destination.is_symlink() or not destination.is_file()
        ):
            raise SourceMaterializationError("existing artifact is not a regular file")
        partial = destination.with_name(f".{destination.name}.partial")
        if partial.is_symlink() or (partial.exists() and not partial.is_file()):
            raise SourceMaterializationError("partial artifact is not a regular file")

    def materialize_one(artifact: DownloadArtifact) -> dict[str, Any]:
        destination = _safe_destination(source_root, artifact.relative_path)
        if destination.exists():
            sha256 = _verify_downloaded_artifact(destination, artifact)
        else:
            partial = destination.with_name(f".{destination.name}.partial")
            _run_curl(curl_binary, artifact.url, partial)
            _verify_downloaded_artifact(partial, artifact)
            try:
                os.link(partial, destination)
            except FileExistsError as exc:
                raise SourceMaterializationError(
                    "artifact appeared concurrently during materialization"
                ) from exc
            partial.unlink()
            sha256 = _verify_downloaded_artifact(destination, artifact)
        return {
            "path": artifact.relative_path,
            "size_bytes": artifact.size_bytes,
            "sha256": sha256,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        inventory_files.extend(executor.map(materialize_one, artifacts))
    inventory_files.sort(key=lambda value: value["path"])
    inventory = {
        "files": inventory_files,
        "format_version": RAW_INVENTORY_FORMAT,
        "materialized_at_utc": _utc_now(),
        "pinned_revision": plan.pinned_revision,
        "source_id": plan.source_id,
        "status": "materialized",
    }
    inventory_bytes = _canonical_json_bytes(inventory)
    _write_create_only_or_verify(source_root / "raw_sha256.json", inventory_bytes)
    return inventory


def _derive_huggingface_artifacts(
    plan: DownloadPlan,
    metadata: Mapping[str, Any],
) -> tuple[DownloadArtifact, ...]:
    if metadata.get("sha") != plan.pinned_revision:
        raise SourceMaterializationError("Hugging Face metadata revision differs")
    siblings = metadata.get("siblings")
    if type(siblings) is not list:
        raise SourceMaterializationError("Hugging Face siblings are missing")
    repo_id = str(plan.locator["dataset_repo_id"])
    candidates: dict[str, DownloadArtifact] = {}
    for index, sibling_value in enumerate(siblings):
        sibling = _object(sibling_value, f"siblings[{index}]")
        path = _safe_relative_path(sibling.get("rfilename"), "Hugging Face path")
        if path in candidates:
            raise SourceMaterializationError("duplicate Hugging Face path")
        size = _nonnegative_int(sibling.get("size"), "Hugging Face file size")
        lfs = sibling.get("lfs")
        if lfs is None:
            digest = _digest(sibling.get("blobId"), _SHA1, "Git blob SHA-1")
            algorithm = "git_blob_sha1"
        else:
            lfs_object = _object(lfs, f"siblings[{index}].lfs")
            digest = _digest(lfs_object.get("sha256"), _SHA256, "LFS SHA-256")
            if _nonnegative_int(lfs_object.get("size"), "LFS size") != size:
                raise SourceMaterializationError("LFS size differs from sibling size")
            algorithm = "sha256"
        encoded_path = quote(path, safe="/")
        url = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/"
            f"{plan.pinned_revision}/{encoded_path}?download=true"
        )
        candidates[path] = DownloadArtifact(path, url, size, algorithm, digest)
    return _select_artifacts(plan.selectors, candidates)


def _derive_zenodo_artifacts(
    plan: DownloadPlan,
    metadata: Mapping[str, Any],
) -> tuple[DownloadArtifact, ...]:
    if metadata.get("id") != plan.locator["record_id"]:
        raise SourceMaterializationError("Zenodo record ID differs")
    files = metadata.get("files")
    if type(files) is not list:
        raise SourceMaterializationError("Zenodo files are missing")
    candidates: dict[str, DownloadArtifact] = {}
    for index, file_value in enumerate(files):
        file_record = _object(file_value, f"files[{index}]")
        path = _safe_relative_path(file_record.get("key"), "Zenodo path")
        if path in candidates:
            raise SourceMaterializationError("duplicate Zenodo path")
        size = _nonnegative_int(file_record.get("size"), "Zenodo file size")
        checksum = _nonempty_string(file_record.get("checksum"), "Zenodo checksum")
        if not checksum.startswith("md5:"):
            raise SourceMaterializationError("unsupported Zenodo checksum")
        digest = _digest(checksum[4:], _MD5, "Zenodo MD5")
        links = _object(file_record.get("links"), f"files[{index}].links")
        url = _https_url(links.get("self"), "Zenodo content URL")
        candidates[path] = DownloadArtifact(path, url, size, "md5", digest)
    return _select_artifacts(plan.selectors, candidates)


def _select_artifacts(
    selectors: Sequence[str],
    candidates: Mapping[str, DownloadArtifact],
) -> tuple[DownloadArtifact, ...]:
    selected: dict[str, DownloadArtifact] = {}
    for selector in selectors:
        matches = [
            path for path in candidates if fnmatch.fnmatchcase(path, selector)
        ]
        if not matches:
            raise SourceMaterializationError(f"selector matched no files: {selector}")
        for path in matches:
            selected[path] = candidates[path]
    return tuple(selected[path] for path in sorted(selected))


def _validate_selection(
    plan: DownloadPlan,
    artifacts: Sequence[DownloadArtifact],
) -> None:
    if len(artifacts) != plan.expected_selected_file_count:
        raise SourceMaterializationError("selected file count differs from plan")
    if sum(value.size_bytes for value in artifacts) != plan.expected_selected_bytes:
        raise SourceMaterializationError("selected byte count differs from plan")


def _run_curl(curl_binary: str, url: str, partial: Path) -> None:
    command = [
        curl_binary,
        "--fail",
        "--location",
        "--retry",
        "5",
        "--retry-all-errors",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "300",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceMaterializationError("artifact download failed") from exc


def _verify_downloaded_artifact(path: Path, artifact: DownloadArtifact) -> str:
    if path.is_symlink() or not path.is_file():
        raise SourceMaterializationError("downloaded artifact is not a regular file")
    if path.stat().st_size != artifact.size_bytes:
        raise SourceMaterializationError("downloaded artifact size differs")
    sha256 = hashlib.sha256()
    if artifact.upstream_algorithm == "git_blob_sha1":
        upstream = hashlib.sha1(
            f"blob {artifact.size_bytes}\0".encode("ascii"), usedforsecurity=False
        )
    elif artifact.upstream_algorithm == "sha256":
        upstream = hashlib.sha256()
    elif artifact.upstream_algorithm == "md5":
        upstream = hashlib.md5(usedforsecurity=False)
    else:
        raise SourceMaterializationError("unsupported upstream digest algorithm")
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            sha256.update(chunk)
            upstream.update(chunk)
    if upstream.hexdigest() != artifact.upstream_digest:
        raise SourceMaterializationError("downloaded artifact digest differs")
    return sha256.hexdigest()


def _prepare_source_root(destination_root: Path, source_id: str) -> Path:
    destination_root = Path(destination_root).expanduser().absolute()
    if destination_root.is_symlink():
        raise SourceMaterializationError("destination root must not be a symlink")
    destination_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    source_root = destination_root / source_id
    if source_root.is_symlink():
        raise SourceMaterializationError("source destination must not be a symlink")
    source_root.mkdir(mode=0o750, exist_ok=True)
    return source_root


def _safe_destination(root: Path, relative_path: str) -> Path:
    canonical = _safe_relative_path(relative_path, "artifact path")
    destination = root.joinpath(*PurePosixPath(canonical).parts)
    if root not in destination.parents:
        raise SourceMaterializationError("artifact path escapes source root")
    return destination


def _ensure_safe_directory(root: Path, relative_directory: PurePosixPath) -> Path:
    current = root
    for part in relative_directory.parts:
        current = current / part
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise SourceMaterializationError("artifact parent is not a safe directory")
        else:
            current.mkdir(mode=0o750)
            if current.is_symlink() or not current.is_dir():
                raise SourceMaterializationError("artifact parent creation was unsafe")
    return current


def _write_create_only_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise SourceMaterializationError("create-only receipt already differs")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strict_json(raw: bytes, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise SourceMaterializationError(f"non-finite constant in {name}: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SourceMaterializationError(f"duplicate key in {name}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceMaterializationError(f"cannot parse {name}") from exc
    if type(value) is not dict:
        raise SourceMaterializationError(f"{name} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise SourceMaterializationError(f"{name} fields differ")


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise SourceMaterializationError(f"{name} must be an object")
    return value


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", value):
        raise SourceMaterializationError(f"{name} is not a canonical identifier")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise SourceMaterializationError(f"{name} must be a canonical string")
    return value


def _https_url(value: Any, name: str) -> str:
    value = _nonempty_string(value, name)
    if not value.startswith("https://"):
        raise SourceMaterializationError(f"{name} must use HTTPS")
    return value


def _digest(value: Any, pattern: re.Pattern[str], name: str) -> str:
    if type(value) is not str or not pattern.fullmatch(value):
        raise SourceMaterializationError(f"{name} is invalid")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise SourceMaterializationError(f"{name} must be positive")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise SourceMaterializationError(f"{name} must be nonnegative")
    return value


def _selector(value: Any) -> str:
    value = _nonempty_string(value, "selector")
    path = PurePosixPath(value)
    if (
        value.startswith("/")
        or not path.parts
        or str(path) != value
        or ".." in path.parts
        or "\\" in value
    ):
        raise SourceMaterializationError("selector is unsafe")
    return value


def _safe_relative_path(value: Any, name: str) -> str:
    value = _nonempty_string(value, name)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or str(path) != value
        or ".." in path.parts
        or "\\" in value
        or value.endswith("/")
    ):
        raise SourceMaterializationError(f"{name} is unsafe")
    return value
