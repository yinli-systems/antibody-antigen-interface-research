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
import shutil
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
    verified_seed_root: Path | None = None,
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
    seed_root = _prepare_verified_seed_root(
        verified_seed_root,
        source_id=plan.source_id,
        metadata_bytes=metadata_bytes,
    )
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
            complete_partial = False
            if partial.exists():
                try:
                    _verify_downloaded_artifact(partial, artifact)
                    complete_partial = True
                except SourceMaterializationError:
                    complete_partial = False
            if not complete_partial and seed_root is not None and not partial.exists():
                complete_partial = _copy_verified_seed_artifact(
                    seed_root=seed_root,
                    partial=partial,
                    artifact=artifact,
                )
            if not complete_partial:
                _run_curl(curl_binary, artifact.url, partial)
                _verify_downloaded_artifact(partial, artifact)
            _publish_verified_partial(partial, destination, artifact)
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


def _prepare_verified_seed_root(
    value: Path | None,
    *,
    source_id: str,
    metadata_bytes: bytes,
) -> Path | None:
    if value is None:
        return None
    root = Path(value).expanduser().absolute()
    authority = root / ".authority" / "provider_metadata.json"
    if (
        root.name != source_id
        or root.is_symlink()
        or not root.is_dir()
        or authority.is_symlink()
        or not authority.is_file()
        or authority.read_bytes() != metadata_bytes
    ):
        raise SourceMaterializationError("verified seed root authority differs")
    return root


def _copy_verified_seed_artifact(
    *,
    seed_root: Path,
    partial: Path,
    artifact: DownloadArtifact,
) -> bool:
    seed = _safe_destination(seed_root, artifact.relative_path)
    if seed.is_symlink() or not seed.is_file():
        return False
    try:
        _verify_downloaded_artifact(seed, artifact)
    except SourceMaterializationError:
        return False
    with seed.open("rb") as source, partial.open("xb") as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    _verify_downloaded_artifact(partial, artifact)
    return True


def _publish_verified_partial(
    partial: Path,
    destination: Path,
    artifact: DownloadArtifact,
) -> None:
    _verify_downloaded_artifact(partial, artifact)
    try:
        os.link(partial, destination)
    except FileExistsError as exc:
        raise SourceMaterializationError(
            "artifact appeared concurrently during materialization"
        ) from exc
    partial.unlink()


def verify_materialized_registered_source(
    *,
    plan_path: Path,
    registry_root: Path,
    source_root: Path,
) -> Mapping[str, Any]:
    """Independently verify a materialized source against its pinned authority."""

    plan = load_download_plan(plan_path)
    registry = load_source_registry(Path(registry_root) / plan.source_id)
    validate_plan_against_registry(plan, registry)
    source_root = Path(source_root).expanduser().absolute()
    if (
        source_root.name != plan.source_id
        or source_root.is_symlink()
        or not source_root.is_dir()
    ):
        raise SourceMaterializationError("materialized source root is invalid")
    inventory_path = source_root / "raw_sha256.json"
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise SourceMaterializationError("raw inventory is missing or unsafe")
    inventory_bytes = inventory_path.read_bytes()
    inventory = _strict_json(inventory_bytes, "raw inventory")
    _exact_fields(
        inventory,
        {
            "files",
            "format_version",
            "materialized_at_utc",
            "pinned_revision",
            "source_id",
            "status",
        },
        "raw inventory",
    )
    if (
        inventory["format_version"] != RAW_INVENTORY_FORMAT
        or inventory["status"] != "materialized"
        or inventory["source_id"] != plan.source_id
        or inventory["pinned_revision"] != plan.pinned_revision
    ):
        raise SourceMaterializationError("raw inventory identity differs")
    _utc_string(inventory["materialized_at_utc"], "materialized_at_utc")
    files = inventory["files"]
    if type(files) is not list or not files:
        raise SourceMaterializationError("raw inventory files are missing")
    inventory_records: dict[str, Mapping[str, Any]] = {}
    for index, record_value in enumerate(files):
        record = _object(record_value, f"files[{index}]")
        _exact_fields(record, {"path", "size_bytes", "sha256"}, "raw file")
        path = _safe_relative_path(record["path"], "raw file path")
        if path in inventory_records:
            raise SourceMaterializationError("raw inventory paths are not unique")
        _nonnegative_int(record["size_bytes"], "raw file size")
        _digest(record["sha256"], _SHA256, "raw file SHA-256")
        inventory_records[path] = record
    if list(inventory_records) != sorted(inventory_records):
        raise SourceMaterializationError("raw inventory paths are not sorted")
    observed_paths: list[str] = []
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise SourceMaterializationError("materialized source contains a symlink")
        if path.is_file():
            relative = path.relative_to(source_root).as_posix()
            if relative != "raw_sha256.json":
                observed_paths.append(relative)
    if sorted(observed_paths) != sorted(inventory_records):
        raise SourceMaterializationError("materialized files differ from raw inventory")
    authority_record = inventory_records.get(".authority/provider_metadata.json")
    if authority_record is None:
        raise SourceMaterializationError("provider metadata is absent from raw inventory")
    authority_path = source_root / ".authority" / "provider_metadata.json"
    metadata_bytes = authority_path.read_bytes()
    artifacts = derive_download_artifacts(plan, metadata_bytes)
    artifact_by_path = {value.relative_path: value for value in artifacts}
    expected_paths = sorted(
        [".authority/provider_metadata.json", *artifact_by_path]
    )
    if sorted(inventory_records) != expected_paths:
        raise SourceMaterializationError("raw inventory selection differs from plan")
    for path, record in inventory_records.items():
        absolute = _safe_destination(source_root, path)
        if absolute.stat().st_size != record["size_bytes"]:
            raise SourceMaterializationError("materialized file size differs from receipt")
        sha256 = _file_sha256(absolute)
        if sha256 != record["sha256"]:
            raise SourceMaterializationError("materialized file SHA-256 differs from receipt")
        artifact = artifact_by_path.get(path)
        if artifact is not None:
            _verify_downloaded_artifact(absolute, artifact)
    verifier_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "authority_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "data_file_count": len(artifacts),
        "format_version": "pairscale_materialization_verification_v1",
        "plan_file_sha256": plan.file_sha256,
        "pinned_revision": plan.pinned_revision,
        "raw_inventory_file_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "source_id": plan.source_id,
        "total_data_bytes": sum(value.size_bytes for value in artifacts),
        "verified_at_utc": _utc_now(),
        "verifier_source_sha256": verifier_source_sha256,
    }


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


def write_create_only_receipt(path: Path, value: Mapping[str, Any]) -> None:
    """Write one canonical verification receipt without replacing prior evidence."""

    path = Path(path).expanduser().absolute()
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise SourceMaterializationError(
            "receipt parent must be an existing non-symlink directory"
        )
    _write_create_only_or_verify(path, _canonical_json_bytes(value))


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_string(value: Any, name: str) -> str:
    value = _nonempty_string(value, name)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise SourceMaterializationError(f"{name} must be UTC with second precision")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
