"""Fail-closed source, license, raw-byte, and normalization registry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .license_evidence import (
    LicenseEvidenceError,
    validate_evidence_declaration,
)


SOURCE_FORMAT = "pairscale_source_manifest_v1"
LICENSE_FORMAT = "pairscale_license_receipt_v2"
RAW_FORMAT = "pairscale_raw_sha256_inventory_v1"
NORMALIZATION_FORMAT = "pairscale_normalization_receipt_v1"
SOURCE_FILENAMES = frozenset(
    {
        "source_manifest.json",
        "license_receipt.json",
        "raw_sha256.json",
        "normalization_receipt.json",
    }
)
LICENSE_ROUTES = frozenset(
    {"commercial_ok", "noncommercial_only", "academic_only", "unresolved"}
)
USE_CHANNELS = frozenset({"commercial", "noncommercial_research", "academic"})
TRAINING_POLICIES = frozenset(
    {"eligible_after_materialization", "external_confirmatory_only", "prohibited"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DataRegistryError(ValueError):
    """Raised when a source registry cannot be admitted safely."""


@dataclass(frozen=True)
class RegistryAdmission:
    """One deterministic admission decision and all blocking reasons."""

    source_id: str
    use_channel: str
    purpose: str
    admitted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SourceRegistry:
    """One validated four-file source registry."""

    root: Path
    source: Mapping[str, Any]
    license: Mapping[str, Any]
    raw: Mapping[str, Any]
    normalization: Mapping[str, Any]
    file_sha256: Mapping[str, str]

    @property
    def source_id(self) -> str:
        return str(self.source["source_id"])


def load_source_registry(root: Path) -> SourceRegistry:
    """Load and fully validate one source directory without following symlinks."""

    root = Path(root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise DataRegistryError("source registry root must be a non-symlink directory")
    observed = {path.name for path in root.iterdir() if path.is_file()}
    if observed != SOURCE_FILENAMES:
        raise DataRegistryError(
            f"source files differ; missing={sorted(SOURCE_FILENAMES - observed)}, "
            f"extra={sorted(observed - SOURCE_FILENAMES)}"
        )
    payloads: dict[str, Mapping[str, Any]] = {}
    digests: dict[str, str] = {}
    for filename in sorted(SOURCE_FILENAMES):
        path = root / filename
        if path.is_symlink():
            raise DataRegistryError(f"registry file is a symlink: {filename}")
        raw_bytes = path.read_bytes()
        digests[filename] = hashlib.sha256(raw_bytes).hexdigest()
        payloads[filename] = _strict_json(raw_bytes, filename)
    _validate_source(payloads["source_manifest.json"])
    _validate_license(payloads["license_receipt.json"])
    _validate_raw(payloads["raw_sha256.json"])
    _validate_normalization(payloads["normalization_receipt.json"])
    source_id = payloads["source_manifest.json"]["source_id"]
    if root.name != source_id:
        raise DataRegistryError("source directory name differs from source_id")
    for filename, payload in payloads.items():
        if payload["source_id"] != source_id:
            raise DataRegistryError(f"{filename} source_id differs")
    pinned_revision = payloads["source_manifest.json"]["pinned_revision"]
    if payloads["raw_sha256.json"]["pinned_revision"] != pinned_revision:
        raise DataRegistryError("raw inventory revision differs from source manifest")
    normalization = payloads["normalization_receipt.json"]
    if (
        normalization["status"] == "complete"
        and normalization["input_raw_inventory_sha256"]
        != digests["raw_sha256.json"]
    ):
        raise DataRegistryError("normalization is not bound to the raw inventory")
    return SourceRegistry(
        root=root,
        source=payloads["source_manifest.json"],
        license=payloads["license_receipt.json"],
        raw=payloads["raw_sha256.json"],
        normalization=payloads["normalization_receipt.json"],
        file_sha256=dict(sorted(digests.items())),
    )


def decide_registry_admission(
    registry: SourceRegistry,
    *,
    use_channel: str,
    purpose: str,
) -> RegistryAdmission:
    """Decide admission without silently upgrading license or materialization state."""

    if use_channel not in USE_CHANNELS:
        raise DataRegistryError("unsupported use channel")
    if purpose not in {"training", "external_confirmatory_evaluation"}:
        raise DataRegistryError("unsupported registry purpose")
    reasons: list[str] = []
    if registry.license["receipt_status"] != "reviewed":
        reasons.append("license_receipt_unresolved")
    data_assets = [
        value for value in registry.license["assets"] if value["asset_scope"] == "data"
    ]
    if not data_assets:
        reasons.append("missing_data_license_asset")
    for asset in data_assets:
        route = asset["license_route"]
        if route == "unresolved":
            reasons.append("data_license_unresolved")
        elif use_channel == "commercial" and route != "commercial_ok":
            reasons.append("data_license_blocks_commercial_use")
        elif use_channel == "noncommercial_research" and route == "academic_only":
            reasons.append("data_license_requires_academic_use")
    training_policy = registry.source["training_policy"]
    if purpose == "training":
        if training_policy == "external_confirmatory_only":
            reasons.append("source_reserved_for_external_confirmatory_evaluation")
        elif training_policy == "prohibited":
            reasons.append("source_training_prohibited")
    elif not registry.source["confirmatory_evaluation_eligible"]:
        reasons.append("source_not_registered_for_confirmatory_evaluation")
    if registry.raw["status"] != "materialized":
        reasons.append("raw_bytes_not_materialized")
    if registry.normalization["status"] != "complete":
        reasons.append("normalization_not_complete")
    if registry.source["duplicate_groups_status"] != "complete":
        reasons.append("duplicate_groups_not_complete")
    canonical_reasons = tuple(sorted(set(reasons)))
    return RegistryAdmission(
        source_id=registry.source_id,
        use_channel=use_channel,
        purpose=purpose,
        admitted=not canonical_reasons,
        reasons=canonical_reasons,
    )


def load_registry_collection(root: Path) -> tuple[SourceRegistry, ...]:
    """Load every source registry in canonical source-id order."""

    root = Path(root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise DataRegistryError("registry collection root is invalid")
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if not directories:
        raise DataRegistryError("registry collection is empty")
    registries = tuple(load_source_registry(path) for path in directories)
    source_ids = [value.source_id for value in registries]
    if source_ids != sorted(source_ids) or len(set(source_ids)) != len(source_ids):
        raise DataRegistryError("registry source IDs are not canonical and unique")
    return registries


def _validate_source(payload: Mapping[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "format_version",
            "source_id",
            "display_name",
            "source_class",
            "canonical_record_url",
            "pinned_revision",
            "planned_roles",
            "training_policy",
            "confirmatory_evaluation_eligible",
            "sealed_evaluation_source",
            "declared_scale",
            "duplicate_groups_status",
            "registered_at_utc",
        },
        "source_manifest",
    )
    if payload["format_version"] != SOURCE_FORMAT:
        raise DataRegistryError("unsupported source manifest format")
    _identifier(payload["source_id"], "source_id")
    _nonempty_string(payload["display_name"], "display_name")
    _nonempty_string(payload["source_class"], "source_class")
    _https_url(payload["canonical_record_url"], "canonical_record_url")
    _nonempty_string(payload["pinned_revision"], "pinned_revision")
    roles = payload["planned_roles"]
    if type(roles) is not list or not roles or roles != sorted(set(roles)):
        raise DataRegistryError("planned_roles must be a sorted unique list")
    if payload["training_policy"] not in TRAINING_POLICIES:
        raise DataRegistryError("unsupported training policy")
    for name in ("confirmatory_evaluation_eligible", "sealed_evaluation_source"):
        if type(payload[name]) is not bool:
            raise DataRegistryError(f"{name} must be boolean")
    if payload["sealed_evaluation_source"] and not payload[
        "confirmatory_evaluation_eligible"
    ]:
        raise DataRegistryError("sealed evaluation source is not evaluation eligible")
    if payload["training_policy"] == "external_confirmatory_only" and not (
        payload["confirmatory_evaluation_eligible"]
        and payload["sealed_evaluation_source"]
    ):
        raise DataRegistryError("confirmatory-only source must be eligible and sealed")
    scale = _object(payload["declared_scale"], "declared_scale")
    _exact_fields(scale, {"count", "unit", "source_claim_only"}, "declared_scale")
    if type(scale["count"]) is not int or scale["count"] <= 0:
        raise DataRegistryError("declared scale count must be positive")
    _nonempty_string(scale["unit"], "declared scale unit")
    if scale["source_claim_only"] is not True:
        raise DataRegistryError("declared scale must remain a source claim")
    if payload["duplicate_groups_status"] not in {"not_computed", "complete"}:
        raise DataRegistryError("duplicate group status is invalid")
    _utc(payload["registered_at_utc"], "registered_at_utc")


def _validate_license(payload: Mapping[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "format_version",
            "source_id",
            "reviewed_at_utc",
            "receipt_status",
            "legal_review_required_for_release",
            "assets",
        },
        "license_receipt",
    )
    if payload["format_version"] != LICENSE_FORMAT:
        raise DataRegistryError("unsupported license receipt format")
    _identifier(payload["source_id"], "source_id")
    _utc(payload["reviewed_at_utc"], "reviewed_at_utc")
    if payload["receipt_status"] not in {"reviewed", "unresolved"}:
        raise DataRegistryError("license receipt status is invalid")
    if type(payload["legal_review_required_for_release"]) is not bool:
        raise DataRegistryError("legal review flag must be boolean")
    assets = payload["assets"]
    if type(assets) is not list or not assets:
        raise DataRegistryError("license assets must be a nonempty list")
    identities: list[tuple[str, str]] = []
    for index, asset_value in enumerate(assets):
        asset = _object(asset_value, f"assets[{index}]")
        _exact_fields(
            asset,
            {
                "asset_scope",
                "asset_name",
                "declared_license_spdx",
                "license_route",
                "commercial_use_permitted",
                "attribution_required",
                "evidence_url",
                "evidence_revision",
                "evidence_digest_method",
                "evidence_projection",
                "evidence_sha256",
                "evidence_note",
            },
            f"assets[{index}]",
        )
        if asset["asset_scope"] not in {"data", "code"}:
            raise DataRegistryError("asset_scope must be data or code")
        _nonempty_string(asset["asset_name"], "asset_name")
        _nonempty_string(asset["declared_license_spdx"], "declared_license_spdx")
        if asset["license_route"] not in LICENSE_ROUTES:
            raise DataRegistryError("unsupported license route")
        for name in ("commercial_use_permitted", "attribution_required"):
            if type(asset[name]) is not bool:
                raise DataRegistryError(f"{name} must be boolean")
        if asset["license_route"] == "commercial_ok" and not asset[
            "commercial_use_permitted"
        ]:
            raise DataRegistryError("commercial route contradicts permission flag")
        if asset["license_route"] != "commercial_ok" and asset[
            "commercial_use_permitted"
        ]:
            raise DataRegistryError("restricted route contradicts permission flag")
        _https_url(asset["evidence_url"], "evidence_url")
        _nonempty_string(asset["evidence_revision"], "evidence_revision")
        _sha256(asset["evidence_sha256"], "evidence_sha256")
        _nonempty_string(asset["evidence_note"], "evidence_note")
        try:
            validate_evidence_declaration(asset)
        except LicenseEvidenceError as exc:
            raise DataRegistryError(str(exc)) from exc
        identities.append((asset["asset_scope"], asset["asset_name"]))
    if identities != sorted(set(identities)):
        raise DataRegistryError("license assets must be sorted and unique")


def _validate_raw(payload: Mapping[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "format_version",
            "source_id",
            "pinned_revision",
            "status",
            "materialized_at_utc",
            "files",
        },
        "raw_sha256",
    )
    if payload["format_version"] != RAW_FORMAT:
        raise DataRegistryError("unsupported raw inventory format")
    _identifier(payload["source_id"], "source_id")
    _nonempty_string(payload["pinned_revision"], "pinned_revision")
    if payload["status"] not in {"not_materialized", "materialized"}:
        raise DataRegistryError("raw inventory status is invalid")
    files = payload["files"]
    if type(files) is not list:
        raise DataRegistryError("raw files must be a list")
    if payload["status"] == "not_materialized":
        if payload["materialized_at_utc"] is not None or files:
            raise DataRegistryError("unmaterialized raw inventory must be empty")
    else:
        _utc(payload["materialized_at_utc"], "materialized_at_utc")
        if not files:
            raise DataRegistryError("materialized raw inventory must not be empty")
    paths: list[str] = []
    for index, file_value in enumerate(files):
        file_record = _object(file_value, f"files[{index}]")
        _exact_fields(file_record, {"path", "size_bytes", "sha256"}, "raw file")
        path = _relative_path(file_record["path"], "raw file path")
        if type(file_record["size_bytes"]) is not int or file_record["size_bytes"] < 0:
            raise DataRegistryError("raw file size is invalid")
        _sha256(file_record["sha256"], "raw file SHA-256")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise DataRegistryError("raw file paths must be sorted and unique")


def _validate_normalization(payload: Mapping[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "format_version",
            "source_id",
            "status",
            "normalized_at_utc",
            "input_raw_inventory_sha256",
            "normalizer_source_sha256",
            "outputs",
        },
        "normalization_receipt",
    )
    if payload["format_version"] != NORMALIZATION_FORMAT:
        raise DataRegistryError("unsupported normalization receipt format")
    _identifier(payload["source_id"], "source_id")
    if payload["status"] not in {"not_run", "complete"}:
        raise DataRegistryError("normalization status is invalid")
    outputs = payload["outputs"]
    if type(outputs) is not list:
        raise DataRegistryError("normalization outputs must be a list")
    if payload["status"] == "not_run":
        if any(
            payload[name] is not None
            for name in (
                "normalized_at_utc",
                "input_raw_inventory_sha256",
                "normalizer_source_sha256",
            )
        ) or outputs:
            raise DataRegistryError("not-run normalization must be empty")
    else:
        _utc(payload["normalized_at_utc"], "normalized_at_utc")
        _sha256(payload["input_raw_inventory_sha256"], "raw inventory SHA-256")
        _sha256(payload["normalizer_source_sha256"], "normalizer source SHA-256")
        if not outputs:
            raise DataRegistryError("complete normalization must have outputs")
    paths: list[str] = []
    for index, output_value in enumerate(outputs):
        output = _object(output_value, f"outputs[{index}]")
        _exact_fields(output, {"path", "size_bytes", "sha256"}, "normalized output")
        path = _relative_path(output["path"], "normalized output path")
        if type(output["size_bytes"]) is not int or output["size_bytes"] < 0:
            raise DataRegistryError("normalized output size is invalid")
        _sha256(output["sha256"], "normalized output SHA-256")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise DataRegistryError("normalized output paths must be sorted and unique")


def _strict_json(raw: bytes, name: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise DataRegistryError(f"non-finite JSON constant in {name}: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DataRegistryError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataRegistryError(f"cannot parse {name}") from exc
    if type(value) is not dict:
        raise DataRegistryError(f"{name} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise DataRegistryError(
            f"{name} fields differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise DataRegistryError(f"{name} must be an object")
    return value


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", value):
        raise DataRegistryError(f"{name} is not a canonical identifier")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise DataRegistryError(f"{name} must be a canonical nonempty string")
    return value


def _https_url(value: Any, name: str) -> str:
    value = _nonempty_string(value, name)
    if not value.startswith("https://"):
        raise DataRegistryError(f"{name} must use HTTPS")
    return value


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise DataRegistryError(f"{name} must be a lowercase SHA-256")
    return value


def _utc(value: Any, name: str) -> str:
    value = _nonempty_string(value, name)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise DataRegistryError(f"{name} must be UTC with second precision")
    return value


def _relative_path(value: Any, name: str) -> str:
    value = _nonempty_string(value, name)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise DataRegistryError(f"{name} is not a safe relative path")
    return value
