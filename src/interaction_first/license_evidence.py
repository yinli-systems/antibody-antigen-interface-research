"""Deterministic verification primitives for primary-source license evidence."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping
from urllib.parse import urlsplit


RAW_BYTES_SHA256_V1 = "raw_bytes_sha256_v1"
ZENODO_LICENSE_PROJECTION_V1 = "zenodo_license_projection_v1"
EVIDENCE_DIGEST_METHODS = frozenset(
    {RAW_BYTES_SHA256_V1, ZENODO_LICENSE_PROJECTION_V1}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ZENODO_RECORD_PATH_RE = re.compile(r"^/api/records/([1-9][0-9]*)/?$")


class LicenseEvidenceError(ValueError):
    """Raised when declared or observed license evidence is invalid."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode one deterministic ASCII JSON value without trailing whitespace."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise LicenseEvidenceError("value is not canonical-JSON encodable") from exc
    return encoded


def zenodo_license_projection(raw_bytes: bytes) -> Mapping[str, object]:
    """Extract stable identity and license fields from one Zenodo API record."""

    payload = _strict_json_object(raw_bytes, "Zenodo record")
    metadata = _mapping(payload.get("metadata"), "Zenodo metadata")
    resource_type = _mapping(
        metadata.get("resource_type"), "Zenodo resource_type"
    )
    license_value = _mapping(metadata.get("license"), "Zenodo license")
    record_id = payload.get("id")
    if type(record_id) is not int or record_id <= 0:
        raise LicenseEvidenceError("Zenodo record id must be a positive integer")
    projection: Mapping[str, object] = {
        "doi": _nonempty_string(payload.get("doi"), "Zenodo DOI"),
        "license_id": _nonempty_string(
            license_value.get("id"), "Zenodo license id"
        ),
        "record_id": record_id,
        "resource_type": _nonempty_string(
            resource_type.get("type"), "Zenodo resource type"
        ),
        "title": _nonempty_string(metadata.get("title"), "Zenodo title"),
    }
    return projection


def evidence_digest(
    raw_bytes: bytes,
    *,
    method: str,
) -> tuple[str, Mapping[str, object] | None]:
    """Compute the method-specific evidence digest and stable projection."""

    if type(raw_bytes) is not bytes:
        raise LicenseEvidenceError("evidence payload must be bytes")
    if method == RAW_BYTES_SHA256_V1:
        return hashlib.sha256(raw_bytes).hexdigest(), None
    if method == ZENODO_LICENSE_PROJECTION_V1:
        projection = zenodo_license_projection(raw_bytes)
        return hashlib.sha256(canonical_json_bytes(projection)).hexdigest(), projection
    raise LicenseEvidenceError("unsupported evidence digest method")


def validate_evidence_declaration(asset: Mapping[str, Any]) -> None:
    """Validate one receipt asset's digest method and declared projection."""

    method = asset.get("evidence_digest_method")
    if method not in EVIDENCE_DIGEST_METHODS:
        raise LicenseEvidenceError("unsupported evidence digest method")
    digest = asset.get("evidence_sha256")
    if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
        raise LicenseEvidenceError("evidence_sha256 must be a lowercase SHA-256")
    projection = asset.get("evidence_projection")
    if method == RAW_BYTES_SHA256_V1:
        if projection is not None:
            raise LicenseEvidenceError(
                "raw-byte evidence must declare a null projection"
            )
        return
    declared_projection = _validate_zenodo_projection(projection)
    projection_digest = hashlib.sha256(
        canonical_json_bytes(declared_projection)
    ).hexdigest()
    if projection_digest != digest:
        raise LicenseEvidenceError(
            "declared Zenodo projection differs from evidence_sha256"
        )
    url = _nonempty_string(asset.get("evidence_url"), "evidence_url")
    parsed = urlsplit(url)
    match = _ZENODO_RECORD_PATH_RE.fullmatch(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "zenodo.org"
        or parsed.query
        or parsed.fragment
        or match is None
    ):
        raise LicenseEvidenceError(
            "Zenodo projection evidence URL must identify one API record"
        )
    if int(match.group(1)) != declared_projection["record_id"]:
        raise LicenseEvidenceError("Zenodo URL record id differs from projection")
    if asset.get("evidence_revision") != declared_projection["doi"]:
        raise LicenseEvidenceError("Zenodo evidence revision differs from DOI")
    declared_spdx = _nonempty_string(
        asset.get("declared_license_spdx"), "declared_license_spdx"
    )
    if declared_spdx.lower() != declared_projection["license_id"]:
        raise LicenseEvidenceError(
            "declared SPDX identifier differs from Zenodo license id"
        )


def verify_observed_evidence(
    asset: Mapping[str, Any],
    raw_bytes: bytes,
) -> Mapping[str, object]:
    """Verify fetched bytes against a validated receipt asset declaration."""

    validate_evidence_declaration(asset)
    digest, projection = evidence_digest(
        raw_bytes,
        method=str(asset["evidence_digest_method"]),
    )
    if digest != asset["evidence_sha256"]:
        raise LicenseEvidenceError("observed evidence digest differs")
    if projection != asset["evidence_projection"]:
        raise LicenseEvidenceError("observed evidence projection differs")
    return {
        "evidence_sha256": digest,
        "projection": projection,
        "raw_bytes_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "size_bytes": len(raw_bytes),
    }


def _validate_zenodo_projection(value: Any) -> Mapping[str, object]:
    projection = _mapping(value, "Zenodo evidence projection")
    expected = {"doi", "license_id", "record_id", "resource_type", "title"}
    if set(projection) != expected:
        raise LicenseEvidenceError(
            "Zenodo evidence projection fields differ; "
            f"missing={sorted(expected - set(projection))}, "
            f"extra={sorted(set(projection) - expected)}"
        )
    record_id = projection["record_id"]
    if type(record_id) is not int or record_id <= 0:
        raise LicenseEvidenceError("Zenodo projection record id is invalid")
    for name in ("doi", "license_id", "resource_type", "title"):
        _nonempty_string(projection[name], f"Zenodo projection {name}")
    return projection


def _strict_json_object(raw_bytes: bytes, label: str) -> Mapping[str, Any]:
    if type(raw_bytes) is not bytes:
        raise LicenseEvidenceError(f"{label} must be bytes")
    try:
        text = raw_bytes.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LicenseEvidenceError(f"{label} is not strict UTF-8 JSON") from exc
    return _mapping(value, label)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LicenseEvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise LicenseEvidenceError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise LicenseEvidenceError(f"{label} must be a nonempty string")
    return value
