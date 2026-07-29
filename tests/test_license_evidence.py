"""Tests for deterministic primary-source license evidence verification."""

from __future__ import annotations

import hashlib
import json

import pytest

from interaction_first.license_evidence import (
    LicenseEvidenceError,
    RAW_BYTES_SHA256_V1,
    ZENODO_LICENSE_PROJECTION_V1,
    canonical_json_bytes,
    evidence_digest,
    validate_evidence_declaration,
    verify_observed_evidence,
    zenodo_license_projection,
)


def _zenodo_payload(*, dynamic_views: int = 10) -> bytes:
    return json.dumps(
        {
            "doi": "10.5281/zenodo.14845086",
            "id": 14845086,
            "metadata": {
                "license": {"id": "cc-by-4.0"},
                "resource_type": {"title": "Dataset", "type": "dataset"},
                "title": "PPIRef",
            },
            "stats": {"views": dynamic_views},
            "updated": f"dynamic-{dynamic_views}",
        },
        sort_keys=True,
    ).encode("utf-8")


def _zenodo_asset() -> dict[str, object]:
    projection = zenodo_license_projection(_zenodo_payload())
    digest = hashlib.sha256(canonical_json_bytes(projection)).hexdigest()
    return {
        "declared_license_spdx": "CC-BY-4.0",
        "evidence_digest_method": ZENODO_LICENSE_PROJECTION_V1,
        "evidence_projection": projection,
        "evidence_revision": "10.5281/zenodo.14845086",
        "evidence_sha256": digest,
        "evidence_url": "https://zenodo.org/api/records/14845086",
    }


def test_zenodo_projection_ignores_dynamic_statistics() -> None:
    first_digest, first_projection = evidence_digest(
        _zenodo_payload(dynamic_views=10),
        method=ZENODO_LICENSE_PROJECTION_V1,
    )
    second_digest, second_projection = evidence_digest(
        _zenodo_payload(dynamic_views=999),
        method=ZENODO_LICENSE_PROJECTION_V1,
    )
    assert first_projection == second_projection
    assert first_digest == second_digest
    observation = verify_observed_evidence(
        _zenodo_asset(), _zenodo_payload(dynamic_views=999)
    )
    assert observation["evidence_sha256"] == first_digest
    assert observation["raw_bytes_sha256"] != hashlib.sha256(
        _zenodo_payload(dynamic_views=10)
    ).hexdigest()


@pytest.mark.parametrize("field", ["license", "title", "doi", "record_id"])
def test_zenodo_identity_or_license_change_fails(field: str) -> None:
    payload = json.loads(_zenodo_payload())
    if field == "license":
        payload["metadata"]["license"]["id"] = "cc0-1.0"
    elif field == "title":
        payload["metadata"]["title"] = "Changed"
    elif field == "doi":
        payload["doi"] = "10.5281/zenodo.1"
    else:
        payload["id"] = 1
    with pytest.raises(LicenseEvidenceError, match="differs"):
        verify_observed_evidence(
            _zenodo_asset(), json.dumps(payload, sort_keys=True).encode("utf-8")
        )


def test_raw_bytes_are_verified_bitwise() -> None:
    raw = b"immutable license bytes\n"
    asset = {
        "declared_license_spdx": "MIT",
        "evidence_digest_method": RAW_BYTES_SHA256_V1,
        "evidence_projection": None,
        "evidence_revision": "commit",
        "evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_url": "https://example.org/license",
    }
    assert verify_observed_evidence(asset, raw)["size_bytes"] == len(raw)
    with pytest.raises(LicenseEvidenceError, match="digest differs"):
        verify_observed_evidence(asset, raw + b"changed")


def test_declarations_fail_closed_on_method_projection_and_url_mismatch() -> None:
    asset = _zenodo_asset()
    for field, value, message in (
        ("evidence_digest_method", "unknown", "unsupported"),
        ("evidence_projection", None, "must be an object"),
        (
            "evidence_url",
            "https://example.org/api/records/14845086",
            "must identify one API record",
        ),
        (
            "evidence_revision",
            "10.5281/zenodo.1",
            "differs from DOI",
        ),
    ):
        changed = dict(asset)
        changed[field] = value
        with pytest.raises(LicenseEvidenceError, match=message):
            validate_evidence_declaration(changed)


def test_duplicate_json_key_is_rejected() -> None:
    raw = (
        b'{"id":1,"id":2,"doi":"x","metadata":'
        b'{"title":"x","resource_type":{"type":"dataset"},'
        b'"license":{"id":"cc-by-4.0"}}}'
    )
    with pytest.raises(LicenseEvidenceError, match="duplicate JSON key"):
        zenodo_license_projection(raw)
