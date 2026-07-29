#!/usr/bin/env python3
"""Fetch and verify every primary-source license evidence declaration."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Mapping
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.data_registry import (  # noqa: E402
    load_registry_collection,
)
from interaction_first.license_evidence import (  # noqa: E402
    LicenseEvidenceError,
    canonical_json_bytes,
    verify_observed_evidence,
)


FORMAT_VERSION = "pairscale_primary_license_evidence_verification_v1"
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--expected-module-sha256", required=True)
    parser.add_argument("--expected-registry-module-sha256", required=True)
    parser.add_argument("--verified-at-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def registry_input_manifest(root: Path) -> tuple[Mapping[str, object], ...]:
    """Build the canonical digest inventory for all strict source registries."""

    root = root.expanduser().absolute()
    registries = load_registry_collection(root)
    records: list[Mapping[str, object]] = []
    for registry in registries:
        for filename, digest in sorted(registry.file_sha256.items()):
            path = registry.root / filename
            records.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                }
            )
    return tuple(records)


def registry_input_sha256(root: Path) -> str:
    """Hash the canonical registry input manifest."""

    return hashlib.sha256(canonical_json_bytes(registry_input_manifest(root))).hexdigest()


def fetch_evidence(url: str, *, timeout_seconds: float) -> bytes:
    """Fetch one bounded HTTPS evidence object with an explicit user agent."""

    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise LicenseEvidenceError("evidence URL must use HTTPS")
    if not isinstance(timeout_seconds, float) or not 0.0 < timeout_seconds <= 120.0:
        raise LicenseEvidenceError("timeout_seconds must be in (0, 120]")
    request = Request(
        url,
        headers={"User-Agent": "PAIRScale-license-evidence-verifier/1"},
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        final_url = response.geturl()
        if urlsplit(final_url).scheme != "https":
            raise LicenseEvidenceError("evidence redirect left HTTPS")
        declared_length = response.headers.get("Content-Length")
        if declared_length is not None:
            try:
                parsed_length = int(declared_length)
            except ValueError as exc:
                raise LicenseEvidenceError(
                    "evidence Content-Length is malformed"
                ) from exc
            if parsed_length < 0 or parsed_length > MAX_EVIDENCE_BYTES:
                raise LicenseEvidenceError("evidence response exceeds byte limit")
        payload = response.read(MAX_EVIDENCE_BYTES + 1)
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise LicenseEvidenceError("evidence response exceeds byte limit")
    if not payload:
        raise LicenseEvidenceError("evidence response is empty")
    return payload


def verify_registry_evidence(
    root: Path,
    *,
    timeout_seconds: float,
) -> tuple[Mapping[str, object], ...]:
    """Verify every asset declaration against freshly fetched source bytes."""

    registries = load_registry_collection(root)
    cache: dict[str, bytes] = {}
    results: list[Mapping[str, object]] = []
    for registry in registries:
        for asset in registry.license["assets"]:
            url = str(asset["evidence_url"])
            if url not in cache:
                cache[url] = fetch_evidence(url, timeout_seconds=timeout_seconds)
            observation = verify_observed_evidence(asset, cache[url])
            results.append(
                {
                    "asset_name": asset["asset_name"],
                    "asset_scope": asset["asset_scope"],
                    "declared_license_spdx": asset["declared_license_spdx"],
                    "evidence_digest_method": asset["evidence_digest_method"],
                    "evidence_revision": asset["evidence_revision"],
                    "evidence_url": url,
                    "observed_evidence_sha256": observation["evidence_sha256"],
                    "observed_projection": observation["projection"],
                    "observed_raw_bytes_sha256": observation["raw_bytes_sha256"],
                    "observed_size_bytes": observation["size_bytes"],
                    "source_id": registry.source_id,
                    "verified": True,
                }
            )
    return tuple(results)


def _sha256_file(path: Path) -> str:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise LicenseEvidenceError(f"expected a regular source file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_sha256(value: str, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise LicenseEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def _utc(value: str) -> str:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise LicenseEvidenceError("verified_at_utc must be a second-resolution UTC time")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise LicenseEvidenceError("verified_at_utc is not a valid UTC time") from exc
    return value


def _publish_create_only(path: Path, payload: object) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o440)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise LicenseEvidenceError("short write while publishing receipt")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o440)


def main() -> None:
    """Verify external anchors, fetch evidence, and publish one receipt."""

    args = _parse_args()
    script_path = Path(__file__).resolve()
    module_path = SOURCE_ROOT / "interaction_first" / "license_evidence.py"
    registry_module_path = SOURCE_ROOT / "interaction_first" / "data_registry.py"
    observed_script_sha256 = _sha256_file(script_path)
    observed_module_sha256 = _sha256_file(module_path)
    observed_registry_module_sha256 = _sha256_file(registry_module_path)
    if observed_script_sha256 != _expected_sha256(
        args.expected_script_sha256, "expected_script_sha256"
    ):
        raise LicenseEvidenceError("verifier script digest mismatch")
    if observed_module_sha256 != _expected_sha256(
        args.expected_module_sha256, "expected_module_sha256"
    ):
        raise LicenseEvidenceError("evidence module digest mismatch")
    if observed_registry_module_sha256 != _expected_sha256(
        args.expected_registry_module_sha256,
        "expected_registry_module_sha256",
    ):
        raise LicenseEvidenceError("registry module digest mismatch")
    initial_registry_manifest = registry_input_manifest(args.registry_root)
    observed_registry_sha256 = hashlib.sha256(
        canonical_json_bytes(initial_registry_manifest)
    ).hexdigest()
    if observed_registry_sha256 != _expected_sha256(
        args.expected_registry_sha256, "expected_registry_sha256"
    ):
        raise LicenseEvidenceError("registry input digest mismatch")
    assets = verify_registry_evidence(
        args.registry_root,
        timeout_seconds=float(args.timeout_seconds),
    )
    if registry_input_manifest(args.registry_root) != initial_registry_manifest:
        raise LicenseEvidenceError("registry inputs changed during verification")
    payload = {
        "all_passed": True,
        "assets": list(assets),
        "format_version": FORMAT_VERSION,
        "network_policy": {
            "https_only": True,
            "maximum_response_bytes": MAX_EVIDENCE_BYTES,
            "timeout_seconds": float(args.timeout_seconds),
        },
        "registry_input_manifest": list(initial_registry_manifest),
        "registry_input_sha256": observed_registry_sha256,
        "source_implementation_sha256": {
            "scripts/verify_data_registry_evidence.py": observed_script_sha256,
            "src/interaction_first/data_registry.py": (
                observed_registry_module_sha256
            ),
            "src/interaction_first/license_evidence.py": observed_module_sha256,
        },
        "verified_at_utc": _utc(args.verified_at_utc),
    }
    _publish_create_only(args.output, payload)


if __name__ == "__main__":
    main()
