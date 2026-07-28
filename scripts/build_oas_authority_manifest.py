#!/usr/bin/env python3
"""Fetch or parse an official OAS search response and create authority artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any
from urllib.request import Request, urlopen

from antibody_pretraining.oas_authority import (
    SOURCE_SEARCH_URLS,
    build_oas_authority_manifest,
    canonical_json_bytes,
    verify_manifest_self_hash,
)


def _source_sha256() -> str:
    """Return the controller source digest."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _write_create_only(path: Path, payload: bytes) -> None:
    """Create one artifact atomically and reject any existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _fetch(source_class: str, timeout_seconds: float) -> tuple[bytes, dict[str, Any]]:
    """Fetch one unfiltered official search response with an empty POST body."""
    url = SOURCE_SEARCH_URLS[source_class]
    request = Request(
        url,
        data=b"",
        method="POST",
        headers={
            "Accept": "text/html",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "antibody-foundation-pretraining-authority-v1",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
        metadata = {
            "final_url": response.geturl(),
            "http_status": response.status,
            "response_content_type": response.headers.get_content_type(),
        }
    if metadata["http_status"] != 200:
        raise RuntimeError(f"OAS authority request returned {metadata['http_status']}")
    if metadata["final_url"] != url:
        raise RuntimeError("OAS authority request redirected unexpectedly")
    if metadata["response_content_type"] != "text/html":
        raise RuntimeError("OAS authority response is not HTML")
    return payload, metadata


def _parser() -> argparse.ArgumentParser:
    """Build the strict command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-class", choices=sorted(SOURCE_SEARCH_URLS), required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--fetch", action="store_true")
    input_group.add_argument("--input-html", type=Path)
    parser.add_argument("--raw-response-out", type=Path)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main() -> int:
    """Create the raw response, canonical manifest, and provenance receipt."""
    args = _parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    fetch_metadata: dict[str, Any]
    if args.fetch:
        if args.raw_response_out is None:
            raise ValueError("--raw-response-out is required with --fetch")
        html_bytes, fetch_metadata = _fetch(args.source_class, args.timeout_seconds)
        _write_create_only(args.raw_response_out, html_bytes)
    else:
        if args.raw_response_out is not None:
            raise ValueError("--raw-response-out is valid only with --fetch")
        html_bytes = args.input_html.read_bytes()
        fetch_metadata = {
            "input_html": str(args.input_html.resolve()),
            "mode": "offline_parse",
        }

    manifest = build_oas_authority_manifest(
        html_bytes,
        source_class=args.source_class,
    )
    if not verify_manifest_self_hash(manifest):
        raise RuntimeError("Internal manifest self-hash verification failed")
    manifest_bytes = canonical_json_bytes(manifest)
    _write_create_only(args.manifest_out, manifest_bytes)

    receipt = {
        "format_version": "oas_authority_creation_receipt_v1",
        "controller_source_sha256": _source_sha256(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fetch": fetch_metadata,
        "host": socket.gethostname(),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_payload_sha256": manifest["manifest_sha256"],
        "raw_response_sha256": hashlib.sha256(html_bytes).hexdigest(),
        "source_class": args.source_class,
        "test_access": {
            "downstream_heldout_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    _write_create_only(args.receipt_out, canonical_json_bytes(receipt))
    json.dump(
        {
            "data_unit_count": manifest["summary"]["data_unit_count"],
            "manifest_payload_sha256": manifest["manifest_sha256"],
            "raw_response_sha256": manifest["source"]["raw_response_sha256"],
            "study_count": manifest["summary"]["observed_study_count"],
            "unique_sequence_count": manifest["summary"][
                "headline_unique_sequence_count"
            ],
        },
        sys.stdout,
        ensure_ascii=True,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

