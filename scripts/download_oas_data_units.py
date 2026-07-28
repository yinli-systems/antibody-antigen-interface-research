#!/usr/bin/env python3
"""Download one deterministic OAS authority shard with integrity receipts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

from antibody_pretraining.oas_authority import canonical_json_bytes
from antibody_pretraining.oas_download import (
    OASDownloadError,
    download_data_unit,
    load_authority_manifest,
    locate_curl,
    select_task_data_units,
)


def _source_sha256() -> str:
    """Return the controller source digest."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _write_create_only(path: Path, payload: bytes) -> None:
    """Create one immutable task artifact and reject existing targets."""
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


def _parser() -> argparse.ArgumentParser:
    """Build the strict command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--manifest-payload-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--maximum-units", type=int)
    parser.add_argument("--controller-source-sha256", required=True)
    parser.add_argument("--task-summary", type=Path, required=True)
    return parser


def main() -> int:
    """Download, validate, and summarize one modulo shard."""
    args = _parser().parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be in [1, 8]")
    observed_source_sha256 = _source_sha256()
    if observed_source_sha256 != args.controller_source_sha256:
        raise ValueError("Downloader controller source digest differs")
    manifest = load_authority_manifest(
        args.manifest,
        expected_file_sha256=args.manifest_file_sha256,
        expected_payload_sha256=args.manifest_payload_sha256,
    )
    selected = select_task_data_units(
        manifest["data_units"],
        task_id=args.task_id,
        task_count=args.task_count,
        maximum_units=args.maximum_units,
    )
    if not selected:
        raise OASDownloadError("Task shard contains no data units")
    source_class = manifest["source"]["source_class"]
    curl_binary, curl_version = locate_curl()
    raw_root = args.output_root / "raw" / source_class
    work_root = args.output_root / "work" / source_class
    receipt_root = args.output_root / "receipts" / source_class

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_entry = {
            executor.submit(
                download_data_unit,
                entry,
                raw_root=raw_root,
                work_root=work_root,
                receipt_root=receipt_root,
                authority_manifest_sha256=manifest["manifest_sha256"],
                controller_source_sha256=observed_source_sha256,
                source_class=source_class,
                curl_binary=curl_binary,
                curl_version=curl_version,
            ): entry
            for entry in selected
        }
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                successes.append(future.result())
            except BaseException as error:
                failures.append(
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "relative_unit": entry["relative_unit"],
                    }
                )

    successes.sort(key=lambda receipt: receipt["relative_unit"])
    failures.sort(key=lambda failure: failure["relative_unit"])
    summary = {
        "format_version": "oas_download_task_summary_v1",
        "authority_manifest_payload_sha256": manifest["manifest_sha256"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "controller_source_sha256": observed_source_sha256,
        "curl_version": curl_version,
        "failed_count": len(failures),
        "failures": failures,
        "host": socket.gethostname(),
        "selected_count": len(selected),
        "source_class": source_class,
        "success_count": len(successes),
        "successful_receipt_sha256": [
            receipt["receipt_sha256"] for receipt in successes
        ],
        "task_count": args.task_count,
        "task_id": args.task_id,
        "test_access": {
            "downstream_heldout_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "workers": args.workers,
    }
    summary["summary_sha256"] = hashlib.sha256(
        canonical_json_bytes(summary)
    ).hexdigest()
    _write_create_only(args.task_summary, canonical_json_bytes(summary))
    json.dump(summary, sys.stdout, ensure_ascii=True, sort_keys=True)
    sys.stdout.write("\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

