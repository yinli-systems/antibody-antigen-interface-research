#!/usr/bin/env python3
"""Freeze the official unfiltered OAS paired-search result into a download plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import urlparse


SUMMARY = re.compile(
    r"Your search yielded\s*<b>([\d,]+)</b> filtered sequences from\s*<b>(\d+)</b> studies"
)
URL = re.compile(
    r'"wget (https://opig\.stats\.ox\.ac\.uk/webapps/ngsdb/paired/[^" ]+\.csv\.gz)"'
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_create_only(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite OAS plan: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_snapshot(html: str) -> dict[str, object]:
    summary = SUMMARY.search(html)
    if summary is None:
        raise ValueError("OAS search summary was not found")
    sequence_count = int(summary.group(1).replace(",", ""))
    study_count = int(summary.group(2))
    urls = sorted(set(URL.findall(html)))
    entries = []
    studies = set()
    relative_paths = set()
    for url in urls:
        parsed = urlparse(url)
        path = PurePosixPath(parsed.path)
        parts = path.parts
        try:
            paired_index = parts.index("paired")
            study = parts[paired_index + 1]
            filename = parts[-1]
        except (ValueError, IndexError) as error:
            raise ValueError(f"Unexpected OAS URL: {url}") from error
        relative_path = f"{study}/{filename}"
        if relative_path in relative_paths:
            raise ValueError(f"Duplicate OAS target path: {relative_path}")
        relative_paths.add(relative_path)
        studies.add(study)
        entries.append({"relative_path": relative_path, "url": url})
    if not entries or len(studies) != study_count:
        raise ValueError("OAS URL inventory differs from the search summary")
    return {
        "authorization": {
            "download": True,
            "stage_b_training": False,
        },
        "expected_data_unit_count": len(entries),
        "expected_sequence_count": sequence_count,
        "expected_study_count": study_count,
        "files": entries,
        "filters": {
            "Age": "*",
            "BSource": "*",
            "BType": "*",
            "Disease": "*",
            "Longitudinal": "*",
            "Species": "*",
            "Subject": "*",
            "Vaccine": "*",
        },
        "format_version": "abpairfm_oas_paired_download_plan_v1",
        "license": "CC-BY-4.0",
        "search_url": "https://opig.stats.ox.ac.uk/webapps/oas/oas_paired",
        "status": "SNAPSHOT_NOT_MATERIALIZED",
        "test_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    html_path = arguments.html.resolve()
    plan = parse_snapshot(html_path.read_text(encoding="utf-8"))
    plan["source_html_sha256"] = _sha256_file(html_path)
    _publish_create_only(arguments.output.resolve(), plan)
    print(json.dumps(plan, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
