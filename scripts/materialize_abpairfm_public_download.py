#!/usr/bin/env python3
"""Materialize one pinned public download plan with create-only evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any
import urllib.request


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
        raise FileExistsError(f"Refusing to overwrite receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_plan(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Download plan must be a JSON object")
    if value.get("format_version") != "abpairfm_public_download_plan_v1":
        raise ValueError("Unsupported download-plan format")
    if value.get("authorization", {}).get("download") is not True:
        raise ValueError("Download plan does not authorize materialization")
    if value.get("authorization", {}).get("stage_a_training") is not False:
        raise ValueError("Download plan must not authorize Stage A training")
    files = value.get("selected_files")
    if not isinstance(files, list) or not files:
        raise ValueError("selected_files must be a non-empty list")
    paths: set[str] = set()
    selected_bytes = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Each selected file must be an object")
        path = entry.get("path")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).name != path
            or path in {".", ".."}
        ):
            raise ValueError("Selected paths must be unique flat file names")
        if path in paths:
            raise ValueError(f"Duplicate selected path: {path}")
        paths.add(path)
        size = entry.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid size for {path}")
        selected_bytes += size
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"Selected URL must use HTTPS: {path}")
        etag = entry.get("etag")
        if not isinstance(etag, str) or not etag:
            raise ValueError(f"Missing pinned ETag: {path}")
        expected_sha256 = entry.get("sha256")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
        ):
            raise ValueError(f"Invalid pinned SHA-256: {path}")
    if value.get("expected_selected_file_count") != len(files):
        raise ValueError("Selected file count differs from plan summary")
    if value.get("expected_selected_bytes") != selected_bytes:
        raise ValueError("Selected byte count differs from plan summary")
    return value


def _normalized_etag(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().removeprefix("W/").strip('"')


def _if_range_value(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("W/") or stripped.startswith('"'):
        return stripped
    return f'"{stripped}"'


def _response_headers(path: Path) -> tuple[int, dict[str, str]]:
    if not path.is_file():
        raise RuntimeError("Range response did not write HTTP headers")
    status: int | None = None
    headers: dict[str, str] = {}
    for raw in path.read_text(encoding="iso-8859-1").splitlines():
        if raw.startswith("HTTP/"):
            fields = raw.split()
            if len(fields) < 2 or not fields[1].isdigit():
                raise RuntimeError("Malformed HTTP status in range response")
            status = int(fields[1])
            headers = {}
        elif ":" in raw and status is not None:
            name, value = raw.split(":", 1)
            headers[name.strip().lower()] = value.strip()
    if status is None:
        raise RuntimeError("Range response did not contain an HTTP status")
    return status, headers


def byte_ranges(total: int, count: int) -> list[tuple[int, int]]:
    if total <= 0 or count <= 0 or count > total:
        raise ValueError("Invalid byte-range partition")
    base, remainder = divmod(total, count)
    ranges = []
    start = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1
    if start != total:
        raise AssertionError("Byte ranges do not cover the pinned object")
    return ranges


def _verify_remote_metadata(entry: dict[str, Any]) -> None:
    request = urllib.request.Request(entry["url"], method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        if int(response.status) != 200:
            raise RuntimeError(f"Unexpected HEAD status for {entry['path']}")
        observed_size = int(response.headers.get("Content-Length", "-1"))
        if observed_size != int(entry["size_bytes"]):
            raise RuntimeError(f"Remote byte count differs for {entry['path']}")
        observed_etag = _normalized_etag(response.headers.get("ETag"))
        if observed_etag != _normalized_etag(str(entry["etag"])):
            raise RuntimeError(f"Remote ETag differs for {entry['path']}")


def _download_part(
    *,
    entry: dict[str, Any],
    index: int,
    start: int,
    end: int,
    parts_root: Path,
) -> dict[str, Any]:
    expected = end - start + 1
    final = parts_root / f"part-{index:03d}.bin"
    partial = parts_root / f"part-{index:03d}.partial"
    if final.exists():
        if final.stat().st_size != expected:
            raise RuntimeError(f"Completed part {index} has the wrong size")
        return {"attempts": 0, "bytes": expected, "index": index, "reused": True}
    if partial.exists() and partial.stat().st_size > expected:
        raise RuntimeError(f"Partial part {index} exceeds its frozen range")
    attempts = 0
    while (partial.stat().st_size if partial.exists() else 0) < expected:
        attempts += 1
        if attempts > 12:
            raise RuntimeError(f"Part {index} exceeded the retry budget")
        observed = partial.stat().st_size if partial.exists() else 0
        requested_start = start + observed
        response_headers = parts_root / f".part-{index:03d}.headers"
        response_headers.unlink(missing_ok=True)
        with partial.open("ab") as output:
            result = subprocess.run(
                [
                    "curl",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--fail",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "900",
                    "--max-filesize",
                    str(expected - observed),
                    "--dump-header",
                    str(response_headers),
                    "--header",
                    f"If-Range: {_if_range_value(str(entry['etag']))}",
                    "--range",
                    f"{requested_start}-{end}",
                    entry["url"],
                ],
                stdout=output,
                stderr=subprocess.PIPE,
                text=False,
            )
            output.flush()
            os.fsync(output.fileno())
        status, headers = _response_headers(response_headers)
        response_headers.unlink(missing_ok=True)
        expected_content_range = (
            f"bytes {requested_start}-{end}/{int(entry['size_bytes'])}"
        )
        response_valid = (
            result.returncode == 0
            and status == 206
            and headers.get("content-range") == expected_content_range
            and _normalized_etag(headers.get("etag"))
            == _normalized_etag(str(entry["etag"]))
        )
        if not response_valid:
            with partial.open("r+b") as output:
                output.truncate(observed)
            raise RuntimeError(
                f"Unsafe range response for part {index}: status={status}, "
                f"content_range={headers.get('content-range')!r}, "
                f"etag={headers.get('etag')!r}, curl={result.returncode}"
            )
        size = partial.stat().st_size
        print(
            json.dumps(
                {
                    "attempt": attempts,
                    "bytes": size,
                    "expected_bytes": expected,
                    "file": entry["path"],
                    "part": index,
                    "returncode": result.returncode,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if size > expected:
            raise RuntimeError(f"Server returned bytes outside part {index}'s range")
        if result.returncode != 0:
            time.sleep(min(30, attempts * 2))
    os.replace(partial, final)
    return {"attempts": attempts, "bytes": expected, "index": index, "reused": False}


def _download_segmented(
    entry: dict[str, Any],
    output_root: Path,
    *,
    parts: int,
    workers: int,
) -> dict[str, Any]:
    target = output_root / entry["path"]
    legacy_partial = target.with_name(f".{target.name}.partial")
    parts_root = target.with_name(f".{target.name}.parts-v1")
    parts_root.mkdir(parents=True, exist_ok=True)
    ranges = byte_ranges(int(entry["size_bytes"]), parts)
    adopted_prefix = 0
    if legacy_partial.exists():
        adopted_prefix = legacy_partial.stat().st_size
        if adopted_prefix <= 0 or adopted_prefix > ranges[0][1] + 1:
            raise RuntimeError("Existing prefix does not fit the first frozen range")
        first_partial = parts_root / "part-000.partial"
        if first_partial.exists() or (parts_root / "part-000.bin").exists():
            raise RuntimeError("Cannot adopt a prefix over an existing first part")
        os.replace(legacy_partial, first_partial)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _download_part,
                entry=entry,
                index=index,
                start=start,
                end=end,
                parts_root=parts_root,
            )
            for index, (start, end) in enumerate(ranges)
        ]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"completed_part": record}, sort_keys=True), flush=True)
    staging = target.with_name(f".{target.name}.assembling")
    if target.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite archive or staging file: {target}")
    digest = hashlib.sha256()
    total = 0
    with staging.open("xb") as destination:
        for index in range(parts):
            with (parts_root / f"part-{index:03d}.bin").open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    destination.write(block)
                    digest.update(block)
                    total += len(block)
        destination.flush()
        os.fsync(destination.fileno())
    if total != int(entry["size_bytes"]):
        raise RuntimeError(f"Assembled byte count differs for {target.name}")
    observed_sha256 = digest.hexdigest()
    expected_sha256 = entry.get("sha256")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        staging.unlink()
        raise RuntimeError(f"Assembled SHA-256 differs for {target.name}")
    os.replace(staging, target)
    target.chmod(0o444)
    return {
        "bytes": total,
        "download": {
            "adopted_prefix_bytes": adopted_prefix,
            "part_count": parts,
            "part_records": sorted(records, key=lambda item: int(item["index"])),
            "worker_count": workers,
        },
        "etag": entry["etag"],
        "sha256": observed_sha256,
        "url": entry["url"],
    }


def _download(
    entry: dict[str, Any],
    output_root: Path,
    *,
    parts: int,
    workers: int,
) -> dict[str, Any]:
    target = output_root / entry["path"]
    partial = target.with_name(f".{target.name}.partial")
    expected_size = int(entry["size_bytes"])
    expected_etag = str(entry["etag"])
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise RuntimeError(f"Target is not a regular file: {target}")
    if not target.exists() and parts > 1 and expected_size >= 64 * 1024 * 1024:
        return _download_segmented(
            entry,
            output_root,
            parts=parts,
            workers=workers,
        )
    if not target.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise RuntimeError(f"Partial file exceeds pinned size: {partial}")
        headers = {}
        if offset:
            headers = {
                "Range": f"bytes={offset}-",
                "If-Range": _if_range_value(expected_etag),
            }
        request = urllib.request.Request(entry["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            status = int(response.status)
            if offset and status != 206:
                raise RuntimeError(
                    f"Server refused safe resume for {target.name}: HTTP {status}"
                )
            if not offset and status != 200:
                raise RuntimeError(f"Unexpected HTTP status for {target.name}: {status}")
            observed_etag = _normalized_etag(response.headers.get("ETag"))
            if observed_etag != _normalized_etag(expected_etag):
                raise RuntimeError(
                    f"ETag differs for {target.name}: {observed_etag!r}"
                )
            if offset:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError(f"Invalid Content-Range for {target.name}")
            mode = "ab" if partial.exists() else "xb"
            with partial.open(mode) as handle:
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
        if partial.stat().st_size != expected_size:
            raise RuntimeError(f"Downloaded byte count differs for {target.name}")
        try:
            os.link(partial, target)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite download: {target}") from error
        partial.unlink()
        target.chmod(0o444)
    if target.stat().st_size != expected_size:
        raise RuntimeError(f"Existing byte count differs for {target.name}")
    observed_sha256 = _sha256_file(target)
    pinned_sha256 = entry.get("sha256")
    if pinned_sha256 is not None and observed_sha256 != pinned_sha256:
        raise RuntimeError(f"Downloaded SHA-256 differs for {target.name}")
    return {
        "bytes": expected_size,
        "etag": expected_etag,
        "sha256": observed_sha256,
        "url": entry["url"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.parts < 1 or not 1 <= arguments.workers <= arguments.parts:
        raise ValueError("Invalid parts/workers configuration")
    plan_path = arguments.plan.resolve()
    plan = _validate_plan(json.loads(plan_path.read_text(encoding="utf-8")))
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for entry in plan["selected_files"]:
        _verify_remote_metadata(entry)
    files = {
        entry["path"]: _download(
            entry,
            output_root,
            parts=arguments.parts,
            workers=arguments.workers,
        )
        for entry in plan["selected_files"]
    }
    receipt = {
        "authorization": {
            "normalization": False,
            "stage_a_training": False,
        },
        "files": files,
        "format_version": "abpairfm_public_materialization_v1",
        "license": plan["license"],
        "pinned_revision": plan["pinned_revision"],
        "plan_sha256": _sha256_file(plan_path),
        "source_id": plan["source_id"],
        "status": "MATERIALIZED_NOT_NORMALIZED",
        "test_access": False,
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }
    _publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
