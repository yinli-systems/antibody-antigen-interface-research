#!/usr/bin/env python3
"""Resumably materialize an authenticated non-test PINDER archive."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


EXPECTED_SIZE = 168_884_420_877
EXPECTED_MD5 = "28e6e2724c1597514c7f6b6774ff32b4"
URL = "https://storage.googleapis.com/pinder/2024-02/pdbs.zip"
GENERATION = "1718880407172355"
ALLOWED_OBJECT_NAMES = {"mappings.zip", "pdbs.zip"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--adopt-prefix", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--expected-size", type=int, default=EXPECTED_SIZE)
    parser.add_argument("--expected-md5", default=EXPECTED_MD5)
    parser.add_argument("--generation", default=GENERATION)
    return parser.parse_args()


def byte_ranges(total: int, count: int) -> list[tuple[int, int]]:
    if total <= 0 or count <= 0 or count > total:
        raise ValueError("Invalid byte-range partition")
    base, remainder = divmod(total, count)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1
    if start != total:
        raise AssertionError("Byte-range partition does not cover the object")
    return ranges


def _publish_create_only(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite PINDER receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _download_part(
    index: int,
    start: int,
    end: int,
    parts_root: Path,
    url: str,
) -> dict[str, object]:
    expected = end - start + 1
    final = parts_root / f"part-{index:03d}.bin"
    partial = parts_root / f"part-{index:03d}.partial"
    if final.exists():
        if final.stat().st_size != expected:
            raise RuntimeError(f"Completed part {index} has the wrong size")
        return {"bytes": expected, "index": index, "reused": True}
    if partial.exists() and partial.stat().st_size > expected:
        raise RuntimeError(f"Partial part {index} exceeds its frozen byte range")
    attempts = 0
    while (partial.stat().st_size if partial.exists() else 0) < expected:
        attempts += 1
        if attempts > 20:
            raise RuntimeError(f"Part {index} exceeded the retry budget")
        observed = partial.stat().st_size if partial.exists() else 0
        requested_start = start + observed
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
                    "1800",
                    "--range",
                    f"{requested_start}-{end}",
                    url,
                ],
                stdout=output,
                stderr=subprocess.PIPE,
                text=False,
            )
            output.flush()
            os.fsync(output.fileno())
        size = partial.stat().st_size
        print(
            json.dumps(
                {
                    "attempt": attempts,
                    "bytes": size,
                    "expected_bytes": expected,
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


def _adopt_prefix(prefix: Path, parts_root: Path, first_end: int) -> int:
    if prefix.is_symlink() or not prefix.is_file():
        raise RuntimeError("Adopted prefix must be a regular file")
    size = prefix.stat().st_size
    expected = first_end + 1
    if size <= 0 or size > expected:
        raise RuntimeError("Adopted prefix does not fit the first byte range")
    destination = parts_root / "part-000.partial"
    if destination.exists() or (parts_root / "part-000.bin").exists():
        raise RuntimeError("Cannot adopt a prefix over an existing first part")
    os.replace(prefix, destination)
    return size


def main() -> None:
    arguments = _arguments()
    if arguments.parts < 2 or arguments.workers < 1 or arguments.workers > arguments.parts:
        raise ValueError("Invalid parts/workers configuration")
    if arguments.expected_size <= 0:
        raise ValueError("Expected size must be positive")
    if len(arguments.expected_md5) != 32 or any(
        token not in "0123456789abcdef" for token in arguments.expected_md5
    ):
        raise ValueError("Expected MD5 must be 32 lowercase hexadecimal characters")
    output = arguments.output.resolve()
    object_name = Path(arguments.url).name
    if object_name not in ALLOWED_OBJECT_NAMES or output.name != object_name:
        raise ValueError(
            "Only explicitly named non-test PINDER pdbs.zip or mappings.zip "
            "objects are allowed"
        )
    parts_root = output.parent / f".{output.name}.parts-v1"
    parts_root.mkdir(parents=True, exist_ok=True)
    ranges = byte_ranges(arguments.expected_size, arguments.parts)
    adopted = 0
    if arguments.adopt_prefix is not None and arguments.adopt_prefix.exists():
        adopted = _adopt_prefix(arguments.adopt_prefix.resolve(), parts_root, ranges[0][1])
    started = time.time()
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = {
            pool.submit(
                _download_part,
                index,
                start,
                end,
                parts_root,
                arguments.url,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"completed_part": record}, sort_keys=True), flush=True)

    staging = output.with_name(f".{output.name}.assembling")
    if output.exists() or staging.exists():
        raise FileExistsError("Refusing to overwrite a PINDER archive or staging file")
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    total = 0
    with staging.open("xb") as destination:
        for index in range(arguments.parts):
            part = parts_root / f"part-{index:03d}.bin"
            with part.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    destination.write(block)
                    md5.update(block)
                    sha256.update(block)
                    total += len(block)
        destination.flush()
        os.fsync(destination.fileno())
    if total != arguments.expected_size or md5.hexdigest() != arguments.expected_md5:
        raise RuntimeError("Assembled PINDER archive failed size or provider MD5")
    os.replace(staging, output)
    receipt = {
        "archive": {
            "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest(),
            "size_bytes": total,
        },
        "claim_boundary": (
            f"Authenticated PINDER training {object_name} archive materialization "
            "only; no test archive was requested."
        ),
        "download": {
            "adopted_prefix_bytes": adopted,
            "elapsed_seconds": time.time() - started,
            "part_count": arguments.parts,
            "part_records": sorted(records, key=lambda item: int(item["index"])),
            "worker_count": arguments.workers,
        },
        "format_version": "pinder_non_test_segmented_materialization_v2",
        "provider": {
            "expected_md5": arguments.expected_md5,
            "expected_size_bytes": arguments.expected_size,
            "gcs_generation": arguments.generation,
            "url": arguments.url,
        },
        "status": "MATERIALIZED",
        "test_access": {
            "pinder_test_rows_used": False,
            "pinder_test_structure_archive_requested": False,
            "pinder_test_structure_bytes_accessed": False,
        },
    }
    _publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
