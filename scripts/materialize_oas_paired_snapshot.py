#!/usr/bin/env python3
"""Materialize a frozen OAS paired data-unit inventory with per-file evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import urllib.request


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
        raise FileExistsError(f"Refusing to overwrite OAS receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_plan(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("OAS plan must be a JSON object")
    if value.get("format_version") != "abpairfm_oas_paired_download_plan_v1":
        raise ValueError("Unexpected OAS plan format")
    if value.get("authorization", {}).get("download") is not True:
        raise ValueError("OAS plan does not authorize download")
    if value.get("authorization", {}).get("stage_b_training") is not False:
        raise ValueError("OAS plan must not authorize Stage B training")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != value.get("expected_data_unit_count"):
        raise ValueError("OAS file inventory is incomplete")
    return value


def inspect_data_unit(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        metadata_row = next(csv.reader([handle.readline()]))
        if len(metadata_row) != 1:
            raise ValueError(f"Malformed OAS metadata row: {path}")
        metadata = json.loads(metadata_row[0])
        header = next(csv.reader([handle.readline()]))
        required = {
            "sequence_alignment_aa_heavy",
            "sequence_alignment_aa_light",
            "sequence_id_heavy",
            "sequence_id_light",
        }
        if not required.issubset(header):
            raise ValueError(f"OAS paired columns are incomplete: {path}")
        rows = sum(1 for _ in handle)
    expected = metadata.get("Unique sequences")
    if type(expected) is not int or expected != rows or rows <= 0:
        raise ValueError(f"OAS metadata row count differs: {path}")
    return {
        "author": metadata.get("Author"),
        "rows": rows,
        "run": metadata.get("Run"),
        "species": metadata.get("Species"),
    }


def _download_one(entry: dict[str, str], output_root: Path) -> dict[str, object]:
    relative = Path(entry["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
        raise ValueError(f"Unsafe OAS target path: {relative}")
    target = output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"OAS target is not a regular file: {target}")
        data_unit = inspect_data_unit(target)
        return {
            **data_unit,
            "bytes": target.stat().st_size,
            "relative_path": entry["relative_path"],
            "reused": True,
            "sha256": _sha256_file(target),
            "url": entry["url"],
        }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary_path = Path(temporary)
    os.close(descriptor)
    try:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(
                    entry["url"], headers={"User-Agent": "ABPairFM-audit/1.0"}
                )
                digest = hashlib.sha256()
                size = 0
                with urllib.request.urlopen(request, timeout=120) as response:
                    if int(response.status) != 200:
                        raise RuntimeError(f"Unexpected HTTP status {response.status}")
                    with temporary_path.open("wb") as output:
                        for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                            output.write(block)
                            digest.update(block)
                            size += len(block)
                        output.flush()
                        os.fsync(output.fileno())
                break
            except Exception as error:
                last_error = error
                temporary_path.write_bytes(b"")
                if attempt == 3:
                    raise RuntimeError(f"OAS download failed: {entry['url']}") from error
                time.sleep(attempt * 2)
        else:
            raise RuntimeError("Unreachable OAS retry state") from last_error
        data_unit = inspect_data_unit(temporary_path)
        os.link(temporary_path, target)
        target.chmod(0o444)
        return {
            **data_unit,
            "bytes": size,
            "relative_path": entry["relative_path"],
            "reused": False,
            "sha256": digest.hexdigest(),
            "url": entry["url"],
        }
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 32:
        raise ValueError("workers must be in [1, 32]")
    plan_path = arguments.plan.resolve()
    plan = _validate_plan(json.loads(plan_path.read_text(encoding="utf-8")))
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = [
            pool.submit(_download_one, entry, output_root) for entry in plan["files"]
        ]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"completed": record["relative_path"]}), flush=True)
    records.sort(key=lambda item: str(item["relative_path"]))
    total_rows = sum(int(item["rows"]) for item in records)
    if total_rows != int(plan["expected_sequence_count"]):
        raise ValueError(
            f"Materialized OAS rows {total_rows} differ from snapshot "
            f"{plan['expected_sequence_count']}"
        )
    receipt = {
        "authorization": {
            "exact_pair_deduplication": False,
            "stage_b_training": False,
        },
        "data_unit_count": len(records),
        "files": records,
        "format_version": "abpairfm_oas_paired_materialization_v1",
        "license": plan["license"],
        "plan_sha256": _sha256_file(plan_path),
        "status": "MATERIALIZED_NOT_DEDUPLICATED_NOT_DECONTAMINATED",
        "test_access": False,
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "total_sequence_rows": total_rows,
    }
    _publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
