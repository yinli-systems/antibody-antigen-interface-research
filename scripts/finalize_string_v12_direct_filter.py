#!/usr/bin/env python3
"""Finalize the streaming STRING direct-evidence filter with an exact receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_COUNTS = {
    "combined_score_ge_700_rows",
    "direct_evidence_kept_rows",
    "malformed_rows",
    "source_data_rows",
    "transferred_only_ge_700_rows",
}


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
        raise FileExistsError(f"Refusing to overwrite filter receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_stats(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("=", 1)
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise ValueError(f"Malformed filter statistic: {line!r}")
        if fields[0] in counts:
            raise ValueError(f"Duplicate filter statistic: {fields[0]}")
        counts[fields[0]] = int(fields[1])
    if set(counts) != REQUIRED_COUNTS:
        raise ValueError("Filter statistics do not match the frozen schema")
    if counts["source_data_rows"] <= 0 or counts["direct_evidence_kept_rows"] <= 0:
        raise ValueError("STRING filter produced no usable evidence")
    if counts["malformed_rows"] != 0:
        raise ValueError("STRING filter observed malformed source rows")
    if counts["direct_evidence_kept_rows"] > counts["combined_score_ge_700_rows"]:
        raise ValueError("Direct evidence count exceeds high-confidence source count")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--filter-script-sha256", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--gpu-name", required=True)
    arguments = parser.parse_args()
    if not SHA256.fullmatch(arguments.input_sha256):
        raise ValueError("Invalid input SHA-256")
    if not SHA256.fullmatch(arguments.filter_script_sha256):
        raise ValueError("Invalid filter-script SHA-256")
    source = arguments.input.resolve()
    output = arguments.output.resolve()
    if source.stat().st_size != 15_528_028_374:
        raise ValueError("STRING physical-links input byte count differs")
    if output.suffix != ".zst" or output.stat().st_size <= 0:
        raise ValueError("Filtered output is not a non-empty Zstandard file")
    counts = parse_stats(arguments.stats.resolve())
    receipt = {
        "authorization": {
            "cluster_deduplication": False,
            "stage_a_training": False,
        },
        "counts": counts,
        "filter": {
            "combined_score_minimum": 700,
            "direct_database_score_minimum_exclusive": 0,
            "direct_experiments_score_minimum_exclusive": 0,
            "preserved_columns": [
                "protein1",
                "protein2",
                "homology",
                "experiments",
                "experiments_transferred",
                "database",
                "database_transferred",
                "textmining",
                "textmining_transferred",
                "combined_score",
            ],
            "transferred_only_evidence_permitted": False,
        },
        "execution": {
            "allocated_gpu_count": 1,
            "allocated_gpu_name": arguments.gpu_name,
            "gpu_compute_used": False,
            "job_id": arguments.job_id,
            "partition": "gpu_4090",
            "reason_for_gpu_allocation": "The gpu_4090 partition rejects jobs that request zero GPUs.",
        },
        "filter_script_sha256": arguments.filter_script_sha256,
        "format_version": "abpairfm_string_direct_filter_v1",
        "input": {
            "bytes": source.stat().st_size,
            "name": source.name,
            "sha256": arguments.input_sha256,
        },
        "output": {
            "bytes": output.stat().st_size,
            "name": output.name,
            "sha256": _sha256_file(output),
        },
        "status": "FILTERED_NOT_CLUSTERED_NOT_DECONTAMINATED",
        "test_access": False,
    }
    _publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
