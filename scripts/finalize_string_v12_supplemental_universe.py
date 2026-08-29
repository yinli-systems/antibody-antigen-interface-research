#!/usr/bin/env python3
"""Freeze the preregistered STRING 600--699 direct-evidence universe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_create_only(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite STRING supplemental receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_stats(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("=", 1)
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise ValueError(f"Malformed supplemental filter statistic: {line!r}")
        if fields[0] in values:
            raise ValueError(f"Duplicate supplemental filter statistic: {fields[0]}")
        values[fields[0]] = int(fields[1])
    if set(values) != {"candidate_rows", "malformed_rows", "source_data_rows"}:
        raise ValueError("Supplemental filter statistics schema differs")
    if values["source_data_rows"] <= 0 or values["candidate_rows"] <= 0 or values["malformed_rows"] != 0:
        raise ValueError("Supplemental filter is empty or observed malformed rows")
    return values


def expected_candidate_rows(grid: dict[str, object]) -> int:
    if grid.get("status") != "AUDITED_NOT_SELECTED_NOT_AUTHORIZED" or grid.get("test_access") is not False:
        raise ValueError("STRING threshold-grid receipt is not eligible")
    matrix = grid.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("STRING threshold-grid matrix is unavailable")
    direct_bins = ("1_99", "100_299", "300_499", "500_699", "ge700")
    return sum(int(matrix[combined][direct]) for combined in ("600_649", "650_699") for direct in direct_bins)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--threshold-grid-receipt", type=Path, required=True)
    parser.add_argument("--filter-script", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--gpu-name", required=True)
    arguments = parser.parse_args()
    if not SHA256.fullmatch(arguments.input_sha256):
        raise ValueError("Invalid STRING input SHA-256")
    source = arguments.input.resolve()
    if source.stat().st_size != 15_528_028_374 or sha256_file(source) != arguments.input_sha256:
        raise ValueError("STRING physical-links source differs")
    grid_path = arguments.threshold_grid_receipt.resolve()
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    if grid.get("input", {}).get("sha256") != arguments.input_sha256:
        raise ValueError("STRING threshold-grid source differs")
    counts = parse_stats(arguments.stats.resolve())
    expected = expected_candidate_rows(grid)
    if counts["candidate_rows"] != expected:
        raise ValueError(f"Supplemental universe differs from threshold grid: {counts['candidate_rows']} != {expected}")
    output = arguments.output.resolve()
    if output.suffix != ".zst" or output.stat().st_size <= 0:
        raise ValueError("STRING supplemental output is not a non-empty Zstandard file")
    receipt = {
        "authorization": {"threshold_selection": False, "stage_a_training": False},
        "claim_boundary": "Broad preregistered STRING supplemental candidate universe only. No threshold has been selected; cluster-pair collapse, development decontamination, tier-1 deduplication, and Stage A authorization remain pending.",
        "counts": counts,
        "execution": {
            "allocated_gpu_count": 1,
            "allocated_gpu_name": arguments.gpu_name,
            "gpu_compute_used": False,
            "job_id": arguments.job_id,
            "partition": "gpu_4090",
            "reason_for_gpu_allocation": "The gpu_4090 partition rejects jobs that request zero GPUs.",
        },
        "filter": {
            "combined_score_maximum_exclusive": 700,
            "combined_score_minimum_inclusive": 600,
            "direct_channel": "max(experiments,database)",
            "direct_channel_minimum_inclusive": 1,
            "transferred_evidence_permitted": False,
        },
        "format_version": "abpairfm_string_supplemental_universe_v1",
        "input": {"bytes": source.stat().st_size, "sha256": arguments.input_sha256},
        "input_receipts": {"threshold_grid_sha256": sha256_file(grid_path)},
        "output": {"bytes": output.stat().st_size, "sha256": sha256_file(output)},
        "status": "FILTERED_UNIVERSE_NOT_SELECTED_NOT_CLUSTERED",
        "test_access": False,
        "filter_script_sha256": sha256_file(arguments.filter_script.resolve()),
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
