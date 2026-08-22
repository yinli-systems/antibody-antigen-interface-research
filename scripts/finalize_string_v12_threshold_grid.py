#!/usr/bin/env python3
"""Validate and freeze a one-pass STRING direct-evidence threshold audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


COMBINED_BINS = ("lt600", "600_649", "650_699", "700_749", "750_799", "800_899", "ge900")
DIRECT_BINS = ("zero", "1_99", "100_299", "300_499", "500_699", "ge700")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_create_only(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise FileExistsError(f"Refusing to overwrite threshold-grid receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_stats(path: Path) -> tuple[int, int, dict[str, dict[str, int]]]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("=", 1)
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise ValueError(f"Malformed threshold statistic: {line!r}")
        if fields[0] in values:
            raise ValueError(f"Duplicate threshold statistic: {fields[0]}")
        values[fields[0]] = int(fields[1])

    required = {"source_data_rows", "malformed_rows"}
    required.update(f"bin_{combined}_{direct}" for combined in COMBINED_BINS for direct in DIRECT_BINS)
    if set(values) != required:
        missing = sorted(required - set(values))
        extra = sorted(set(values) - required)
        raise ValueError(f"Threshold statistics schema differs: missing={missing} extra={extra}")
    source_rows = values.pop("source_data_rows")
    malformed_rows = values.pop("malformed_rows")
    if source_rows <= 0 or malformed_rows != 0:
        raise ValueError("STRING threshold audit has no source rows or observed malformed rows")
    matrix = {
        combined: {direct: values[f"bin_{combined}_{direct}"] for direct in DIRECT_BINS}
        for combined in COMBINED_BINS
    }
    if sum(sum(row.values()) for row in matrix.values()) != source_rows:
        raise ValueError("Threshold bins do not partition all STRING source rows")
    return source_rows, malformed_rows, matrix


def cumulative_candidates(matrix: dict[str, dict[str, int]]) -> list[dict[str, int]]:
    direct_minimum_bins = {
        700: ("ge700",),
        500: ("500_699", "ge700"),
        300: ("300_499", "500_699", "ge700"),
        100: ("100_299", "300_499", "500_699", "ge700"),
        1: ("1_99", "100_299", "300_499", "500_699", "ge700"),
    }
    combined_floor_bins = {
        650: ("650_699",),
        600: ("600_649", "650_699"),
    }
    candidates: list[dict[str, int]] = []
    rank = 0
    for direct_minimum, direct_bins in direct_minimum_bins.items():
        for combined_minimum, combined_bins in combined_floor_bins.items():
            rank += 1
            rows = sum(matrix[c][d] for c in combined_bins for d in direct_bins)
            candidates.append(
                {
                    "combined_maximum_exclusive": 700,
                    "combined_minimum_inclusive": combined_minimum,
                    "direct_channel_maximum_minimum_inclusive": direct_minimum,
                    "raw_rows": rows,
                    "selection_rank": rank,
                }
            )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--gpu-name", required=True)
    arguments = parser.parse_args()
    if not SHA256.fullmatch(arguments.input_sha256):
        raise ValueError("Invalid STRING input SHA-256")
    source = arguments.input.resolve()
    if source.stat().st_size != 15_528_028_374:
        raise ValueError("STRING physical-links input byte count differs")
    if sha256_file(source) != arguments.input_sha256:
        raise ValueError("STRING physical-links SHA-256 differs")
    source_rows, malformed_rows, matrix = parse_stats(arguments.stats.resolve())
    candidates = cumulative_candidates(matrix)
    receipt = {
        "authorization": {
            "candidate_filtering": False,
            "stage_a_training": False,
            "threshold_selection": False,
        },
        "candidate_selection_protocol": {
            "gate": "The first ranked candidate whose exact 50%-identity cluster-pair collapse, development decontamination, cross-source exact-pair deduplication, and license audit raise Stage A to at least 2,500,000 unique pair examples.",
            "ordering": "Maximize the strongest direct STRING evidence channel first (700, 500, 300, 100, 1), then maximize the combined-score floor (650 before 600).",
            "post_hoc_threshold_changes_permitted": False,
            "structural_views_count_as_unique_pairs": False,
        },
        "counts": {
            "malformed_rows": malformed_rows,
            "source_data_rows": source_rows,
        },
        "execution": {
            "allocated_gpu_count": 1,
            "allocated_gpu_name": arguments.gpu_name,
            "gpu_compute_used": False,
            "job_id": arguments.job_id,
            "partition": "gpu_4090",
            "reason_for_gpu_allocation": "The gpu_4090 partition rejects jobs that request zero GPUs.",
        },
        "format_version": "abpairfm_string_direct_threshold_grid_v1",
        "input": {
            "bytes": source.stat().st_size,
            "name": source.name,
            "sha256": arguments.input_sha256,
        },
        "matrix": matrix,
        "ranked_candidates": candidates,
        "scoring": {
            "combined_score_bins": list(COMBINED_BINS),
            "direct_channel": "max(experiments,database)",
            "direct_score_bins": list(DIRECT_BINS),
            "transferred_evidence_permitted": False,
        },
        "status": "AUDITED_NOT_SELECTED_NOT_AUTHORIZED",
        "test_access": False,
        "threshold_audit_script_sha256": sha256_file(arguments.audit_script.resolve()),
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
