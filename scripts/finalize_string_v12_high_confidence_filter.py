#!/usr/bin/env python3
"""Finalize the STRING high-confidence physical-link scale filter."""

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
    "direct_channel_rows",
    "high_confidence_rows",
    "malformed_rows",
    "other_calibrated_channel_rows",
    "source_data_rows",
    "transferred_channel_rows",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if counts["source_data_rows"] <= 0 or counts["high_confidence_rows"] <= 0:
        raise ValueError("STRING high-confidence filter produced no usable evidence")
    if counts["malformed_rows"] != 0:
        raise ValueError("STRING high-confidence filter observed malformed rows")
    channel_partition = (
        counts["direct_channel_rows"]
        + counts["transferred_channel_rows"]
        + counts["other_calibrated_channel_rows"]
    )
    if channel_partition != counts["high_confidence_rows"]:
        raise ValueError("STRING evidence strata do not partition the output")
    if counts["high_confidence_rows"] > counts["source_data_rows"]:
        raise ValueError("STRING high-confidence rows exceed the source")
    return counts


def publish_create_only(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
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
        raise FileExistsError(f"Refusing to overwrite filter receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--filter-script", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--gpu-name", required=True)
    arguments = parser.parse_args()
    if not SHA256.fullmatch(arguments.input_sha256):
        raise ValueError("Invalid input SHA-256")
    source = arguments.input.resolve()
    output = arguments.output.resolve()
    protocol_path = arguments.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    construction = protocol.get("stage_a_scale_pool", {}).get("string_construction", {})
    if (
        protocol.get("format_version") != "abpairfm_3b_scale_expansion_v2"
        or protocol.get("authorization", {}).get("scale_pool_materialization") is not True
        or protocol.get("authorization", {}).get("production_pretraining") is not False
        or construction.get("combined_score_minimum") != 700
        or construction.get("evidence_channel_requirement")
        != "none_beyond_physical_network_membership"
    ):
        raise ValueError("Scale expansion protocol does not authorize this filter")
    if source.stat().st_size != 15_528_028_374:
        raise ValueError("STRING physical-links input byte count differs")
    if sha256_file(source) != arguments.input_sha256:
        raise ValueError("STRING physical-links SHA-256 differs")
    if output.suffix != ".zst" or output.stat().st_size <= 0:
        raise ValueError("Filtered output is not a non-empty Zstandard file")
    counts = parse_stats(arguments.stats.resolve())
    receipt = {
        "authorization": {
            "cluster_deduplication": False,
            "production_pretraining": False,
            "scale_pool_materialization": True,
        },
        "claim_boundary": "All STRING v12 physical-network rows at the externally documented conservative combined-score threshold. Endpoint clustering, development decontamination, pair collapse, real-data training, and SOTA claims remain unauthorized.",
        "counts": counts,
        "filter": {
            "combined_score_minimum": 700,
            "evidence_channel_requirement": "none_beyond_physical_network_membership",
            "score_documentation_url": "https://string-db.org/help/scores/",
        },
        "format_version": "abpairfm_string_high_confidence_filter_v2",
        "input": {
            "bytes": source.stat().st_size,
            "name": source.name,
            "sha256": arguments.input_sha256,
        },
        "output": {
            "bytes": output.stat().st_size,
            "name": output.name,
            "sha256": sha256_file(output),
        },
        "provenance": {
            "filter_script_sha256": sha256_file(arguments.filter_script.resolve()),
            "protocol_sha256": sha256_file(protocol_path),
        },
        "status": "FILTERED_HIGH_CONFIDENCE_NOT_CLUSTERED_NOT_DECONTAMINATED",
        "test_access": False,
    }
    publish_create_only(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
