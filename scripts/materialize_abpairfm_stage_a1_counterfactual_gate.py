#!/usr/bin/env python3
"""Materialize a deterministic train-only Stage-A1 partner-swap gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable, Sequence


FORMAT_VERSION = "abpairfm_stage_a1_counterfactual_materialization_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zstd_lines(path: Path) -> Iterable[str]:
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        yield from process.stdout
    finally:
        process.stdout.close()
        error = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")


def _selection_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _length_bin(length: int) -> int:
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("Counterfactual endpoint length must be positive")
    return math.floor(4.0 * math.log2(length))


def _pair(left: str, right: str) -> tuple[str, str]:
    first, second = sorted((left, right))
    return first, second


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    required = (
        "endpoint_cluster1",
        "endpoint_cluster2",
        "endpoint_exact1_sha256",
        "endpoint_exact2_sha256",
        "example_id",
        "known_positive_group",
        "pilot_record_index",
        "sequence1",
        "sequence2",
        "source",
    )
    if any(not record.get(name) and record.get(name) != 0 for name in required):
        raise ValueError("Counterfactual record is missing identity metadata")
    return {
        "clusters": {
            str(record["endpoint_cluster1"]),
            str(record["endpoint_cluster2"]),
        },
        "exact": {
            str(record["endpoint_exact1_sha256"]),
            str(record["endpoint_exact2_sha256"]),
        },
        "group": f"{record['source']}\0{record['known_positive_group']}",
        "left_cluster": str(record["endpoint_cluster1"]),
        "left_exact": str(record["endpoint_exact1_sha256"]),
        "left_length": len(str(record["sequence1"])),
        "right_cluster": str(record["endpoint_cluster2"]),
        "right_exact": str(record["endpoint_exact2_sha256"]),
        "right_length": len(str(record["sequence2"])),
    }


def materialize_counterfactuals(
    records: Sequence[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Counterfactual seed must be a non-negative integer")
    retrieval_records = [record for record in records if record.get("retrieval_supervision")]
    if len(retrieval_records) < 2:
        raise ValueError("Counterfactual gate requires at least two retrieval records")
    if len({int(record["pilot_record_index"]) for record in retrieval_records}) != len(
        retrieval_records
    ):
        raise ValueError("Counterfactual pilot record indices are not unique")

    metadata = [_metadata(record) for record in retrieval_records]
    known_exact_pairs = {
        _pair(value["left_exact"], value["right_exact"]) for value in metadata
    }
    known_cluster_pairs = {
        _pair(value["left_cluster"], value["right_cluster"]) for value in metadata
    }
    pools: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, value in enumerate(metadata):
        pools[
            (_length_bin(value["left_length"]), _length_bin(value["right_length"]))
        ].append(index)
    for key, indices in pools.items():
        indices.sort(
            key=lambda index: _selection_seed(
                seed,
                "pool",
                key[0],
                key[1],
                retrieval_records[index]["example_id"],
            )
        )

    mapping_digest = hashlib.sha256()
    rejection_counts: Counter[str] = Counter()
    admissible_counts: list[int] = []
    candidate_use: Counter[int] = Counter()
    left_length_deltas: list[int] = []
    right_length_deltas: list[int] = []
    mapped = 0
    for anchor_index, anchor in enumerate(metadata):
        key = (
            _length_bin(anchor["left_length"]),
            _length_bin(anchor["right_length"]),
        )
        pool = pools[key]
        start = _selection_seed(
            seed,
            "anchor",
            retrieval_records[anchor_index]["example_id"],
        ) % len(pool)
        ordered = pool[start:] + pool[:start]
        admissible: list[int] = []
        local_rejections: Counter[str] = Counter()
        for candidate_index in ordered:
            candidate = metadata[candidate_index]
            if candidate_index == anchor_index:
                local_rejections["self"] += 1
                continue
            if anchor["exact"] & candidate["exact"]:
                local_rejections["exact_endpoint_overlap"] += 1
                continue
            if anchor["clusters"] & candidate["clusters"]:
                local_rejections["source_cluster_overlap"] += 1
                continue
            if anchor["group"] == candidate["group"]:
                local_rejections["positive_group_overlap"] += 1
                continue
            if (
                _pair(anchor["left_exact"], candidate["right_exact"])
                in known_exact_pairs
                or _pair(candidate["left_exact"], anchor["right_exact"])
                in known_exact_pairs
            ):
                local_rejections["known_exact_positive_swap"] += 1
                continue
            if (
                _pair(anchor["left_cluster"], candidate["right_cluster"])
                in known_cluster_pairs
                or _pair(candidate["left_cluster"], anchor["right_cluster"])
                in known_cluster_pairs
            ):
                local_rejections["known_cluster_positive_swap"] += 1
                continue
            admissible.append(candidate_index)
        rejection_counts.update(local_rejections)
        admissible_counts.append(len(admissible))
        if not admissible:
            rejection_counts["unmatched_anchor"] += 1
            continue
        candidate_index = admissible[0]
        candidate = metadata[candidate_index]
        mapped += 1
        candidate_use[candidate_index] += 1
        left_length_deltas.append(
            abs(anchor["left_length"] - candidate["left_length"])
        )
        right_length_deltas.append(
            abs(anchor["right_length"] - candidate["right_length"])
        )
        mapping_digest.update(
            (
                f"{retrieval_records[anchor_index]['pilot_record_index']}\t"
                f"{retrieval_records[candidate_index]['pilot_record_index']}\n"
            ).encode("ascii")
        )

    def quantiles(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"maximum": None, "median": None, "p95": None}
        ordered = sorted(values)
        return {
            "maximum": ordered[-1],
            "median": statistics.median(ordered),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        }

    return {
        "admissible_candidates_per_anchor": {
            "maximum": max(admissible_counts),
            "median": statistics.median(admissible_counts),
            "minimum": min(admissible_counts),
        },
        "candidate_reuse": {
            "maximum": max(candidate_use.values(), default=0),
            "unique_candidates": len(candidate_use),
        },
        "counts": {
            "mapped_anchors": mapped,
            "retrieval_records": len(retrieval_records),
            "unmatched_anchors": len(retrieval_records) - mapped,
        },
        "coverage": mapped / len(retrieval_records),
        "length_matching": {
            "definition": "orientation-preserving joint quarter-octave bins floor(4*log2(length))",
            "left_absolute_delta": quantiles(left_length_deltas),
            "right_absolute_delta": quantiles(right_length_deltas),
        },
        "mapping_sha256": mapping_digest.hexdigest(),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "seed": seed,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--expected-manifest-receipt-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--minimum-coverage", type=float, default=0.995)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if (
        arguments.manifest.is_symlink()
        or arguments.manifest_receipt.is_symlink()
        or not arguments.manifest.is_file()
        or not arguments.manifest_receipt.is_file()
        or sha256_file(arguments.manifest) != arguments.expected_manifest_sha256
        or sha256_file(arguments.manifest_receipt)
        != arguments.expected_manifest_receipt_sha256
    ):
        raise RuntimeError("Counterfactual input authentication failed")
    if arguments.receipt.exists():
        raise FileExistsError("Counterfactual receipt is create-only")
    if not math.isfinite(arguments.minimum_coverage) or not (
        0.0 < arguments.minimum_coverage <= 1.0
    ):
        raise ValueError("minimum coverage must be in (0, 1]")
    source_receipt = json.loads(
        arguments.manifest_receipt.read_text(encoding="utf-8")
    )
    if (
        source_receipt.get("status") != "PASS_STAGE_A1_PILOT_SOURCE_ADMISSION"
        or source_receipt.get("test_access") is not False
        or source_receipt.get("output", {}).get("sha256")
        != arguments.expected_manifest_sha256
    ):
        raise RuntimeError("Counterfactual source receipt differs")
    records = [json.loads(line) for line in zstd_lines(arguments.manifest)]
    materialization = materialize_counterfactuals(records, seed=arguments.seed)
    passed = materialization["coverage"] >= arguments.minimum_coverage
    receipt = {
        "authorization": {
            "development_evaluation": False,
            "gpu_training": False,
            "production_training": False,
            "test_access": False,
        },
        "claim_boundary": "Train-only deterministic counterfactual candidate materialization. Exclusions are evidence-aware controls, not proof of biological noninteraction. PPIRef endpoint clusters remain exact-sequence fallbacks, so a production cluster-aware gate is still required.",
        "format_version": FORMAT_VERSION,
        "inputs": {
            "manifest_receipt_sha256": arguments.expected_manifest_receipt_sha256,
            "manifest_sha256": arguments.expected_manifest_sha256,
            "records": len(records),
        },
        "materialization": materialization,
        "minimum_coverage": arguments.minimum_coverage,
        "status": (
            "PASS_TRAIN_ONLY_MATERIALIZATION"
            if passed
            else "FAIL_TRAIN_ONLY_MATERIALIZATION_COVERAGE"
        ),
        "test_access": False,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        arguments.receipt,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    if not passed:
        raise RuntimeError("Counterfactual coverage gate failed")


if __name__ == "__main__":
    main()
