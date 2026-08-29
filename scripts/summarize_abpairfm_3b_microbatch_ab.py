#!/usr/bin/env python3
"""Freeze a same-allocation ABPairFM 3B HSDP microbatch decision."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-throughput-speedup", type=float, default=1.1)
    parser.add_argument("--maximum-p95-to-median", type=float, default=1.2)
    parser.add_argument("--maximum-memory-fraction", type=float, default=0.92)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Receipt is not a JSON object: {path}")
    return value


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
        raise FileExistsError(f"Refusing to overwrite microbatch receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _nearest_rank_p95(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Step timings must be finite and positive")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _validate(receipt: dict[str, Any]) -> None:
    if receipt.get("format_version") != "abpairfm_3b_fsdp2_systems_smoke_v1":
        raise ValueError("Unexpected systems receipt format")
    if receipt.get("status") != "PASS" or receipt.get("synthetic_data") is not True:
        raise ValueError("Input systems evidence did not pass")
    if receipt.get("world_size") != 8:
        raise ValueError("Microbatch A/B requires eight ranks")
    if receipt.get("fsdp2", {}).get("mesh_shape_replicate_by_shard") != [2, 4]:
        raise ValueError("Microbatch A/B requires a 2x4 HSDP mesh")
    if receipt.get("fsdp2", {}).get("reshard_after_forward") is not False:
        raise ValueError("Microbatch A/B requires keep-gathered FSDP2")
    test_access = receipt.get("test_access", {})
    if test_access.get("asep_test_accessed") is not False:
        raise ValueError("AsEP test boundary was not sealed")
    if test_access.get("sabdab2_heldout_accessed") is not False:
        raise ValueError("SAbDab2 test boundary was not sealed")
    ranks = receipt.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 8:
        raise ValueError("Rank evidence is incomplete")


def summarize(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_throughput_speedup: float,
    maximum_p95_to_median: float,
    maximum_memory_fraction: float,
) -> dict[str, Any]:
    _validate(baseline)
    _validate(candidate)
    if baseline.get("microbatch_pairs_per_rank", 1) != 1:
        raise ValueError("Baseline must contain one pair per rank")
    candidate_pairs = candidate.get("microbatch_pairs_per_rank")
    if not isinstance(candidate_pairs, int) or candidate_pairs <= 1:
        raise ValueError("Candidate must contain more than one pair per rank")
    for field in (
        "architecture",
        "initialization",
        "precision",
        "sequence_tokens_per_rank",
        "torch",
    ):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"Microbatch A/B differs in frozen field: {field}")
    for field in ("measured_step_count", "warmup_step_count"):
        if baseline["throughput"].get(field) != candidate["throughput"].get(field):
            raise ValueError(f"Microbatch A/B differs in {field}")
    baseline_hosts = {rank["hostname"] for rank in baseline["ranks"]}
    candidate_hosts = {rank["hostname"] for rank in candidate["ranks"]}
    if len(baseline_hosts) != 1 or baseline_hosts != candidate_hosts:
        raise ValueError("Receipts are not a same-node comparison")

    baseline_throughput = float(
        baseline["throughput"]["residue_tokens_per_second_mean"]
    )
    candidate_throughput = float(
        candidate["throughput"]["residue_tokens_per_second_mean"]
    )
    if baseline_throughput <= 0 or candidate_throughput <= 0:
        raise ValueError("Throughput must be positive")
    speedup = candidate_throughput / baseline_throughput
    candidate_steps = [
        float(value)
        for value in candidate["throughput"]["step_seconds_max_by_step"]
    ]
    p95_to_median = _nearest_rank_p95(candidate_steps) / statistics.median(
        candidate_steps
    )
    memory_fractions = []
    for rank in candidate["ranks"]:
        total = int(rank["total_memory_bytes"])
        allocated = int(rank["max_memory_allocated_bytes"])
        if total <= 0 or allocated < 0:
            raise ValueError("Invalid GPU memory evidence")
        memory_fractions.append(allocated / total)
    peak_memory_fraction = max(memory_fractions)
    gates = {
        "memory_fraction": peak_memory_fraction <= maximum_memory_fraction,
        "step_time_stability": p95_to_median <= maximum_p95_to_median,
        "throughput_speedup": speedup >= minimum_throughput_speedup,
    }
    promoted = all(gates.values())
    return {
        "claim_boundary": "Same-node synthetic full-model microbatch evidence only; it does not authorize biological training, real-data batch sizing, or SOTA claims.",
        "comparison": {
            "baseline_microbatch_pairs_per_rank": 1,
            "baseline_residue_tokens_per_second": baseline_throughput,
            "candidate_microbatch_pairs_per_rank": candidate_pairs,
            "candidate_p95_to_median_step_time": p95_to_median,
            "candidate_peak_allocated_memory_fraction": peak_memory_fraction,
            "candidate_residue_tokens_per_second": candidate_throughput,
            "candidate_throughput_speedup": speedup,
            "hostname": next(iter(baseline_hosts)),
        },
        "decision": (
            f"PROMOTE_MICROBATCH_{candidate_pairs}"
            if promoted
            else f"REJECT_MICROBATCH_{candidate_pairs}"
        ),
        "format_version": "abpairfm_3b_microbatch_ab_v1",
        "gates": gates,
        "status": "PASS",
        "test_access": {
            "asep_test_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "thresholds": {
            "maximum_memory_fraction": maximum_memory_fraction,
            "maximum_p95_to_median_step_time": maximum_p95_to_median,
            "minimum_throughput_speedup": minimum_throughput_speedup,
        },
    }


def main() -> None:
    arguments = _arguments()
    receipt = summarize(
        _load(arguments.baseline.resolve()),
        _load(arguments.candidate.resolve()),
        minimum_throughput_speedup=arguments.minimum_throughput_speedup,
        maximum_p95_to_median=arguments.maximum_p95_to_median,
        maximum_memory_fraction=arguments.maximum_memory_fraction,
    )
    _publish_create_only(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
