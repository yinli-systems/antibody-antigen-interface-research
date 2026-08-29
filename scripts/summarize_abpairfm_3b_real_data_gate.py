#!/usr/bin/env python3
"""Validate and condense a full ABPairFM 3B real-data systems receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(receipt: dict[str, object], *, job_id: int) -> dict[str, object]:
    if (
        receipt.get("format_version")
        not in {"abpairfm_3b_real_data_gate_v1", "abpairfm_3b_real_data_gate_v2"}
        or receipt.get("status") != "PASS"
        or receipt.get("world_size") != 8
        or receipt.get("precision") != "bfloat16_compute_fp32_optimizer"
    ):
        raise ValueError("Real-data gate receipt identity differs")
    test_access = receipt.get("test_access")
    if not isinstance(test_access, dict) or any(test_access.values()):
        raise ValueError("Real-data gate crossed a test boundary")
    authorization = receipt.get("authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("production_pretraining") is not False
        or authorization.get("real_data_gpu_gate_passed") is not True
    ):
        raise ValueError("Real-data gate authorization differs")

    data = receipt["data"]
    optimization = receipt["optimization"]
    loss = receipt["loss"]
    stability = receipt["stability"]
    throughput = receipt["throughput"]
    ranks = receipt["ranks"]
    if not all(isinstance(value, dict) for value in (data, optimization, loss, stability, throughput)):
        raise ValueError("Real-data gate sections are malformed")
    if not isinstance(ranks, list) or len(ranks) != 8:
        raise ValueError("Real-data gate rank inventory differs")
    microbatch_pairs = int(optimization.get("microbatch_pairs_per_rank", 0))
    expected_records = 200 * 2 * 8 * microbatch_pairs
    if (
        data.get("manifest_records_consumed") != expected_records
        or data.get("manifest_records_expected") != expected_records
        or data.get("duplicate_manifest_records") != 0
        or data.get("manifest_skipped_records") != 0
        or data.get("development_decontaminated") is not True
        or optimization.get("updates") != 200
        or microbatch_pairs not in {7, 8}
        or optimization.get("accumulation_steps") != 2
    ):
        raise ValueError("Real-data gate data cursor or optimization differs")
    for rank_number, rank in enumerate(ranks):
        if not isinstance(rank, dict):
            raise ValueError("Real-data gate rank record is malformed")
        if (
            rank.get("rank") != rank_number
            or rank.get("device_name") != "NVIDIA GeForce RTX 4090"
            or rank.get("local_manifest_records") != 200 * 2 * microbatch_pairs
            or not isinstance(rank.get("updates"), list)
            or len(rank["updates"]) != 200
        ):
            raise ValueError("Real-data gate per-rank evidence differs")

    first_loss = float(loss["first_20_update_mean"])
    final_loss = float(loss["final_20_update_mean"])
    p95_ratio = float(stability["p95_to_median_update_time"])
    median_throughput = float(throughput["residue_tokens_per_second_median"])
    maximum_memory_fraction = max(
        int(rank["max_memory_allocated_bytes"]) / int(rank["total_memory_bytes"])
        for rank in ranks
    )
    finite = all(
        math.isfinite(value)
        for value in (first_loss, final_loss, p95_ratio, median_throughput, maximum_memory_fraction)
    )
    gates = {
        "all_summary_values_finite": finite,
        "final_20_loss_not_above_first_20_by_more_than_five_percent": final_loss
        <= first_loss * 1.05,
        "maximum_allocated_memory_fraction_at_most_0_92": maximum_memory_fraction
        <= 0.92,
        "p95_to_median_update_time_at_most_1_20": p95_ratio <= 1.2,
    }
    passed = all(gates.values())
    return {
        "authorization": {
            "production_pretraining": False,
            "quality_per_gpu_hour_gate": passed,
            "test_access": False,
        },
        "claim_boundary": "Full-2.839B-model, eight-RTX-4090 train-only numerical and systems gate on a deterministic development-decontaminated STRING slice. Validation quality, biological improvement, production training, and SOTA claims remain pending.",
        "decision": (
            "PROMOTE_TO_QUALITY_PER_GPU_HOUR_GATE"
            if passed
            else "FAIL_CLOSED_REAL_DATA_GATE"
        ),
        "format_version": "abpairfm_3b_real_data_gate_summary_v1",
        "gates": {**gates, "passed": passed},
        "job_id": job_id,
        "loss": {
            "final_20_update_mean": final_loss,
            "first_20_update_mean": first_loss,
            "final_to_first_ratio": final_loss / first_loss,
        },
        "memory": {
            "maximum_allocated_fraction": maximum_memory_fraction,
        },
        "optimization": {
            "accumulation_steps": 2,
            "manifest_records_consumed": expected_records,
            "microbatch_pairs_per_rank": microbatch_pairs,
            "updates": 200,
            "world_size": 8,
        },
        "stability": {
            "p95_to_median_update_time": p95_ratio,
        },
        "test_access": {
            "asep_test_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "throughput": {
            "consumed_residue_tokens": int(throughput["consumed_residue_tokens"]),
            "five_billion_token_wall_clock_days_excluding_overhead": 5_000_000_000
            / median_throughput
            / 86_400,
            "residue_tokens_per_second_mean": float(
                throughput["residue_tokens_per_second_mean"]
            ),
            "residue_tokens_per_second_median": median_throughput,
        },
    }


def publish_create_only(path: Path, value: object) -> None:
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
        raise FileExistsError(f"Refusing to overwrite real-data summary: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt_path = arguments.receipt.resolve()
    value = summarize(json.loads(receipt_path.read_text(encoding="utf-8")), job_id=arguments.job_id)
    value["receipt_sha256"] = sha256_file(receipt_path)
    publish_create_only(arguments.output.resolve(), value)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
