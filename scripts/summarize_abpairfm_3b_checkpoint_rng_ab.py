#!/usr/bin/env python3
"""Freeze the same-node ABPairFM 3B checkpoint RNG-stashing A/B decision."""

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
    parser.add_argument("--minimum-throughput-speedup", type=float, default=1.02)
    parser.add_argument("--maximum-p95-to-median", type=float, default=1.2)
    parser.add_argument("--maximum-memory-fraction", type=float, default=0.92)
    parser.add_argument("--maximum-loss-absolute-delta", type=float, default=5e-4)
    parser.add_argument(
        "--maximum-gradient-norm-relative-delta", type=float, default=1e-3
    )
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
        raise FileExistsError(f"Refusing to overwrite checkpoint-RNG receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _nearest_rank_p95(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Step timings must be finite and positive")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _validate(receipt: dict[str, Any], *, preserve_rng_state: bool) -> None:
    if receipt.get("format_version") != "abpairfm_3b_fsdp2_systems_smoke_v1":
        raise ValueError("Unexpected systems receipt format")
    if receipt.get("status") != "PASS" or receipt.get("synthetic_data") is not True:
        raise ValueError("Input systems evidence did not pass")
    if receipt.get("world_size") != 8:
        raise ValueError("Checkpoint-RNG A/B requires eight ranks")
    if receipt.get("microbatch_pairs_per_rank") != 7:
        raise ValueError("Checkpoint-RNG A/B requires frozen microbatch seven")
    if receipt.get("sequence_tokens_per_rank") != 1024:
        raise ValueError("Checkpoint-RNG A/B requires 1024-token pairs")
    if receipt.get("fsdp2") != {
        "mesh_shape_replicate_by_shard": [2, 4],
        "reshard_after_forward": False,
    }:
        raise ValueError("Checkpoint-RNG A/B requires keep-gathered HSDP 2x4")
    architecture = receipt.get("architecture", {})
    if architecture.get("activation_checkpointing") is not True:
        raise ValueError("Activation checkpointing is not enabled")
    if architecture.get("activation_checkpoint_policy") != "full":
        raise ValueError("Checkpoint-RNG A/B requires full per-block checkpointing")
    if architecture.get("activation_checkpoint_preserve_rng_state") is not preserve_rng_state:
        raise ValueError("Checkpoint RNG-preservation setting differs")
    if receipt.get("torch", {}).get("flash_sdpa_forced") is not True:
        raise ValueError("Flash SDPA was not forced")
    if receipt.get("test_access") != {
        "asep_test_accessed": False,
        "sabdab2_heldout_accessed": False,
    }:
        raise ValueError("Held-out test boundary differs")
    ranks = receipt.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 8:
        raise ValueError("Rank evidence is incomplete")


def _without_rng_setting(architecture: dict[str, Any]) -> dict[str, Any]:
    value = dict(architecture)
    value.pop("activation_checkpoint_preserve_rng_state", None)
    return value


def _rank_steps(receipt: dict[str, Any]) -> list[dict[str, float | int]]:
    ranks = sorted(receipt["ranks"], key=lambda rank: int(rank["rank"]))
    return [step for rank in ranks for step in rank["steps"]]


def summarize(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_throughput_speedup: float,
    maximum_p95_to_median: float,
    maximum_memory_fraction: float,
    maximum_loss_absolute_delta: float,
    maximum_gradient_norm_relative_delta: float,
) -> dict[str, Any]:
    _validate(baseline, preserve_rng_state=True)
    _validate(candidate, preserve_rng_state=False)
    if _without_rng_setting(baseline["architecture"]) != _without_rng_setting(
        candidate["architecture"]
    ):
        raise ValueError("Checkpoint-RNG A/B architecture differs beyond RNG setting")
    for field in ("initialization", "precision", "sequence_tokens_per_rank", "torch"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"Checkpoint-RNG A/B differs in frozen field: {field}")
    for field in ("measured_step_count", "warmup_step_count"):
        if baseline["throughput"].get(field) != candidate["throughput"].get(field):
            raise ValueError(f"Checkpoint-RNG A/B differs in {field}")
    baseline_hosts = {rank["hostname"] for rank in baseline["ranks"]}
    candidate_hosts = {rank["hostname"] for rank in candidate["ranks"]}
    if len(baseline_hosts) != 1 or baseline_hosts != candidate_hosts:
        raise ValueError("Receipts are not a same-node comparison")

    baseline_steps = _rank_steps(baseline)
    candidate_steps = _rank_steps(candidate)
    if len(baseline_steps) != len(candidate_steps):
        raise ValueError("Rank step evidence differs")
    loss_deltas: list[float] = []
    gradient_relative_deltas: list[float] = []
    for baseline_step, candidate_step in zip(
        baseline_steps, candidate_steps, strict=True
    ):
        baseline_loss = float(baseline_step["global_mean_loss"])
        candidate_loss = float(candidate_step["global_mean_loss"])
        baseline_gradient = float(baseline_step["gradient_norm"])
        candidate_gradient = float(candidate_step["gradient_norm"])
        values = (baseline_loss, candidate_loss, baseline_gradient, candidate_gradient)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Numerical comparison contains non-finite values")
        loss_deltas.append(abs(candidate_loss - baseline_loss))
        gradient_relative_deltas.append(
            abs(candidate_gradient - baseline_gradient)
            / max(abs(baseline_gradient), 1e-12)
        )

    baseline_throughput = float(
        baseline["throughput"]["residue_tokens_per_second_mean"]
    )
    candidate_throughput = float(
        candidate["throughput"]["residue_tokens_per_second_mean"]
    )
    if baseline_throughput <= 0 or candidate_throughput <= 0:
        raise ValueError("Throughput must be positive")
    speedup = candidate_throughput / baseline_throughput
    candidate_timings = [
        float(value)
        for value in candidate["throughput"]["step_seconds_max_by_step"]
    ]
    p95_to_median = _nearest_rank_p95(candidate_timings) / statistics.median(
        candidate_timings
    )
    memory_fractions = []
    for rank in candidate["ranks"]:
        allocated = int(rank["max_memory_allocated_bytes"])
        total = int(rank["total_memory_bytes"])
        if allocated < 0 or total <= 0:
            raise ValueError("Invalid GPU memory evidence")
        memory_fractions.append(allocated / total)
    peak_memory_fraction = max(memory_fractions)
    maximum_loss_delta = max(loss_deltas, default=0.0)
    maximum_gradient_relative_delta = max(gradient_relative_deltas, default=0.0)
    gates = {
        "gradient_norm_parity": maximum_gradient_relative_delta
        <= maximum_gradient_norm_relative_delta,
        "loss_parity": maximum_loss_delta <= maximum_loss_absolute_delta,
        "memory_fraction": peak_memory_fraction <= maximum_memory_fraction,
        "step_time_stability": p95_to_median <= maximum_p95_to_median,
        "throughput_speedup": speedup >= minimum_throughput_speedup,
    }
    promoted = all(gates.values())
    return {
        "authorization": {
            "production_pretraining": False,
            "real_data_gate": promoted,
            "test_access": False,
        },
        "claim_boundary": "Same-node synthetic full-model systems evidence only. Promotion authorizes one frozen 200-update train-only real-data gate, not production or a quality claim.",
        "comparison": {
            "baseline_preserve_rng_state": True,
            "baseline_residue_tokens_per_second": baseline_throughput,
            "candidate_p95_to_median_step_time": p95_to_median,
            "candidate_peak_allocated_memory_fraction": peak_memory_fraction,
            "candidate_preserve_rng_state": False,
            "candidate_residue_tokens_per_second": candidate_throughput,
            "candidate_throughput_speedup": speedup,
            "hostname": next(iter(baseline_hosts)),
            "maximum_gradient_norm_relative_delta": maximum_gradient_relative_delta,
            "maximum_loss_absolute_delta": maximum_loss_delta,
        },
        "decision": (
            "PROMOTE_NO_RNG_STASH_TO_REAL_DATA_GATE"
            if promoted
            else "REJECT_NO_RNG_STASH"
        ),
        "format_version": "abpairfm_3b_checkpoint_rng_ab_v1",
        "gates": gates,
        "status": "PASS",
        "test_access": {
            "asep_test_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "thresholds": {
            "maximum_gradient_norm_relative_delta": maximum_gradient_norm_relative_delta,
            "maximum_loss_absolute_delta": maximum_loss_absolute_delta,
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
        maximum_loss_absolute_delta=arguments.maximum_loss_absolute_delta,
        maximum_gradient_norm_relative_delta=(
            arguments.maximum_gradient_norm_relative_delta
        ),
    )
    _publish_create_only(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
