#!/usr/bin/env python3
"""Attribute long-horizon numerical divergence in the paired DDP gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


RUNS = (
    "safe_eager_r1",
    "fp32_compiled_r1",
    "bf16_selected_r1",
    "bf16_selected_r2",
    "fp32_compiled_r2",
    "safe_eager_r2",
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bindings(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, item = value.partition("=")
        if not separator or name not in RUNS or name in result or not item:
            raise RuntimeError(f"Invalid receipt binding: {value}")
        result[name] = Path(item)
    if tuple(result) != RUNS:
        raise RuntimeError("Receipt bindings must use the frozen run order")
    return result


def _relative_difference(reference: float, candidate: float) -> float:
    return abs(candidate - reference) / max(abs(reference), 1e-12)


def _comparison(
    reference: dict[str, object],
    candidate: dict[str, object],
    thresholds: dict[str, object],
) -> dict[str, object]:
    if len(reference["trace"]) != 500 or len(candidate["trace"]) != 500:
        raise RuntimeError("Attribution requires two 500-update receipts")

    def maximum_loss(items: int) -> float:
        return max(
            abs(float(left["loss"]) - float(right["loss"]))
            for left, right in zip(
                reference["trace"][:items], candidate["trace"][:items]
            )
        )

    def maximum_gradient(items: int) -> float:
        return max(
            _relative_difference(
                float(left["gradient_norm"]), float(right["gradient_norm"])
            )
            for left, right in zip(
                reference["trace"][:items], candidate["trace"][:items]
            )
        )

    moment_differences = {
        key: _relative_difference(
            float(reference["parameter_moments"][key]),
            float(candidate["parameter_moments"][key]),
        )
        for key in ("absolute_maximum", "square_sum", "sum")
    }
    early_loss = maximum_loss(20)
    early_gradient = maximum_gradient(20)
    early_passed = (
        early_loss <= thresholds["maximum_final_loss_absolute_difference"]
        and early_gradient
        <= thresholds["maximum_gradient_norm_relative_difference"]
    )
    reference_tail = [float(item["loss"]) for item in reference["trace"][-50:]]
    candidate_tail = [float(item["loss"]) for item in candidate["trace"][-50:]]
    return {
        "early_20_updates": {
            "maximum_gradient_norm_relative_difference": early_gradient,
            "maximum_loss_absolute_difference": early_loss,
            "passed_trace_thresholds": early_passed,
        },
        "final_update": {
            "gradient_norm_relative_difference": _relative_difference(
                float(reference["trace"][-1]["gradient_norm"]),
                float(candidate["trace"][-1]["gradient_norm"]),
            ),
            "loss_absolute_difference": abs(
                float(reference["trace"][-1]["loss"])
                - float(candidate["trace"][-1]["loss"])
            ),
        },
        "full_500_updates": {
            "maximum_gradient_norm_relative_difference": maximum_gradient(500),
            "maximum_loss_absolute_difference": maximum_loss(500),
        },
        "parameter_moment_relative_differences": moment_differences,
        "tail_50_updates": {
            "mean_loss_absolute_difference": abs(
                statistics.fmean(reference_tail)
                - statistics.fmean(candidate_tail)
            ),
            "median_loss_absolute_difference": abs(
                statistics.median(reference_tail)
                - statistics.median(candidate_tail)
            ),
        },
    }


def _scientifically_exact(
    left: dict[str, object], right: dict[str, object]
) -> bool:
    keys = tuple(key for key in left["trace"][0] if key != "seconds")
    return (
        all(
            all(left_item[key] == right_item[key] for key in keys)
            for left_item, right_item in zip(left["trace"], right["trace"])
        )
        and left["parameter_moments"] == right["parameter_moments"]
        and left["final_model_sha256"] == right["final_model_sha256"]
        and left["final_optimizer_sha256"] == right["final_optimizer_sha256"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", default=[])
    parser.add_argument("--failed-gate-result", type=Path, required=True)
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = _bindings(args.receipt)
    receipts = {name: _load(path) for name, path in paths.items()}
    failed_gate = _load(args.failed_gate_result)
    protocol = _load(args.efficiency_protocol)
    if (
        failed_gate.get("format_version")
        != "antibody_ddp_communication_paired_gate_result_v1"
        or failed_gate.get("gate_passed") is not False
        or failed_gate.get("gates", {}).get("numerical_correctness") is not False
    ):
        raise RuntimeError("Attribution input is not the preserved numerical failure")
    if protocol.get("format_version") != "antibody_efficiency_gate_v1":
        raise RuntimeError("Unexpected efficiency protocol")
    if len({receipt["membership_sha256"] for receipt in receipts.values()}) != 1:
        raise RuntimeError("Attribution receipts do not share membership")
    comparisons = {}
    for replicate in ("r1", "r2"):
        eager = receipts[f"safe_eager_{replicate}"]
        compiled = receipts[f"fp32_compiled_{replicate}"]
        bf16 = receipts[f"bf16_selected_{replicate}"]
        comparisons[f"compiler_{replicate}"] = _comparison(
            eager, compiled, protocol["correctness_gate"]
        )
        comparisons[f"communication_increment_{replicate}"] = _comparison(
            compiled, bf16, protocol["correctness_gate"]
        )
        comparisons[f"combined_{replicate}"] = _comparison(
            eager, bf16, protocol["correctness_gate"]
        )
    replicate_exact = {
        mode: _scientifically_exact(
            receipts[f"{mode}_r1"], receipts[f"{mode}_r2"]
        )
        for mode in ("safe_eager", "fp32_compiled", "bf16_selected")
    }
    early_passed = all(
        comparison["early_20_updates"]["passed_trace_thresholds"]
        for comparison in comparisons.values()
    )
    long_horizon_failed = any(
        comparison["full_500_updates"][
            "maximum_gradient_norm_relative_difference"
        ]
        > protocol["correctness_gate"][
            "maximum_gradient_norm_relative_difference"
        ]
        or comparison["full_500_updates"]["maximum_loss_absolute_difference"]
        > protocol["correctness_gate"][
            "maximum_final_loss_absolute_difference"
        ]
        for comparison in comparisons.values()
    )
    payload = {
        "authorization": {
            "exact_recovery": False,
            "pilot_16_gpu": False,
            "pilot_32_gpu": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Post-failure numerical attribution on repeated-shard training; "
            "no gate relaxation, scaling, production, biological, or "
            "state-of-the-art claim"
        ),
        "comparisons": comparisons,
        "conclusion": (
            "early_numerical_correctness_passes_but_long_horizon_"
            "trajectory_equivalence_fails"
        ),
        "early_20_update_thresholds_passed": early_passed,
        "failed_gate_result_sha256": _sha256(args.failed_gate_result),
        "format_version": "antibody_ddp_numerical_divergence_attribution_v1",
        "input_receipt_sha256": {
            name: _sha256(path) for name, path in paths.items()
        },
        "long_horizon_thresholds_failed": long_horizon_failed,
        "replicate_exact": replicate_exact,
        "source": {
            "auditor_sha256": _sha256(Path(__file__)),
            "efficiency_protocol_sha256": _sha256(args.efficiency_protocol),
        },
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
