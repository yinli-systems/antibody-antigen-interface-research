#!/usr/bin/env python3
"""Apply a fail-closed train-only learning gate to the Stage-A1 geometry canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active_delta(loss: dict[str, object]) -> tuple[float, float, float]:
    first = float(loss["first_10_active_update_mean"])
    final = float(loss["final_10_active_update_mean"])
    if not math.isfinite(first) or not math.isfinite(final) or first <= 0.0:
        raise RuntimeError("Stage-A1 active loss summary is nonfinite")
    return first, final, (first - final) / first


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists():
        raise FileExistsError("Stage-A1 geometry canary audit is create-only")
    result = json.loads(arguments.result.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS_STAGE_A1_EFFICIENCY_GATE_CHECKPOINT_VERIFIED"
        or result.get("test_access") is not False
        or result.get("source_sha256") != arguments.expected_source_sha256
        or result.get("checkpoint", {}).get("content_sha256")
        != result.get("checkpoint", {}).get("restore_content_sha256")
    ):
        raise RuntimeError("Stage-A1 geometry canary result authority differs")
    receipt_protocol = result.get("checkpoint", {}).get("provenance_hashes", {}).get(
        "protocol"
    )
    if receipt_protocol not in {None, arguments.expected_protocol_sha256}:
        raise RuntimeError("Stage-A1 geometry canary protocol provenance differs")

    losses = result["loss"]
    contact_first, contact_final, contact_reduction = _active_delta(
        losses["contact_loss"]
    )
    distance_first, distance_final, distance_reduction = _active_delta(
        losses["distance_loss"]
    )
    if (
        losses["contact_loss"].get("active_update_count") != 20
        or losses["distance_loss"].get("active_update_count") != 20
    ):
        raise RuntimeError("Stage-A1 geometry active-update count differs")
    gates = {
        "contact_first_to_final_relative_reduction_at_least_0p02": (
            contact_reduction >= 0.02
        ),
        "distance_final_below_uniform_8_bin_cross_entropy": (
            distance_final < math.log(8.0)
        ),
        "distance_first_to_final_relative_reduction_at_least_0p02": (
            distance_reduction >= 0.02
        ),
    }
    passed = all(gates.values())
    payload = {
        "authorization": {
            "development_evaluation": passed,
            "test_access": False,
        },
        "format_version": "abpairfm_stage_a1_geometry_canary_audit_v1",
        "gates": gates,
        "loss": {
            "contact": {
                "final_10_active_update_mean": contact_final,
                "first_10_active_update_mean": contact_first,
                "relative_reduction": contact_reduction,
            },
            "distance": {
                "final_10_active_update_mean": distance_final,
                "first_10_active_update_mean": distance_first,
                "relative_reduction": distance_reduction,
                "uniform_8_bin_cross_entropy": math.log(8.0),
            },
        },
        "result_sha256": _sha256(arguments.result),
        "status": (
            "PASS_STAGE_A1_GEOMETRY_CANARY_LEARNING_GATE"
            if passed
            else "FAIL_STAGE_A1_GEOMETRY_CANARY_LEARNING_GATE"
        ),
        "test_access": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
