"""Validate the frozen BF16 cross-world and warmed-profile receipts."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/pretraining"


def _load(name: str) -> dict[str, object]:
    """Load one checked-in machine receipt without mutating it."""
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_bf16_cross_world_recovery_passes_in_both_directions() -> None:
    one_to_eight = _load("frontier_fsdp2_cross_world_1to8_bf16_v1.json")
    eight_to_one = _load("frontier_fsdp2_cross_world_8to1_bf16_v1.json")
    expected_directions = ((one_to_eight, 1, 8), (eight_to_one, 8, 1))
    for receipt, source_world_size, load_world_size in expected_directions:
        assert receipt["format"] == "frontier_fsdp2_cross_world_reshard_result_v1"
        assert receipt["status"] == "passed"
        assert receipt["mode"] == "load"
        assert receipt["precision"] == "bfloat16"
        assert receipt["source_world_size"] == source_world_size
        assert receipt["load_world_size"] == load_world_size
        assert receipt["global_batch_size"] == 8
        assert receipt["absolute_tolerance"] == 0.005
        assert receipt["relative_tolerance"] == 0.02
        assert receipt["outer_or_test_access"] is False
        assert receipt["biological_training_authorized"] is False

        checkpoint = receipt["checkpoint_state_difference"]
        assert checkpoint == {
            "element_count": 44531,
            "max_absolute": 0.0,
            "max_relative": 0.0,
            "max_scaled_error": 0.0,
            "tensor_count": 81,
        }
        continuation = receipt["continuation_state_difference"]
        assert continuation["tensor_count"] == 81
        assert continuation["element_count"] == 44531
        assert math.isfinite(continuation["max_scaled_error"])
        assert continuation["max_scaled_error"] <= 1.0

        expected_loss = receipt["expected_continuation_global_loss"]
        allowed_loss = 0.005 + 0.02 * abs(expected_loss)
        assert receipt["continuation_loss_absolute_difference"] <= allowed_loss

    assert one_to_eight["source_sha256"] == eight_to_one["source_sha256"]


def test_world_eight_profile_uses_the_frozen_warmed_window() -> None:
    receipt = _load("frontier_fsdp2_world8_bf16_profile_v1.json")
    assert receipt["format"] == "frontier_fsdp2_warmed_profile_result_v1"
    assert receipt["status"] == "passed"
    assert receipt["precision"] == "bfloat16"
    assert receipt["world_size"] == 8
    assert receipt["warmup_steps"] == 50
    assert receipt["measured_steps"] == 500
    assert receipt["batch_size_per_rank"] == 8
    assert receipt["global_examples_per_step"] == 64
    assert receipt["global_residue_tokens_per_step"] == 1024
    assert receipt["outer_or_test_access"] is False
    assert receipt["biological_training_authorized"] is False

    rank_receipts = receipt["rank_receipts"]
    assert [rank["rank"] for rank in rank_receipts] == list(range(8))
    assert all(len(rank["step_seconds"]) == 500 for rank in rank_receipts)

    step = receipt["step_seconds"]
    assert step["minimum"] <= step["p50"] <= step["p95"] <= step["p99"]
    assert step["p99"] <= step["maximum"]
    tokens = receipt["global_residue_tokens_per_second"]
    assert math.isclose(tokens["maximum"], 1024 / step["minimum"], rel_tol=1e-12)
    assert math.isclose(tokens["minimum"], 1024 / step["maximum"], rel_tol=1e-12)
    assert math.isclose(tokens["p50"], 1024 / step["p50"], rel_tol=1e-7)

    checkpoint = receipt["checkpoint"]
    assert checkpoint["receipt_content_sha256"] == checkpoint[
        "loaded_receipt_content_sha256"
    ]
    assert checkpoint["restore_state_difference"]["max_scaled_error"] == 0.0
