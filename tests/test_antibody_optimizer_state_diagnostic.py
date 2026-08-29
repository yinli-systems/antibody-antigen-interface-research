"""Test exact tensor-delta accounting for optimizer diagnostics."""

from __future__ import annotations

import torch

import diagnose_antibody_optimizer_state as diagnostic


def test_tensor_delta_reports_exact_and_perturbed_float_state() -> None:
    left = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32)
    exact = diagnostic._tensor_delta(left, left.clone())
    assert exact["equal"] is True
    assert exact["different_elements"] == 0
    assert exact["maximum_absolute_difference"] == 0.0

    right = torch.tensor([0.0, 1.0 + 2.0**-20, -2.0], dtype=torch.float32)
    changed = diagnostic._tensor_delta(left, right)
    assert changed["equal"] is False
    assert changed["different_elements"] == 1
    assert changed["maximum_absolute_difference"] > 0.0
    assert changed["maximum_ulp_distance"] == 8


def test_tensor_delta_reports_integer_state() -> None:
    delta = diagnostic._tensor_delta(
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )
    assert delta == {
        "different_elements": 1,
        "dtype": "torch.int64",
        "equal": False,
        "shape": [1],
    }
