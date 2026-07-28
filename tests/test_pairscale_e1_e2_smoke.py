"""Tests for the real-cache E1/E2 GPU smoke evidence helpers."""

from __future__ import annotations

import importlib.util
import inspect

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def test_exact_replay_helper_checks_nested_tensor_and_optimizer_state() -> None:
    import torch

    from smoke_pairscale_e1_e2 import _assert_exact, _cpu_clone

    reference = {
        "model": {"weight": torch.tensor([[1.0, 2.0]])},
        "optimizer": [{"step": torch.tensor(2), "lr": 0.001}],
    }
    clone = _cpu_clone(reference)
    _assert_exact(reference, clone)
    clone["model"]["weight"][0, 0] = 9.0
    with pytest.raises(RuntimeError, match="Tensor replay differs"):
        _assert_exact(reference, clone)


def test_smoke_loads_only_a_canonical_train_subset() -> None:
    from smoke_pairscale_e1_e2 import main

    source = inspect.getsource(main)
    assert "train_indices, _ = split_dataset_indices" in source
    assert "sorted(" in source
    assert ".instance_id" in source
    assert "_load_examples(dataset, smoke_indices)" in source
    assert "holdout_examples" not in source
    assert '"development_holdout_tensor_accessed": False' in source
    assert '"new_final_test_accessed": False' in source


def test_smoke_requires_exact_model_and_optimizer_resume() -> None:
    from smoke_pairscale_e1_e2 import main

    source = inspect.getsource(main)
    assert "restore_checkpoint_generation" in source
    assert "expected_receipt_sha256=receipt_sha256" in source
    assert "_assert_exact(reference_model_state" in source
    assert "reference_optimizer_state" in source
    assert '"resume_model_state_exact": True' in source
    assert '"resume_optimizer_state_exact": True' in source
