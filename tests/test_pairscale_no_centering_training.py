"""Tests for the fixed PAIRScale no-centering ablation."""

from __future__ import annotations

import inspect

import pytest

from interaction_first.pairscale_no_centering_training import (
    FIXED_STOP_EPOCH,
    SCHEDULER_HORIZON_EPOCHS,
    locked_no_centering_config,
    raw_topk_partner_ranking_loss,
    run_no_centering_ablation,
    train_one_epoch_without_centering,
)
from interaction_first.pairscale_overfit import PairscaleOverfitConfig


def test_locked_config_matches_reference_e400_trajectory() -> None:
    reference = PairscaleOverfitConfig(
        expected_complex_count=192,
        epochs=400,
    ).canonical_payload()
    locked = locked_no_centering_config().canonical_payload()
    assert locked == reference
    assert FIXED_STOP_EPOCH == 260
    assert SCHEDULER_HORIZON_EPOCHS == 400


def test_training_loss_source_never_calls_centering() -> None:
    loss_source = inspect.getsource(raw_topk_partner_ranking_loss)
    epoch_source = inspect.getsource(train_one_epoch_without_centering)
    assert "double_center" not in loss_source
    assert "double_center" not in epoch_source
    assert "raw_topk_partner_ranking_loss" in epoch_source


def test_holdout_is_loaded_once_after_fixed_checkpoint() -> None:
    source = inspect.getsource(run_no_centering_ablation)
    checkpoint_check = source.index(
        'if int(last_receipt["epoch"]) != FIXED_STOP_EPOCH:'
    )
    holdout_load = source.index(
        "holdout_examples = _load_examples(dataset, holdout_indices)"
    )
    access_receipt = source.index('"HOLDOUT_ACCESS_RECEIPT.json"')
    evaluation = source.index('"HOLDOUT_EVALUATION.json"')
    assert checkpoint_check < holdout_load < access_receipt < evaluation
    assert source.count(
        "holdout_examples = _load_examples(dataset, holdout_indices)"
    ) == 1
    assert "break" not in source


def test_raw_topk_loss_matches_closed_form() -> None:
    torch = pytest.importorskip("torch")
    matched = {"contact_logits": torch.tensor([[[1.0, 2.0]]])}
    decoy = {"contact_logits": torch.tensor([[[0.5, 0.0]]])}
    observed = raw_topk_partner_ranking_loss(
        matched,
        decoy,
        positive_pair_count=1,
        margin=0.1,
        temperature=1.0,
    )
    expected = torch.nn.functional.softplus(torch.tensor(0.1 - 2.0 + 0.5))
    assert torch.equal(observed, expected)


@pytest.mark.parametrize("name", ("margin", "temperature"))
def test_raw_topk_loss_rejects_nonpositive_hyperparameters(name: str) -> None:
    torch = pytest.importorskip("torch")
    outputs = {"contact_logits": torch.ones((1, 1, 1))}
    arguments = {"margin": 0.1, "temperature": 1.0}
    arguments[name] = 0.0
    with pytest.raises(ValueError, match=name):
        raw_topk_partner_ranking_loss(
            outputs,
            outputs,
            positive_pair_count=1,
            **arguments,
        )
