"""Tests for the fixed centered multi-decoy PAIRScale runner."""

from __future__ import annotations

import inspect

from interaction_first.pairscale_centered_multidecoy import (
    locked_centered_multidecoy_config,
    train_one_epoch_centered_multidecoy,
)
from interaction_first.pairscale_centered_multidecoy_training import (
    run_centered_multidecoy_repair,
)
from interaction_first.pairscale_no_centering_training import (
    FIXED_STOP_EPOCH,
    SCHEDULER_HORIZON_EPOCHS,
    locked_no_centering_config,
)


def test_locked_config_matches_frozen_reference() -> None:
    assert locked_centered_multidecoy_config().canonical_payload() == (
        locked_no_centering_config().canonical_payload()
    )
    assert FIXED_STOP_EPOCH == 260
    assert SCHEDULER_HORIZON_EPOCHS == 400


def test_training_epoch_uses_centered_multidecoy_objective_only() -> None:
    source = inspect.getsource(train_one_epoch_centered_multidecoy)
    assert "centered_multidecoy_partner_ranking_loss" in source
    assert "raw_top" not in source
    assert "positive_pair_count" not in source


def test_holdout_is_loaded_once_after_immutable_epoch_260_checkpoint() -> None:
    source = inspect.getsource(run_centered_multidecoy_repair)
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


def test_decoy_assignment_is_bound_into_checkpoint_provenance() -> None:
    source = inspect.getsource(run_centered_multidecoy_repair)
    assignment_write = source.index('"DECOY_ASSIGNMENT.json"')
    provenance = source.index('"decoy_assignment_sha256": decoy_assignment_sha256')
    first_checkpoint = source.index("create_checkpoint_generation(")
    assert assignment_write < provenance < first_checkpoint
    assert '"score_or_label_fields_accessed": False' in inspect.getsource(
        __import__(
            "interaction_first.pairscale_centered_multidecoy_training",
            fromlist=["_decoy_assignment_payload"],
        )._decoy_assignment_payload
    )
