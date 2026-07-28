"""Tests for the fixed-horizon PAIRScale E1/E2 controller."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest

from interaction_first.development_contact_dataset import DevelopmentContactExample
from interaction_first.pairscale_e1_e2_training import (
    SCREENING_SEEDS,
    _decoy_assignment_payload,
    e1_e2_source_hashes,
    locked_e1_e2_config,
    run_e1_e2_screen,
)
from interaction_first.pairscale_partner_specificity import (
    select_score_free_bidirectional_decoys,
)


def _examples() -> tuple[DevelopmentContactExample, ...]:
    import torch

    result = []
    for index in range(7):
        antibody_length = 3 + index
        antigen_length = 8 + index
        result.append(
            DevelopmentContactExample(
                instance_id=f"instance-{index}",
                antibody_id=f"antibody-{index}",
                antigen_id=f"antigen-{index}",
                antibody_embeddings=torch.full(
                    (antibody_length, 2),
                    float(index),
                ),
                antigen_embeddings=torch.full(
                    (antigen_length, 2),
                    float(index),
                ),
                contact_targets=torch.zeros(
                    (antibody_length, antigen_length),
                ),
                contact_mask=torch.ones(
                    (antibody_length, antigen_length),
                    dtype=torch.bool,
                ),
                positive_pair_count=index + 1,
            )
        )
    return tuple(result)


def test_execution_lock_matches_every_controller_training_field() -> None:
    lock_path = (
        Path(__file__).parents[1]
        / "protocols"
        / "pairscale_e1_e2_execution_lock_v1.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    training = lock["training"]
    config = locked_e1_e2_config(17)
    payload = config.canonical_payload()
    direct_fields = (
        "checkpoint_every_epochs",
        "cognate_margin",
        "cognate_margin_temperature",
        "cognate_margin_weight",
        "contact_ranking_temperature",
        "contact_ranking_weight",
        "d_model",
        "decoy_count_per_direction",
        "dropout",
        "epochs",
        "eval_every_epochs",
        "ffn_dim",
        "gradient_clip_norm",
        "hard_negative_ratio",
        "joint_learning_rate_multiplier",
        "learning_rate",
        "num_fusion_layers",
        "num_heads",
        "pair_chunk_size",
        "pair_dim",
        "positive_weight",
        "precision",
        "retrieval_temperature",
        "retrieval_weight",
        "weight_decay",
    )
    assert {name: training[name] for name in direct_fields} == {
        name: payload[name] for name in direct_fields
    }
    assert training["phase_1_unary_warmup_end_epoch"] == (
        payload["unary_warmup_end_epoch"]
    )
    assert training["phase_2_detached_unary_residual_end_epoch"] == (
        payload["residual_phase_end_epoch"]
    )
    assert training["phase_3_joint_finetune_end_epoch"] == payload["epochs"]
    assert training["contact_logit_mode"] == (
        "unary_plus_interaction_residual"
    )
    assert training["specificity_score_source"] == (
        "double_centered_independent_specificity_interaction_residual"
    )
    assert tuple(lock["screening_seeds"]) == SCREENING_SEEDS
    with pytest.raises(ValueError, match="preregistered"):
        locked_e1_e2_config(99)


def test_decoy_receipt_is_value_and_label_independent() -> None:
    import torch

    examples = _examples()
    panel = select_score_free_bidirectional_decoys(examples, count=2, seed=17)
    baseline = _decoy_assignment_payload(examples, panel)
    altered = tuple(
        replace(
            example,
            antibody_embeddings=torch.full_like(
                example.antibody_embeddings,
                float("nan"),
            ),
            antigen_embeddings=torch.full_like(
                example.antigen_embeddings,
                float("nan"),
            ),
            contact_targets=torch.full_like(example.contact_targets, 99),
            contact_mask=torch.zeros_like(example.contact_mask),
            positive_pair_count=10_000 + index,
        )
        for index, example in enumerate(examples)
    )
    altered_panel = select_score_free_bidirectional_decoys(
        altered,
        count=2,
        seed=17,
    )
    assert _decoy_assignment_payload(altered, altered_panel) == baseline
    assert baseline["forbidden_fields_accessed"] is False
    assert baseline["embedding_values_accessed"] is False


def test_controller_freezes_checkpoint_before_any_holdout_deserialization() -> None:
    source = inspect.getsource(run_e1_e2_screen)
    checkpoint_gate = source.index('if int(last_receipt["epoch"])')
    access_receipt = source.index("DEVELOPMENT_HOLDOUT_ACCESS_RECEIPT.json")
    holdout_load = source.index("holdout_examples = _load_examples")
    assert checkpoint_gate < access_receipt < holdout_load
    assert '"new_final_test_accessed": False' in source
    assert '"scaling_authorized": False' in source
    assert "include_exhaustive_retrieval=True" in source[holdout_load:]


def test_controller_binds_all_new_scientific_modules() -> None:
    hashes = e1_e2_source_hashes()
    for name in (
        "pair_geometry.py",
        "pairscale_e1_e2_evaluation.py",
        "pairscale_e1_e2_step.py",
        "pairscale_e1_e2_training.py",
        "pairscale_partner_specificity.py",
    ):
        assert name in hashes
        assert len(hashes[name]) == 64
        assert set(hashes[name]) <= set("0123456789abcdef")
