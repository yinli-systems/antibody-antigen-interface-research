"""Tests for fixed E1/E2 contact and partner evaluation."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from interaction_first.development_contact_dataset import DevelopmentContactExample
from interaction_first.pairscale_e1_e2_evaluation import evaluate_e1_e2
from interaction_first.pairscale_e1_e2_step import (
    PairscaleE1E2Config,
    build_e1_e2_model,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _config() -> PairscaleE1E2Config:
    return PairscaleE1E2Config(
        epochs=3,
        unary_warmup_end_epoch=1,
        residual_phase_end_epoch=2,
        decoy_count_per_direction=1,
        d_model=12,
        pair_dim=8,
        num_heads=3,
        num_fusion_layers=2,
        ffn_dim=24,
        pair_chunk_size=2,
        precision="fp32",
    )


def _examples() -> tuple[DevelopmentContactExample, ...]:
    import torch

    torch.manual_seed(20260729)
    endpoint_pairs = (
        ("ab-0", "ag-0"),
        ("ab-0", "ag-1"),
        ("ab-1", "ag-0"),
        ("ab-2", "ag-2"),
        ("ab-3", "ag-3"),
        ("ab-4", "ag-4"),
    )
    result = []
    for index, (antibody_id, antigen_id) in enumerate(endpoint_pairs):
        targets = torch.zeros((3, 4))
        targets[0, 0] = 1.0
        targets[2, 3] = 1.0
        mask = torch.ones_like(targets, dtype=torch.bool)
        mask[1, 2] = False
        result.append(
            DevelopmentContactExample(
                instance_id=f"instance-{index}",
                antibody_id=antibody_id,
                antigen_id=antigen_id,
                antibody_embeddings=torch.randn(3, 5),
                antigen_embeddings=torch.randn(4, 7),
                contact_targets=targets,
                contact_mask=mask,
                positive_pair_count=2,
            )
        )
    return tuple(result)


def test_evaluation_reports_contact_decomposition_and_both_retrieval_axes() -> None:
    import torch

    model = build_e1_e2_model(
        SimpleNamespace(antibody_input_dim=5, antigen_input_dim=7),
        _config(),
    )
    result = evaluate_e1_e2(
        model=model,
        examples=_examples(),
        device=torch.device("cpu"),
        decoy_count=1,
        seed=17,
        include_partner_panel=True,
        include_exhaustive_retrieval=True,
    )
    assert result["format_version"] == "pairscale_e1_e2_development_evaluation_v1"
    assert result["model_selection_authorized"] is False
    assert result["unknown_cross_pairs_assigned_binary_labels"] is False
    assert result["complex_count"] == 6
    assert len(result["per_complex"]) == 6
    assert all(
        "unary_additive_auprc" in item
        and "interaction_residual_auprc" in item
        and "ap_lift" in item
        for item in result["per_complex"]
    )
    controls = result["partner_controls"]
    assert controls["decoy_count_per_direction"] == 1
    for direction in ("antibody_to_antigen", "antigen_to_antibody"):
        assert controls[direction]["centered_residual"][
            "pairwise_comparison_count"
        ] == 6
        assert controls[direction]["raw_residual"][
            "pairwise_comparison_count"
        ] == 6
    exhaustive = result["exhaustive_bidirectional_retrieval"]
    assert exhaustive["observed_cognate_pair_count"] == 6
    assert exhaustive["candidate_pair_count"] == 25
    assert exhaustive["unknown_cross_pair_semantics"] == (
        "unlabeled_retrieval_control_not_nonbinder"
    )
    for score_type in ("centered_residual", "raw_residual"):
        assert exhaustive[score_type]["negative_candidate_subsampling_performed"] is False
        assert exhaustive[score_type]["antibody_to_antigen"]["query_count"] == 5
        assert exhaustive[score_type]["antigen_to_antibody"]["query_count"] == 5
    numeric = (
        result["macro_pair_auprc"],
        result["pooled_pair_auprc"],
        result["macro_ap_lift"],
        result["macro_unary_additive_auprc"],
        result["macro_interaction_residual_auprc"],
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in numeric)


def test_fast_training_evaluation_can_skip_all_partner_cross_products() -> None:
    import torch

    model = build_e1_e2_model(
        SimpleNamespace(antibody_input_dim=5, antigen_input_dim=7),
        _config(),
    )
    result = evaluate_e1_e2(
        model=model,
        examples=_examples(),
        device=torch.device("cpu"),
        decoy_count=1,
        seed=17,
        include_partner_panel=False,
        include_exhaustive_retrieval=False,
    )
    assert result["partner_controls"] is None
    assert result["exhaustive_bidirectional_retrieval"] is None
