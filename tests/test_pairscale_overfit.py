"""Focused tests for the train-only PAIRScale overfit gate."""

from __future__ import annotations

import importlib.util
import inspect

import pytest

from interaction_first.pairscale_overfit import (
    PairscaleOverfitConfig,
    contact_only_geometry_schema,
    deterministic_epoch_order,
    hard_negative_contact_ranking_loss,
    run_overfit_training,
)


def test_configuration_and_contact_schema_are_explicit() -> None:
    config = PairscaleOverfitConfig()
    config.validate()
    assert config.canonical_payload()["gate_auprc"] == 0.95
    schema = contact_only_geometry_schema()
    assert schema.contact_distance_bin_count == 2
    assert schema.num_distance_bins == 5
    assert schema.primary_contact_definition.endswith("le_4.5_angstrom")


def test_epoch_order_is_stateless_complete_and_epoch_specific() -> None:
    first = deterministic_epoch_order(32, 17, 1)
    replay = deterministic_epoch_order(32, 17, 1)
    second = deterministic_epoch_order(32, 17, 2)
    assert first == replay
    assert sorted(first) == list(range(32))
    assert first != second


def test_resume_path_keeps_checkpoint_rng_tensors_on_cpu() -> None:
    source = inspect.getsource(run_overfit_training)
    assert 'map_location="cpu"' in source


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("positive_weight", 0.0),
        ("dropout", 1.0),
        ("precision", "fp16"),
        ("gate_partner_win_rate", 1.1),
    ],
)
def test_invalid_training_configuration_fails_closed(field: str, value: object) -> None:
    payload = PairscaleOverfitConfig().canonical_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        PairscaleOverfitConfig(**payload).validate()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)
def test_hard_negative_ranking_prefers_separated_contact_logits() -> None:
    import torch

    targets = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    separated = torch.tensor([[[4.0, -1.0, -2.0, 3.0]]])
    inverted = -separated
    separated_loss = hard_negative_contact_ranking_loss(
        separated,
        targets,
        mask,
        negative_ratio=2,
        temperature=1.0,
    )
    inverted_loss = hard_negative_contact_ranking_loss(
        inverted,
        targets,
        mask,
        negative_ratio=2,
        temperature=1.0,
    )
    assert float(separated_loss) < float(inverted_loss)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)
def test_contact_only_model_executes_without_unlabeled_heads() -> None:
    import torch

    from interaction_first.pair_geometry import ProspectivePairGeometryModel

    schema = contact_only_geometry_schema()
    model = ProspectivePairGeometryModel(
        antibody_input_dim=5,
        antigen_input_dim=7,
        d_model=12,
        pair_dim=8,
        num_heads=3,
        num_fusion_layers=1,
        ffn_dim=24,
        geometry_schema=schema,
        pair_chunk_size=2,
        dropout=0.0,
    )
    antibody = torch.randn(1, 3, 5)
    antigen = torch.randn(1, 4, 7)
    antibody_mask = torch.ones((1, 3), dtype=torch.bool)
    antigen_mask = torch.ones((1, 4), dtype=torch.bool)
    pair_mask = antibody_mask.unsqueeze(2) & antigen_mask.unsqueeze(1)
    outputs = model(
        antibody,
        antigen,
        antibody_mask,
        antigen_mask,
        pair_mask,
        return_pair_representations=False,
        enabled_heads={"contact", "distance"},
    )
    assert outputs["executed_heads"] == ("contact", "distance")
    assert "orientation_bin_logits" not in outputs
    assert "pair_interactions" not in outputs
    assert tuple(outputs["contact_logits"].shape) == (1, 3, 4)
