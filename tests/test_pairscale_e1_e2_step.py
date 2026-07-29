"""Tests for the fixed staged PAIRScale E1/E2 training step."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from interaction_first.development_contact_dataset import DevelopmentContactExample
from interaction_first.pairscale_e1_e2_step import (
    PHASE_JOINT_FINETUNE,
    PHASE_RESIDUAL_SPECIFICITY,
    PHASE_UNARY_WARMUP,
    PAIR_PATH_MODULES,
    UNARY_PATH_MODULES,
    UNUSED_FROZEN_MODULES,
    PairscaleE1E2Config,
    _forward_cognate,
    build_e1_e2_model,
    parameter_path_state_sha256,
    set_phase_trainable_parameters,
    train_one_epoch_e1_e2,
    training_phase,
)
from interaction_first.pairscale_partner_specificity import (
    select_score_free_bidirectional_decoys,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _config() -> PairscaleE1E2Config:
    return PairscaleE1E2Config(
        seed=17,
        epochs=3,
        unary_warmup_end_epoch=1,
        residual_phase_end_epoch=2,
        learning_rate=0.01,
        joint_learning_rate_multiplier=0.1,
        positive_weight=4.0,
        contact_ranking_weight=0.5,
        hard_negative_ratio=2,
        retrieval_weight=0.5,
        retrieval_temperature=0.7,
        cognate_margin_weight=0.5,
        cognate_margin=0.1,
        cognate_margin_temperature=0.7,
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

    torch.manual_seed(20260728)
    examples = []
    for index in range(4):
        antibody_length = 3 + index % 2
        antigen_length = 4 + index % 2
        targets = torch.zeros((antibody_length, antigen_length))
        targets[0, 0] = 1.0
        targets[-1, -1] = 1.0
        mask = torch.ones_like(targets, dtype=torch.bool)
        mask[0, -1] = False
        examples.append(
            DevelopmentContactExample(
                instance_id=f"instance-{index}",
                antibody_id=f"antibody-{index}",
                antigen_id=f"antigen-{index}",
                antibody_embeddings=torch.randn(antibody_length, 5),
                antigen_embeddings=torch.randn(antigen_length, 7),
                contact_targets=targets,
                contact_mask=mask,
                positive_pair_count=2,
            )
        )
    return tuple(examples)


def _clone_parameter(module, name: str):
    import torch

    parameter = dict(module.named_parameters())[name]
    return parameter.detach().clone()


def _clone_modules(model, module_names: tuple[str, ...]):
    return {
        f"{module_name}.{name}": parameter.detach().clone()
        for module_name in module_names
        for name, parameter in getattr(model, module_name).named_parameters()
    }


def _assert_state_exact(reference, observed) -> None:
    import torch

    assert set(reference) == set(observed)
    for name in reference:
        assert torch.equal(reference[name], observed[name]), name


def _unary_logits(model, example):
    import torch

    model.eval()
    with torch.inference_mode():
        outputs, _, _ = _forward_cognate(model, example, torch.device("cpu"))
    return (
        outputs["antibody_unary_logits"].detach().clone(),
        outputs["antigen_unary_logits"].detach().clone(),
    )


def test_phase_schedule_and_joint_learning_rate_are_frozen() -> None:
    config = _config()
    assert training_phase(1, config) == PHASE_UNARY_WARMUP
    assert training_phase(2, config) == PHASE_RESIDUAL_SPECIFICITY
    assert training_phase(3, config) == PHASE_JOINT_FINETUNE
    with pytest.raises(ValueError, match="exceeds"):
        training_phase(4, config)
    with pytest.raises(ValueError, match="must be true"):
        PairscaleE1E2Config(
            freeze_complete_unary_path_during_residual_phase=False
        ).validate()


def test_cognate_specificity_uses_complete_support_while_contact_uses_mask() -> None:
    import torch

    examples = _examples()
    dataset = SimpleNamespace(antibody_input_dim=5, antigen_input_dim=7)
    model = build_e1_e2_model(dataset, _config()).eval()
    outputs, targets, supervision_mask = _forward_cognate(
        model,
        examples[0],
        torch.device("cpu"),
    )
    assert bool(outputs["candidate_pair_mask"].all())
    assert not bool(supervision_mask.all())
    torch.testing.assert_close(targets[0], examples[0].contact_targets)
    assert torch.equal(supervision_mask[0], examples[0].contact_mask)


def test_three_phases_update_only_the_authorized_heads() -> None:
    import torch

    config = _config()
    examples = _examples()
    panel = select_score_free_bidirectional_decoys(
        examples,
        count=config.decoy_count_per_direction,
        seed=config.seed,
    )
    dataset = SimpleNamespace(antibody_input_dim=5, antigen_input_dim=7)
    model = build_e1_e2_model(dataset, config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    pair_before_warmup = _clone_modules(model, PAIR_PATH_MODULES)
    unused_before_training = _clone_modules(model, UNUSED_FROZEN_MODULES)
    tracked = {
        "antibody_unary_head.weight": _clone_parameter(
            model,
            "antibody_unary_head.weight",
        ),
        "interaction_head.weight": _clone_parameter(
            model,
            "interaction_head.weight",
        ),
        "specificity_head.weight": _clone_parameter(
            model,
            "specificity_head.weight",
        ),
    }

    warmup_metrics, warmup_steps = train_one_epoch_e1_e2(
        model=model,
        optimizer=optimizer,
        examples=examples,
        device=torch.device("cpu"),
        config=config,
        epoch=1,
        decoy_panel=panel,
    )
    assert warmup_steps == len(examples)
    assert warmup_metrics["phase"] == PHASE_UNARY_WARMUP
    assert warmup_metrics["bidirectional_retrieval"] == 0.0
    assert not torch.equal(
        tracked["antibody_unary_head.weight"],
        _clone_parameter(model, "antibody_unary_head.weight"),
    )
    torch.testing.assert_close(
        tracked["interaction_head.weight"],
        _clone_parameter(model, "interaction_head.weight"),
    )
    torch.testing.assert_close(
        tracked["specificity_head.weight"],
        _clone_parameter(model, "specificity_head.weight"),
    )
    _assert_state_exact(pair_before_warmup, _clone_modules(model, PAIR_PATH_MODULES))
    assert warmup_metrics["unary_path_trainable"] is True
    assert warmup_metrics["pair_path_trainable"] is False

    unary_before_residual = _clone_modules(model, UNARY_PATH_MODULES)
    unary_hash_before_residual = parameter_path_state_sha256(model, "unary")
    pair_hash_before_residual = parameter_path_state_sha256(model, "pair")
    unary_logits_before_residual = _unary_logits(model, examples[0])
    interaction_before_residual = _clone_parameter(
        model,
        "interaction_head.weight",
    )
    specificity_before_residual = _clone_parameter(
        model,
        "specificity_head.weight",
    )
    residual_metrics, residual_steps = train_one_epoch_e1_e2(
        model=model,
        optimizer=optimizer,
        examples=examples,
        device=torch.device("cpu"),
        config=config,
        epoch=2,
        decoy_panel=panel,
    )
    assert residual_steps == len(examples)
    assert residual_metrics["phase"] == PHASE_RESIDUAL_SPECIFICITY
    assert float(residual_metrics["bidirectional_retrieval"]) > 0.0
    _assert_state_exact(
        unary_before_residual,
        _clone_modules(model, UNARY_PATH_MODULES),
    )
    assert parameter_path_state_sha256(model, "unary") == (
        unary_hash_before_residual
    )
    assert parameter_path_state_sha256(model, "pair") != pair_hash_before_residual
    unary_logits_after_residual = _unary_logits(model, examples[0])
    assert torch.equal(
        unary_logits_before_residual[0],
        unary_logits_after_residual[0],
    )
    assert torch.equal(
        unary_logits_before_residual[1],
        unary_logits_after_residual[1],
    )
    assert not torch.equal(
        interaction_before_residual,
        _clone_parameter(model, "interaction_head.weight"),
    )
    assert not torch.equal(
        specificity_before_residual,
        _clone_parameter(model, "specificity_head.weight"),
    )
    assert residual_metrics["unary_path_trainable"] is False
    assert residual_metrics["pair_path_trainable"] is True

    unary_before_joint = _clone_parameter(model, "antibody_unary_head.weight")
    joint_metrics, joint_steps = train_one_epoch_e1_e2(
        model=model,
        optimizer=optimizer,
        examples=examples,
        device=torch.device("cpu"),
        config=config,
        epoch=3,
        decoy_panel=panel,
    )
    assert joint_steps == len(examples)
    assert joint_metrics["phase"] == PHASE_JOINT_FINETUNE
    assert joint_metrics["learning_rate"] == pytest.approx(0.001)
    assert not torch.equal(
        unary_before_joint,
        _clone_parameter(model, "antibody_unary_head.weight"),
    )
    assert joint_metrics["unary_path_trainable"] is True
    assert joint_metrics["pair_path_trainable"] is True
    assert joint_metrics["unused_geometry_heads_trainable"] is False
    _assert_state_exact(
        unused_before_training,
        _clone_modules(model, UNUSED_FROZEN_MODULES),
    )
    assert all(
        isinstance(value, str) or torch.isfinite(torch.tensor(value))
        for value in joint_metrics.values()
    )


def test_parameter_ownership_rejects_an_unclassified_module() -> None:
    import torch

    dataset = SimpleNamespace(antibody_input_dim=5, antigen_input_dim=7)
    model = build_e1_e2_model(dataset, _config())
    model.unclassified = torch.nn.Linear(3, 3)
    with pytest.raises(RuntimeError, match="Unclassified"):
        set_phase_trainable_parameters(model, PHASE_UNARY_WARMUP)
    with pytest.raises(ValueError, match="unary, pair, or unused"):
        del model.unclassified
        parameter_path_state_sha256(model, "unknown")
