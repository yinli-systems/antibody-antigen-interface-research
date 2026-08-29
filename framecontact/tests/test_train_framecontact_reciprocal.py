"""Training-harness contracts for reciprocal wrong-partner construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_contact_gate_curriculum_has_zero_warmup_linear_ramp_and_unit_tail() -> None:
    from pathlib import Path
    import sys

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    import train_framecontact

    schedule = train_framecontact._contact_gate_curriculum
    assert schedule(0, 100, warmup_fraction=0.05, ramp_fraction=0.10) == 0.0
    assert schedule(5, 100, warmup_fraction=0.05, ramp_fraction=0.10) == 0.0
    assert schedule(10, 100, warmup_fraction=0.05, ramp_fraction=0.10) == 0.5
    assert schedule(15, 100, warmup_fraction=0.05, ramp_fraction=0.10) == 1.0
    assert schedule(99, 100, warmup_fraction=0.05, ramp_fraction=0.10) == 1.0
    with pytest.raises(ValueError, match="sum"):
        schedule(0, 100, warmup_fraction=0.8, ramp_fraction=0.3)


def _training_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_framecontact.py"
    spec = importlib.util.spec_from_file_location("train_framecontact_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Dataset:
    def __init__(self) -> None:
        self.records = tuple({"instance_id": f"case_{index}"} for index in range(4))
        self.items = [
            {
                "instance_id": f"case_{index}",
                "antibody_features": f"ab_{index}",
                "antigen_features": f"ag_{index}",
                "antibody_labels": f"ab_labels_{index}",
                "antigen_labels": f"ag_labels_{index}",
                "pair_labels": f"pair_labels_{index}",
                "pair_candidate_mask": f"pair_mask_{index}",
            }
            for index in range(4)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return dict(self.items[index])


def test_reciprocal_training_sample_swaps_exactly_one_partner() -> None:
    module = _training_module()
    entries = []
    for index in range(4):
        candidates = [f"case_{other}" for other in range(4) if other != index]
        entries.append(
            {
                "instance_id": f"case_{index}",
                "wrong_antigen_instance_ids": candidates,
                "wrong_antibody_instance_ids": list(reversed(candidates)),
            }
        )
    panel = {
        "format_version": module.RECIPROCAL_PARTNER_PANEL_FORMAT,
        "entries": entries,
    }
    dataset = _Dataset()
    wrapped = module._FrozenPanelTrainingDataset(dataset, panel, seed=17)
    wrapped.set_epoch(3)
    focal, wrong_antigen, wrong_antibody = wrapped[0]
    assert wrong_antigen["antibody_features"] == focal["antibody_features"]
    assert wrong_antigen["antigen_features"] != focal["antigen_features"]
    assert wrong_antibody["antigen_features"] == focal["antigen_features"]
    assert wrong_antibody["antibody_features"] != focal["antibody_features"]
    assert "pair_labels" not in wrong_antigen
    assert "pair_labels" not in wrong_antibody


def test_reciprocal_partner_choice_is_seed_epoch_deterministic() -> None:
    module = _training_module()
    entries = [
        {
            "instance_id": f"case_{index}",
            "wrong_antigen_instance_ids": [
                f"case_{other}" for other in range(4) if other != index
            ],
            "wrong_antibody_instance_ids": [
                f"case_{other}" for other in range(4) if other != index
            ],
        }
        for index in range(4)
    ]
    panel = {
        "format_version": module.RECIPROCAL_PARTNER_PANEL_FORMAT,
        "entries": entries,
    }
    first = module._FrozenPanelTrainingDataset(_Dataset(), panel, seed=73)
    second = module._FrozenPanelTrainingDataset(_Dataset(), panel, seed=73)
    first.set_epoch(11)
    second.set_epoch(11)
    assert first[2][1]["instance_id"] == second[2][1]["instance_id"]
    assert first[2][2]["instance_id"] == second[2][2]["instance_id"]


def test_reciprocal_training_supports_unique_deterministic_multi_negatives() -> None:
    module = _training_module()
    entries = [
        {
            "instance_id": f"case_{index}",
            "wrong_antigen_instance_ids": [
                f"case_{other}" for other in range(4) if other != index
            ],
            "wrong_antibody_instance_ids": [
                f"case_{other}" for other in range(4) if other != index
            ],
        }
        for index in range(4)
    ]
    panel = {
        "format_version": module.RECIPROCAL_PARTNER_PANEL_FORMAT,
        "entries": entries,
    }
    first = module._FrozenPanelTrainingDataset(
        _Dataset(), panel, seed=17, negative_count=3
    )
    second = module._FrozenPanelTrainingDataset(
        _Dataset(), panel, seed=17, negative_count=3
    )
    first.set_epoch(5)
    second.set_epoch(5)
    first_sample = first[0]
    second_sample = second[0]
    assert len(first_sample) == 3
    assert len(first_sample[1]) == len(first_sample[2]) == 3
    assert [item["instance_id"] for item in first_sample[1]] == [
        item["instance_id"] for item in second_sample[1]
    ]
    assert len({item["instance_id"] for item in first_sample[1]}) == 3
    assert len({item["instance_id"] for item in first_sample[2]}) == 3
    assert all(
        item["antibody_features"] == first_sample[0]["antibody_features"]
        for item in first_sample[1]
    )
    assert all(
        item["antigen_features"] == first_sample[0]["antigen_features"]
        for item in first_sample[2]
    )


def test_multi_negative_count_cannot_exceed_the_frozen_panel() -> None:
    module = _training_module()
    entries = [
        {
            "instance_id": f"case_{index}",
            "wrong_antigen_instance_ids": [
                f"case_{other}" for other in range(4) if other != index
            ],
            "wrong_antibody_instance_ids": [
                f"case_{other}" for other in range(4) if other != index
            ],
        }
        for index in range(4)
    ]
    panel = {
        "format_version": module.RECIPROCAL_PARTNER_PANEL_FORMAT,
        "entries": entries,
    }
    with pytest.raises(ValueError, match="fewer candidates"):
        module._FrozenPanelTrainingDataset(
            _Dataset(), panel, seed=17, negative_count=4
        )


def test_checkpointed_wrong_view_matches_direct_forward_and_has_gradients() -> None:
    torch = pytest.importorskip("torch")
    module = _training_module()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, batch):
            logits = self.scale * batch["features"]
            return {
                "contact_logits": logits,
                "contact_interaction_logits": logits + 1.0,
                "pair_candidate_mask": batch["mask"],
            }

    model = Model()
    batch = {
        "features": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        "mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "instance_id": ["example"],
    }
    direct = module._wrong_training_view(model(batch))
    checkpointed = module._checkpointed_wrong_training_view(model, batch)
    for key in direct:
        torch.testing.assert_close(checkpointed[key], direct[key])
    checkpointed["contact_interaction_logits"].sum().backward()
    assert model.scale.grad is not None
    assert float(model.scale.grad.abs()) > 0


def test_checkpointed_wrong_view_preserves_training_dropout_rng() -> None:
    torch = pytest.importorskip("torch")
    module = _training_module()

    class DropoutModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.5))
            self.dropout = torch.nn.Dropout(p=0.5)

        def forward(self, batch):
            logits = self.dropout(self.scale * batch["features"])
            return {
                "contact_logits": logits,
                "contact_interaction_logits": logits + 1.0,
                "pair_candidate_mask": batch["mask"],
            }

    model = DropoutModel().train()
    batch = {
        "features": torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        "mask": torch.ones(1, 3, 4, dtype=torch.bool),
        "instance_id": ["dropout-example"],
    }
    torch.manual_seed(12345)
    direct = module._wrong_training_view(model(batch))
    torch.manual_seed(12345)
    checkpointed = module._checkpointed_wrong_training_view(model, batch)
    for key in direct:
        torch.testing.assert_close(checkpointed[key], direct[key])
    checkpointed["contact_interaction_logits"].sum().backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_checkpointed_wrong_view_preserves_contact_bottleneck_score() -> None:
    torch = pytest.importorskip("torch")
    module = _training_module()

    class BottleneckModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.25))
            # The harness uses the same explicit enablement signal as FrameContact.
            self.contact_bottleneck_head = torch.nn.Identity()

        def forward(self, batch):
            logits = self.scale * batch["features"]
            return {
                "contact_logits": logits,
                "contact_interaction_logits": logits,
                "pair_candidate_mask": batch["mask"],
                "contact_bottleneck_compatibility": logits.mean(dim=(1, 2)),
            }

    model = BottleneckModel()
    batch = {
        "features": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        "mask": torch.ones(1, 2, 3, dtype=torch.bool),
    }
    direct = module._wrong_training_view(model(batch))
    checkpointed = module._checkpointed_wrong_training_view(model, batch)
    assert set(checkpointed) == set(direct)
    for key in direct:
        torch.testing.assert_close(checkpointed[key], direct[key])
    checkpointed["contact_bottleneck_compatibility"].sum().backward()
    assert model.scale.grad is not None and float(model.scale.grad.abs()) > 0


def test_training_matrix_contains_real_invariant_point_baseline() -> None:
    module = _training_module()
    specification = module.EXPERIMENTS["frozen_frame_ipa"]
    assert specification == {
        "geometry_feature_mode": "none",
        "geometry_bias_mode": "invariant_points",
        "geometry": True,
        "cross": False,
        "fusion_mode": "late",
    }


def test_partner_aware_geometry_ablated_controls_keep_pair_paths() -> None:
    module = _training_module()
    expected_common = {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": False,
        "cross": True,
        "fusion_mode": "late",
        "residue_output_mode": "gated_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
    }
    binary = module.EXPERIMENTS[
        "framecontact_geometry_ablated_contact_hybrid"
    ]
    distogram = module.EXPERIMENTS["framecontact_geometry_ablated_distogram"]
    assert binary == expected_common | {"pair_output_mode": "binary_contact"}
    assert distogram == expected_common | {"pair_output_mode": "distogram"}


def test_model_factory_wires_default_off_contact_bottleneck_candidate() -> None:
    torch = pytest.importorskip("torch")
    module = _training_module()
    first = {
        "antibody_features": torch.zeros(3, 12),
        "antigen_features": torch.zeros(4, 10),
    }
    configuration = {
        "d_model": 16,
        "num_heads": 4,
        "num_self_layers": 1,
        "num_rbf": 8,
        "neighbor_k": 4,
        "neighbor_radius": 16.0,
        "sequence_window": 8,
        "dropout": 0.0,
        "attention_kernel": "reference",
    }
    disabled = module._model_from_configuration(
        first, configuration, "framecontact_contact_hybrid"
    )
    assert disabled.contact_bottleneck_head is None

    enabled = module._model_from_configuration(
        first,
        configuration
        | {
            "contact_bottleneck_projection_dim": 7,
            "contact_bottleneck_gradient_mode": "joint",
        },
        "framecontact_contact_hybrid",
    )
    assert enabled.contact_bottleneck_head is not None
    assert enabled.contact_bottleneck_head.gradient_mode == "joint"


def test_contact_bottleneck_weight_diagnostics_detect_collapse_and_diffusion() -> None:
    np = pytest.importorskip("numpy")
    module = _training_module()
    mask = np.asarray([True, True, True, False])
    concentrated = module._weight_distribution_diagnostics(
        np.asarray([1.0, 0.0, 0.0, 999.0]), mask
    )
    assert concentrated["positive_weight_fraction"] == pytest.approx(1 / 3)
    assert concentrated["normalized_entropy"] == pytest.approx(0.0)
    assert concentrated["effective_support"] == pytest.approx(1.0)
    assert concentrated["top1_mass_fraction"] == pytest.approx(1.0)

    diffuse = module._weight_distribution_diagnostics(
        np.asarray([1.0, 1.0, 1.0, 999.0]), mask
    )
    assert diffuse["positive_weight_fraction"] == 1.0
    assert diffuse["normalized_entropy"] == pytest.approx(1.0)
    assert diffuse["effective_support"] == pytest.approx(3.0)
    assert diffuse["top1_mass_fraction"] == pytest.approx(1 / 3)

    zero = module._weight_distribution_diagnostics(np.zeros(4), mask)
    assert zero["zero_total_weight"] is True
    assert zero["normalized_entropy"] is None
    assert zero["effective_support"] == 0.0
    summary = module._aggregate_weight_diagnostics(
        [concentrated, diffuse, zero]
    )
    assert summary["complex_count"] == 3
    assert summary["zero_total_weight_complex_count"] == 1
    assert summary["mean_by_metric_over_defined_complexes"][
        "normalized_entropy"
    ] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="nonnegative"):
        module._weight_distribution_diagnostics(
            np.asarray([1.0, -0.1, 0.0, 0.0]), mask
        )


def test_execution_configuration_fails_closed_on_batch_drift() -> None:
    module = _training_module()
    implementation = {
        "d_model": 128,
        "num_heads": 8,
        "num_self_layers": 2,
        "num_rbf": 16,
        "neighbor_k": 32,
        "neighbor_radius_angstrom": 16.0,
        "sequence_window": 32,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 1,
        "evaluation_batch_size": 1,
        "precision": "bf16",
        "calibration_max_pairs": 2_000_000,
        "offload_saved_tensors": True,
    }
    values = {
        "d_model": 128,
        "num_heads": 8,
        "num_self_layers": 2,
        "num_rbf": 16,
        "neighbor_k": 32,
        "neighbor_radius": 16.0,
        "sequence_window": 32,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 2,
        "evaluation_batch_size": 1,
        "precision": "bf16",
        "calibration_max_pairs": 2_000_000,
        "offload_saved_tensors": True,
        "contact_loss_weight": 1.0,
        "paratope_loss_weight": 0.25,
        "epitope_loss_weight": 0.25,
        "specificity_loss_weight": 0.25,
        "specificity_margin": 0.05,
        "score_fraction": 0.01,
    }
    protocol = {
        "implementation": implementation,
        "training_objective": {
            "loss_weights": {
                "contact": 1.0,
                "paratope": 0.25,
                "epitope": 0.25,
                "specificity": 0.25,
            },
            "specificity_margin": 0.05,
            "score_fraction": 0.01,
        },
    }
    with pytest.raises(ValueError, match="batch_size differs"):
        module._validate_execution_configuration(protocol, SimpleNamespace(**values))


def test_execution_configuration_supports_frozen_condition_specific_losses() -> None:
    module = _training_module()
    implementation = {
        "d_model": 128,
        "num_heads": 8,
        "num_self_layers": 2,
        "num_rbf": 16,
        "neighbor_k": 32,
        "neighbor_radius_angstrom": 16.0,
        "sequence_window": 32,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 1,
        "evaluation_batch_size": 1,
        "precision": "bf16",
        "calibration_max_pairs": 2_000_000,
        "offload_saved_tensors": True,
        "triangle_neighbor_radius_angstrom": 12.0,
        "triangle_neighbor_k": 8,
        "triangle_slack_angstrom": 0.5,
        "antigen_patch_top_seeds": 1,
        "antigen_patch_radius_angstrom": 12.0,
    }
    values = {
        "d_model": 128,
        "num_heads": 8,
        "num_self_layers": 2,
        "num_rbf": 16,
        "neighbor_k": 32,
        "neighbor_radius": 16.0,
        "sequence_window": 32,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 1,
        "evaluation_batch_size": 1,
        "precision": "bf16",
        "calibration_max_pairs": 2_000_000,
        "offload_saved_tensors": True,
        "triangle_neighbor_radius": 12.0,
        "triangle_neighbor_k": 8,
        "triangle_slack": 0.5,
        "antigen_patch_top_seeds": 1,
        "antigen_patch_radius": 12.0,
        "contact_loss_weight": 1.0,
        "paratope_loss_weight": 0.25,
        "epitope_loss_weight": 0.25,
        "specificity_loss_weight": 0.25,
        "distogram_loss_weight": 0.5,
        "triangle_loss_weight": 0.0,
        "specificity_margin": 0.05,
        "score_fraction": 0.01,
        "experiment": "framecontact_distogram",
    }
    base_weights = {
        "contact": 1.0,
        "paratope": 0.25,
        "epitope": 0.25,
        "specificity": 0.25,
        "distogram": 0.5,
    }
    protocol = {
        "implementation": implementation,
        "training_objective": {
            "loss_weights_by_condition": {
                "framecontact_distogram": {**base_weights, "triangle": 0.0},
                "framecontact_distogram_triangle": {
                    **base_weights,
                    "triangle": 0.1,
                },
            },
            "specificity_margin": 0.05,
            "score_fraction": 0.01,
        },
    }
    module._validate_execution_configuration(protocol, SimpleNamespace(**values))
    values["experiment"] = "framecontact_distogram_triangle"
    values["triangle_loss_weight"] = 0.1
    module._validate_execution_configuration(protocol, SimpleNamespace(**values))
    values["triangle_loss_weight"] = 0.0
    with pytest.raises(ValueError, match="triangle_loss_weight differs"):
        module._validate_execution_configuration(protocol, SimpleNamespace(**values))


def test_execution_configuration_locks_residue_contrastive_semantics() -> None:
    module = _training_module()
    protocol = {
        "implementation": {},
        "training_objective": {
            "loss_weights": {},
            "specificity_margin": 0.05,
            "score_fraction": 0.01,
            "residue_contrastive_margin": 0.5,
            "residue_contrastive_mode": "pairwise_softplus",
            "residue_contrastive_temperature": 1.0,
        },
    }
    values = SimpleNamespace(
        contact_loss_weight=0.0,
        paratope_loss_weight=0.0,
        epitope_loss_weight=0.0,
        specificity_loss_weight=0.0,
        distogram_loss_weight=0.0,
        triangle_loss_weight=0.0,
        edm_loss_weight=0.0,
        residue_partner_contrastive_loss_weight=0.0,
        specificity_margin=0.05,
        score_fraction=0.01,
        residue_contrastive_margin=0.5,
        residue_contrastive_mode="pairwise_softplus",
        residue_contrastive_temperature=1.0,
    )
    module._validate_execution_configuration(protocol, values)
    values.residue_contrastive_temperature = 0.5
    with pytest.raises(ValueError, match="residue_contrastive_temperature differs"):
        module._validate_execution_configuration(protocol, values)
