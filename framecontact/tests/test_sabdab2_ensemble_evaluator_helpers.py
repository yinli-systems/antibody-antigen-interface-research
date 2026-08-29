from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def _module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "evaluate_framecontact_sabdab2_ensemble_test.py"
    spec = importlib.util.spec_from_file_location("sabdab2_ensemble_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_uses_role_specific_cluster_units() -> None:
    module = _module()
    dataset = SimpleNamespace(
        records=[
            {
                "instance_id": "a",
                "antigen_cluster_id": "ag-large",
                "antibody_cluster_id": "ab-one",
            },
            {
                "instance_id": "b",
                "antigen_cluster_id": "ag-large",
                "antibody_cluster_id": "ab-two",
            },
            {
                "instance_id": "c",
                "antigen_cluster_id": "ag-small",
                "antibody_cluster_id": "ab-two",
            },
        ]
    )
    metrics = {
        "partner_retrieval_by_direction": {
            "fixed_antigen_wrong_antibody": {
                "cluster_margins": {"ag-large": 0.2, "ag-small": 0.1}
            },
            "fixed_antibody_wrong_antigen": {
                "cluster_margins": {"ab-one": 0.3, "ab-two": 0.0}
            },
        },
        "contact": {"per_complex_auprc_by_instance": {"a": 1.0, "b": 1.0, "c": 0.0}},
        "epitope": {"per_complex_auprc_by_instance": {"a": 1.0, "b": 1.0, "c": 0.0}},
        "paratope": {"per_complex_auprc_by_instance": {"a": 1.0, "b": 0.0, "c": 0.0}},
    }
    module._add_bootstrap(metrics, dataset, replicates=100, seed=13)
    assert (
        metrics["contact"]["mean_per_complex_auprc_cluster_bootstrap_ci"][
            "estimate"
        ]
        == 0.5
    )
    assert (
        metrics["paratope"]["mean_per_complex_auprc_cluster_bootstrap_ci"][
            "group_count"
        ]
        == 2
    )


def test_independent_se3_control_compares_every_probability_element() -> None:
    module = _module()

    class InvariantDummy(torch.nn.Module):
        def forward(self, batch, *, return_attention=False):
            antibody_mask = batch["antibody_mask"]
            antigen_mask = batch["antigen_mask"]
            pair_mask = antibody_mask[:, :, None] & antigen_mask[:, None, :]
            batch_size, antibody_length, antigen_length = pair_mask.shape
            return {
                "contact_logits": torch.zeros(
                    batch_size, antibody_length, antigen_length
                ),
                "contact_interaction_logits": torch.zeros(
                    batch_size, antibody_length, antigen_length
                ),
                "paratope_logits": torch.zeros(batch_size, antibody_length),
                "epitope_logits": torch.zeros(batch_size, antigen_length),
                "pair_candidate_mask": pair_mask,
                "antibody_mask": antibody_mask,
                "antigen_mask": antigen_mask,
            }

    backbone = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]]
    )
    batch = {
        "antibody_backbone": backbone.clone(),
        "antigen_backbone": backbone.clone(),
        "antibody_mask": torch.ones(1, 1, dtype=torch.bool),
        "antigen_mask": torch.ones(1, 1, dtype=torch.bool),
        "antibody_frame_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        "antigen_frame_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        "antibody_missing_frame_mask": torch.zeros(1, 1, dtype=torch.bool),
        "antigen_missing_frame_mask": torch.zeros(1, 1, dtype=torch.bool),
    }
    result = module._independent_se3_prediction_invariance(
        InvariantDummy(),
        [batch],
        torch.device("cpu"),
        use_bf16=False,
        seed=17,
    )
    assert result["passed"] is True
    assert result["overall_max_absolute_probability_difference"] == 0.0
    assert all(value["prediction_count"] == 1 for value in result["metrics"].values())
