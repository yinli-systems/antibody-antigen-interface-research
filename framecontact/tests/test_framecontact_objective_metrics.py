"""Tests for pre-registered FrameContact training and scoring semantics."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def test_bernoulli_js_consistency_is_symmetric_and_zero_for_matching_heads() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        bernoulli_js_consistency_per_complex,
    )

    direct = torch.tensor([[[0.2, -1.0], [1.5, 0.0]]], requires_grad=True)
    distance = torch.tensor([[[-0.3, -0.2], [0.7, 0.4]]], requires_grad=True)
    mask = torch.tensor([[[True, True], [True, False]]])
    forward = bernoulli_js_consistency_per_complex(direct, distance, mask)
    reverse = bernoulli_js_consistency_per_complex(distance, direct, mask)
    torch.testing.assert_close(forward, reverse)
    assert forward.item() > 0.0
    torch.testing.assert_close(
        bernoulli_js_consistency_per_complex(direct, direct, mask),
        torch.zeros(()),
        atol=1e-8,
        rtol=0.0,
    )
    forward.backward()
    assert direct.grad is not None and torch.isfinite(direct.grad).all()
    assert distance.grad is not None and torch.isfinite(distance.grad).all()


def test_pair_node_consistency_is_masked_balanced_and_differentiable() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        pair_node_consistency_per_complex,
    )

    paratope_unary = torch.tensor([[1.0, -50.0]], requires_grad=True)
    paratope_contact = torch.tensor([[-0.5, 50.0]], requires_grad=True)
    antibody_mask = torch.tensor([[True, False]])
    epitope_unary = torch.tensor([[0.2, -0.7, 50.0]], requires_grad=True)
    epitope_contact = torch.tensor([[-0.3, 0.9, -50.0]], requires_grad=True)
    antigen_mask = torch.tensor([[True, True, False]])

    total, paratope, epitope = pair_node_consistency_per_complex(
        paratope_unary_logits=paratope_unary,
        paratope_contact_logits=paratope_contact,
        antibody_mask=antibody_mask,
        epitope_unary_logits=epitope_unary,
        epitope_contact_logits=epitope_contact,
        antigen_mask=antigen_mask,
    )
    torch.testing.assert_close(total, 0.5 * (paratope + epitope))
    assert total.item() > 0.0

    masked_values_changed = pair_node_consistency_per_complex(
        paratope_unary_logits=paratope_unary.detach().clone().index_fill(
            1, torch.tensor([1]), 500.0
        ),
        paratope_contact_logits=paratope_contact.detach().clone().index_fill(
            1, torch.tensor([1]), -500.0
        ),
        antibody_mask=antibody_mask,
        epitope_unary_logits=epitope_unary.detach().clone().index_fill(
            1, torch.tensor([2]), -500.0
        ),
        epitope_contact_logits=epitope_contact.detach().clone().index_fill(
            1, torch.tensor([2]), 500.0
        ),
        antigen_mask=antigen_mask,
    )[0]
    torch.testing.assert_close(total.detach(), masked_values_changed)

    total.backward()
    for logits in (
        paratope_unary,
        paratope_contact,
        epitope_unary,
        epitope_contact,
    ):
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


def test_pair_node_consistency_objective_is_fail_closed_and_uses_unary_heads() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        FrameContactLossWeights,
        FrameContactObjective,
    )

    output, batch = _outputs_and_batch()
    weights = FrameContactLossWeights(
        contact=0.0,
        paratope=0.0,
        epitope=0.0,
        specificity=0.0,
        pair_node_consistency=1.0,
    )
    objective = FrameContactObjective(weights=weights)
    with pytest.raises(ValueError, match="pair-node consistency requires"):
        objective(output, batch)

    output.update(
        {
            "paratope_unary_logits": torch.tensor(
                [[0.4, -0.8]], requires_grad=True
            ),
            "paratope_contact_logits": torch.tensor(
                [[0.4, -0.8]], requires_grad=True
            ),
            "epitope_unary_logits": torch.tensor(
                [[-0.3, 0.9, -1.2]], requires_grad=True
            ),
            "epitope_contact_logits": torch.tensor(
                [[-0.3, 0.9, -1.2]], requires_grad=True
            ),
        }
    )
    # Final residue outputs deliberately disagree; consistency must compare the
    # independent unary heads with contact-derived marginals, not hybrid heads.
    output["paratope_logits"] = torch.full((1, 2), 9.0, requires_grad=True)
    output["epitope_logits"] = torch.full((1, 3), -9.0, requires_grad=True)
    result = objective(output, batch)
    torch.testing.assert_close(
        result["pair_node_consistency_loss"],
        torch.zeros(()),
        atol=1e-8,
        rtol=0.0,
    )
    torch.testing.assert_close(result["loss"], result["pair_node_consistency_loss"])


pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("sklearn", "torch")),
    reason="FrameContact metric dependencies are unavailable",
)


def _outputs_and_batch():
    import torch

    contact_logits = torch.tensor(
        [[[3.0, -2.0, -1.0], [-2.0, 2.0, -3.0]]], requires_grad=True
    )
    pair_labels = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    pair_mask = torch.ones_like(pair_labels, dtype=torch.bool)
    output = {
        "contact_logits": contact_logits,
        "contact_interaction_logits": contact_logits,
        "pair_candidate_mask": pair_mask,
        "paratope_logits": torch.tensor([[1.5, -0.5]], requires_grad=True),
        "epitope_logits": torch.tensor([[1.0, 0.5, -1.0]], requires_grad=True),
        "antibody_mask": torch.ones(1, 2, dtype=torch.bool),
        "antigen_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    batch = {
        "pair_labels": pair_labels,
        "antibody_labels": torch.tensor([[1.0, 0.0]]),
        "antigen_labels": torch.tensor([[1.0, 1.0, 0.0]]),
    }
    return output, batch


def test_multitask_objective_uses_ranking_not_fake_wrong_pair_labels() -> None:
    import torch

    from interaction_first.framecontact_objective import FrameContactObjective

    cognate, batch = _outputs_and_batch()
    wrong = {
        "contact_logits": torch.full((1, 2, 3), -1.0, requires_grad=True),
        "contact_interaction_logits": torch.full(
            (1, 2, 3), -1.0, requires_grad=True
        ),
        "pair_candidate_mask": torch.ones(1, 2, 3, dtype=torch.bool),
    }
    result = FrameContactObjective(score_fraction=0.5)(
        cognate, batch, wrong_partner_output=wrong
    )
    assert torch.isfinite(result["loss"])
    assert result["mean_compatibility_margin"].item() > 0
    result["loss"].backward()
    assert cognate["contact_logits"].grad is not None
    assert wrong["contact_interaction_logits"].grad is not None
    assert float(wrong["contact_interaction_logits"].grad.abs().sum()) > 0


def test_specificity_ranking_ignores_single_chain_unary_shortcuts() -> None:
    import torch

    from interaction_first.framecontact_objective import FrameContactObjective

    cognate, batch = _outputs_and_batch()
    cognate["contact_logits"] = torch.full((1, 2, 3), 20.0, requires_grad=True)
    cognate["contact_interaction_logits"] = torch.full(
        (1, 2, 3), -2.0, requires_grad=True
    )
    wrong = {
        "contact_logits": torch.full((1, 2, 3), -20.0, requires_grad=True),
        "contact_interaction_logits": torch.full(
            (1, 2, 3), 2.0, requires_grad=True
        ),
        "pair_candidate_mask": torch.ones(1, 2, 3, dtype=torch.bool),
    }
    result = FrameContactObjective(score_fraction=0.5)(
        cognate, batch, wrong_partner_output=wrong
    )
    assert result["specificity_score_source"] == "interaction_only"
    assert result["mean_compatibility_margin"].item() < 0
    result["specificity_loss"].backward()
    assert cognate["contact_interaction_logits"].grad is not None
    assert wrong["contact_interaction_logits"].grad is not None
    assert cognate["contact_logits"].grad is None
    assert wrong["contact_logits"].grad is None


def test_contact_bottleneck_specificity_uses_direct_complex_cosine() -> None:
    import torch

    from interaction_first.framecontact_objective import FrameContactObjective

    cognate, batch = _outputs_and_batch()
    cognate_score = torch.tensor([0.8], requires_grad=True)
    wrong_score = torch.tensor([-0.2], requires_grad=True)
    cognate["contact_bottleneck_compatibility"] = cognate_score
    wrong = {
        "contact_logits": torch.zeros(1, 2, 3, requires_grad=True),
        "contact_interaction_logits": torch.full(
            (1, 2, 3), 100.0, requires_grad=True
        ),
        "pair_candidate_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "contact_bottleneck_compatibility": wrong_score,
    }
    result = FrameContactObjective(
        specificity_score_source="contact_bottleneck_cosine",
        specificity_loss_mode="listwise_contrastive",
    )(cognate, batch, wrong_partner_output=wrong)
    torch.testing.assert_close(
        result["mean_compatibility_margin"], torch.tensor(1.0)
    )
    result["specificity_loss"].backward()
    assert cognate_score.grad is not None and float(cognate_score.grad.abs()) > 0
    assert wrong_score.grad is not None and float(wrong_score.grad.abs()) > 0
    assert wrong["contact_interaction_logits"].grad is None


def test_contact_bottleneck_specificity_fails_closed_without_score() -> None:
    from interaction_first.framecontact_objective import FrameContactObjective

    cognate, batch = _outputs_and_batch()
    objective = FrameContactObjective(
        specificity_score_source="contact_bottleneck_cosine"
    )
    with pytest.raises(ValueError, match="contact_bottleneck_compatibility"):
        objective(cognate, batch, wrong_partner_output=_wrong_output(-1.0))


def test_reciprocal_specificity_objective_averages_both_directions() -> None:
    import torch

    from interaction_first.framecontact_objective import FrameContactObjective

    cognate, batch = _outputs_and_batch()
    cognate["contact_interaction_logits"] = cognate["contact_logits"]
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    wrong_outputs = {
        "fixed_antibody_wrong_antigen": {
            "contact_logits": torch.full((1, 2, 3), -2.0),
            "contact_interaction_logits": torch.full(
                (1, 2, 3), -2.0, requires_grad=True
            ),
            "pair_candidate_mask": mask,
        },
        "fixed_antigen_wrong_antibody": {
            "contact_logits": torch.full((1, 2, 3), -1.0),
            "contact_interaction_logits": torch.full(
                (1, 2, 3), -1.0, requires_grad=True
            ),
            "pair_candidate_mask": mask,
        },
    }
    result = FrameContactObjective(score_fraction=0.5)(
        cognate, batch, wrong_partner_outputs=wrong_outputs
    )
    assert set(result["specificity_margin_by_direction"]) == set(wrong_outputs)
    assert result["mean_compatibility_margin"].item() > 0
    result["loss"].backward()
    for output in wrong_outputs.values():
        assert output["contact_interaction_logits"].grad is not None


def _wrong_output(value: float):
    import torch

    logits = torch.full((1, 2, 3), value, requires_grad=True)
    return {
        "contact_logits": logits,
        "contact_interaction_logits": logits,
        "pair_candidate_mask": torch.ones(1, 2, 3, dtype=torch.bool),
    }


def test_listwise_specificity_uses_every_negative_and_is_permutation_invariant() -> None:
    import torch

    from interaction_first.framecontact_objective import FrameContactObjective

    first_cognate, batch = _outputs_and_batch()
    first_cognate["contact_interaction_logits"] = first_cognate["contact_logits"]
    first_negatives = [_wrong_output(value) for value in (-2.0, 0.0, 1.0)]
    objective = FrameContactObjective(
        score_fraction=0.5,
        specificity_loss_mode="listwise_contrastive",
        specificity_temperature=0.5,
    )
    first = objective(
        first_cognate,
        batch,
        wrong_partner_outputs={"fixed_antibody_wrong_antigen": first_negatives},
    )
    first["specificity_loss"].backward()
    assert first["specificity_negative_count_by_direction"] == {
        "fixed_antibody_wrong_antigen": 3
    }
    for output in first_negatives:
        gradient = output["contact_interaction_logits"].grad
        assert gradient is not None
        assert float(gradient.abs().sum()) > 0

    second_cognate, second_batch = _outputs_and_batch()
    second_cognate["contact_interaction_logits"] = second_cognate["contact_logits"]
    second = objective(
        second_cognate,
        second_batch,
        wrong_partner_outputs={
            "fixed_antibody_wrong_antigen": [
                _wrong_output(value) for value in (1.0, -2.0, 0.0)
            ]
        },
    )
    torch.testing.assert_close(first["specificity_loss"], second["specificity_loss"])


def test_listwise_specificity_loss_decreases_when_cognate_separates() -> None:
    from interaction_first.framecontact_objective import FrameContactObjective

    objective = FrameContactObjective(
        score_fraction=1.0,
        specificity_loss_mode="listwise_contrastive",
    )
    low_cognate, low_batch = _outputs_and_batch()
    low_cognate["contact_interaction_logits"] = _wrong_output(0.0)[
        "contact_interaction_logits"
    ]
    high_cognate, high_batch = _outputs_and_batch()
    high_cognate["contact_interaction_logits"] = _wrong_output(3.0)[
        "contact_interaction_logits"
    ]
    negatives = [_wrong_output(value) for value in (0.0, 1.0, 2.0)]
    low = objective(
        low_cognate,
        low_batch,
        wrong_partner_outputs={"direction": negatives},
    )
    high = objective(
        high_cognate,
        high_batch,
        wrong_partner_outputs={"direction": negatives},
    )
    assert high["specificity_loss"] < low["specificity_loss"]
    assert high["specificity_margin_by_direction"]["direction"] > low[
        "specificity_margin_by_direction"
    ]["direction"]


def test_listwise_k1_temperature_one_matches_pairwise_softplus() -> None:
    import torch

    from interaction_first.framecontact_objective import FrameContactObjective

    pairwise_cognate, pairwise_batch = _outputs_and_batch()
    pairwise_cognate["contact_interaction_logits"] = pairwise_cognate[
        "contact_logits"
    ]
    listwise_cognate, listwise_batch = _outputs_and_batch()
    listwise_cognate["contact_interaction_logits"] = listwise_cognate[
        "contact_logits"
    ]
    pairwise_wrong = _wrong_output(1.25)
    listwise_wrong = _wrong_output(1.25)
    pairwise = FrameContactObjective(
        score_fraction=0.5,
        specificity_loss_mode="pairwise_softplus",
    )(
        pairwise_cognate,
        pairwise_batch,
        wrong_partner_outputs={"direction": [pairwise_wrong]},
    )
    listwise = FrameContactObjective(
        score_fraction=0.5,
        specificity_loss_mode="listwise_contrastive",
        specificity_temperature=1.0,
    )(
        listwise_cognate,
        listwise_batch,
        wrong_partner_outputs={"direction": [listwise_wrong]},
    )
    torch.testing.assert_close(
        pairwise["specificity_loss"], listwise["specificity_loss"]
    )
    pairwise["specificity_loss"].backward()
    listwise["specificity_loss"].backward()
    torch.testing.assert_close(
        pairwise_wrong["contact_interaction_logits"].grad,
        listwise_wrong["contact_interaction_logits"].grad,
    )


def test_smooth_max_direction_aggregation_emphasizes_the_worse_direction() -> None:
    from interaction_first.framecontact_objective import FrameContactObjective

    cognate, batch = _outputs_and_batch()
    cognate["contact_interaction_logits"] = cognate["contact_logits"]
    wrong_outputs = {
        "easy": [_wrong_output(-5.0), _wrong_output(-4.0)],
        "hard": [_wrong_output(2.0), _wrong_output(3.0)],
    }
    mean = FrameContactObjective(
        score_fraction=0.5,
        specificity_loss_mode="listwise_contrastive",
        specificity_direction_aggregation="mean",
    )(cognate, batch, wrong_partner_outputs=wrong_outputs)
    smooth_max = FrameContactObjective(
        score_fraction=0.5,
        specificity_loss_mode="listwise_contrastive",
        specificity_direction_aggregation="smooth_max",
        specificity_direction_temperature=0.05,
    )(cognate, batch, wrong_partner_outputs=wrong_outputs)
    assert smooth_max["specificity_loss"] > mean["specificity_loss"]


def test_balanced_loss_handles_a_real_zero_contact_complex_without_fabrication() -> None:
    import torch

    from interaction_first.framecontact_objective import balanced_binary_loss_per_complex

    logits = torch.tensor([[[-1.0, 0.0, 1.0]]], requires_grad=True)
    labels = torch.zeros_like(logits)
    mask = torch.ones_like(labels, dtype=torch.bool)
    loss = balanced_binary_loss_per_complex(logits, labels, mask)
    expected = torch.nn.functional.softplus(logits).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad is not None


def _edm_backbone(ca_coordinates):
    import torch

    ca = torch.as_tensor(ca_coordinates, dtype=torch.float32)
    backbone = torch.zeros(1, ca.shape[0], 3, 3, dtype=torch.float32)
    backbone[0, :, 0, :] = ca + torch.tensor([-1.0, 0.0, 0.0])
    backbone[0, :, 1, :] = ca
    backbone[0, :, 2, :] = ca + torch.tensor([1.0, 0.0, 0.0])
    return backbone


def test_local_edm_loss_accepts_a_true_joint_3d_embedding() -> None:
    import torch

    from interaction_first.framecontact_objective import local_edm_realizability_loss

    antibody_ca = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    antigen_ca = torch.tensor([[0.0, 2.0, 1.0], [2.0, 2.0, 1.0]])
    predicted = torch.cdist(antibody_ca, antigen_ca)[None].requires_grad_(True)
    loss = local_edm_realizability_loss(
        predicted,
        torch.ones_like(predicted, dtype=torch.bool),
        _edm_backbone(antibody_ca),
        _edm_backbone(antigen_ca),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert loss.item() < 1e-10
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_local_edm_loss_rejects_constant_cross_distance_triangle_escape() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        local_edm_realizability_loss,
        sparse_triangle_inequality_loss,
    )

    antibody_ca = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    antigen_ca = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    antibody = _edm_backbone(antibody_ca)
    antigen = _edm_backbone(antigen_ca)
    predicted = torch.ones(1, 2, 2, requires_grad=True)
    mask = torch.ones_like(predicted, dtype=torch.bool)
    active = torch.ones(1, 2, dtype=torch.bool)
    chain_ids = torch.zeros(1, 2, dtype=torch.long)

    triangle = sparse_triangle_inequality_loss(
        predicted,
        mask,
        antibody,
        antigen,
        chain_ids,
        chain_ids,
        active,
        active,
        neighbor_radius=3.0,
        neighbor_k=1,
        slack=0.0,
    )
    edm = local_edm_realizability_loss(
        predicted, mask, antibody, antigen, active, active
    )

    assert triangle.item() == 0.0
    assert edm.item() > 1e-4
    edm.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum().item() > 0


def test_local_edm_loss_keeps_spectral_path_in_fp32_under_autocast() -> None:
    import torch

    from interaction_first.framecontact_objective import local_edm_realizability_loss

    antibody_ca = torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    antigen_ca = torch.tensor([[0.0, 4.0, 0.0], [3.8, 4.0, 0.0]])
    antibody = _edm_backbone(antibody_ca)
    antigen = _edm_backbone(antigen_ca)
    cross = torch.cdist(antibody_ca, antigen_ca).unsqueeze(0).requires_grad_(True)
    pair_mask = torch.ones_like(cross, dtype=torch.bool)
    active = torch.ones(1, 2, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = local_edm_realizability_loss(
            cross,
            pair_mask,
            antibody,
            antigen,
            active,
            active,
        )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert cross.grad is not None
    assert torch.isfinite(cross.grad).all()


def test_validation_temperature_fitting_reduces_overconfident_nll() -> None:
    import torch

    from interaction_first.framecontact_objective import ValidationTemperatureScaler

    logits = torch.tensor([[8.0, 7.0, -7.0, -8.0]])
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    before = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    scaler = ValidationTemperatureScaler()
    temperature = scaler.fit(logits, labels, mask, max_iter=40)
    after = torch.nn.functional.binary_cross_entropy_with_logits(
        scaler(logits), labels
    )
    assert temperature > 1.0
    assert after < before


def test_binary_metrics_include_discrimination_and_calibration() -> None:
    from interaction_first.framecontact_metrics import binary_classification_metrics

    result = binary_classification_metrics(
        [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], threshold=0.5
    )
    assert result["auprc"] == 1.0
    assert result["auroc"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["predicted_positive_count"] == 2
    assert result["predicted_positive_ratio"] == 0.5
    assert result["brier"] < 0.03
    assert result["ece_15"] > 0
    assert len(result["reliability_bins_15"]) == 15
    assert sum(value["count"] for value in result["reliability_bins_15"]) == 4


def test_logistic_calibration_regression_is_explicitly_opt_in() -> None:
    from interaction_first.framecontact_metrics import binary_classification_metrics

    probabilities = np.tile([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], 20)
    labels = np.tile([0, 0, 1, 0, 1, 0, 1, 1], 20)
    default = binary_classification_metrics(labels, probabilities, threshold=0.5)
    assert "calibration_slope" not in default
    final = binary_classification_metrics(
        labels,
        probabilities,
        threshold=0.5,
        include_calibration_regression=True,
    )
    assert np.isfinite(final["calibration_intercept"])
    assert np.isfinite(final["calibration_slope"])
    assert final["calibration_regression_converged"] is True


def test_ranked_pair_metrics_freeze_fixed_and_lmin_cutoffs() -> None:
    from interaction_first.framecontact_metrics import ranked_pair_metrics

    labels = np.zeros(100, dtype=np.int8)
    labels[[0, 2, 4, 80]] = 1
    probabilities = np.linspace(1.0, 0.0, 100)
    result = ranked_pair_metrics(
        labels,
        probabilities,
        antibody_length=30,
        antigen_length=80,
    )
    assert result["length_definition"].startswith("Lmin")
    assert result["cutoffs"]["10"]["precision"] == pytest.approx(0.3)
    assert result["cutoffs"]["10"]["recall"] == pytest.approx(0.75)
    assert result["cutoffs"]["lmin_over_10"]["requested_k"] == 3
    assert result["cutoffs"]["lmin_over_5"]["requested_k"] == 6
    assert result["cutoffs"]["lmin"]["requested_k"] == 30


def test_ranked_pair_metrics_rejects_missing_positive_contact() -> None:
    from interaction_first.framecontact_metrics import ranked_pair_metrics

    with pytest.raises(ValueError, match="at least one positive"):
        ranked_pair_metrics(
            [0, 0],
            [0.2, 0.1],
            antibody_length=2,
            antigen_length=2,
        )


def test_positive_required_average_precision_marks_zero_positive_unit_ineligible() -> None:
    from interaction_first.framecontact_metrics import (
        positive_required_average_precision,
    )

    ineligible = positive_required_average_precision([0, 0], [0.9, 0.1])
    assert ineligible == {
        "eligible": False,
        "value": None,
        "ineligible_reason": "zero_positive_labels",
        "positive_count": 0,
        "candidate_count": 2,
    }

    eligible = positive_required_average_precision([1, 0], [0.9, 0.1])
    assert eligible["eligible"] is True
    assert eligible["value"] == 1.0
    assert eligible["positive_count"] == 1


def test_partner_retrieval_is_cluster_weighted_and_bootstrapped() -> None:
    from interaction_first.framecontact_metrics import (
        bootstrap_cluster_weighted_margin,
        partner_retrieval_metrics,
    )

    result = partner_retrieval_metrics(
        [0.9, 0.6, 0.8],
        [[0.2, 0.3], [0.7, 0.1], [0.4, 0.5]],
        ["large_cluster", "large_cluster", "small_cluster"],
    )
    assert result["top1_retrieval_accuracy"] == pytest.approx(2 / 3)
    assert result["antigen_cluster_count"] == 2
    cluster_values = result["cluster_margins"]
    expected = np.mean(list(cluster_values.values()))
    assert result["cluster_weighted_cognate_compatibility_margin"] == pytest.approx(
        expected
    )
    interval = bootstrap_cluster_weighted_margin(
        cluster_values, replicates=1000, seed=59
    )
    assert interval["estimate"] == pytest.approx(expected)
    assert interval["ci95_lower"] <= interval["estimate"] <= interval["ci95_upper"]


def test_group_bootstrap_does_not_let_large_family_dominate() -> None:
    from interaction_first.framecontact_metrics import bootstrap_group_weighted_values

    values = {**{f"large-{i}": 1.0 for i in range(9)}, "small-0": -1.0}
    groups = {**{f"large-{i}": "large" for i in range(9)}, "small-0": "small"}
    interval = bootstrap_group_weighted_values(
        values, groups, replicates=1000, seed=11
    )
    assert interval["estimate"] == pytest.approx(0.0)
    assert interval["eligible_complex_count"] == 10
    assert interval["group_count"] == 2
    assert interval["ci95_lower"] <= 0 <= interval["ci95_upper"]


def test_distogram_loss_and_expected_distance_share_registered_bins() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        distogram_cross_entropy_per_complex,
        expected_distance_from_distogram,
    )

    boundaries = torch.tensor([2.0, 4.5, 8.0])
    distances = torch.tensor([[[1.0, 3.0, 7.0, 12.0]]])
    mask = torch.ones_like(distances, dtype=torch.bool)
    correct = torch.full((1, 1, 4, 4), -8.0)
    target = torch.bucketize(distances, boundaries)
    correct.scatter_(-1, target[..., None], 8.0)
    wrong = correct.flip(-1)
    assert distogram_cross_entropy_per_complex(
        correct, distances, mask, boundaries
    ) < distogram_cross_entropy_per_complex(wrong, distances, mask, boundaries)
    expected = expected_distance_from_distogram(correct, boundaries)
    assert expected.shape == distances.shape
    assert torch.isfinite(expected).all()


def test_sparse_triangle_loss_detects_reverse_triangle_violation() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        sparse_triangle_inequality_loss,
    )

    antibody_backbone = torch.zeros(1, 2, 3, 3)
    antibody_backbone[0, 1, :, 0] = 1.0
    antigen_backbone = torch.zeros(1, 1, 3, 3)
    pair_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    antibody_active = torch.ones(1, 2, dtype=torch.bool)
    antigen_active = torch.ones(1, 1, dtype=torch.bool)
    chain_ab = torch.zeros(1, 2, dtype=torch.long)
    chain_ag = torch.zeros(1, 1, dtype=torch.long)
    consistent = sparse_triangle_inequality_loss(
        torch.tensor([[[3.0], [2.0]]]),
        pair_mask,
        antibody_backbone,
        antigen_backbone,
        chain_ab,
        chain_ag,
        antibody_active,
        antigen_active,
        neighbor_radius=2.0,
        neighbor_k=1,
        slack=0.0,
    )
    violated = sparse_triangle_inequality_loss(
        torch.tensor([[[5.0], [1.0]]]),
        pair_mask,
        antibody_backbone,
        antigen_backbone,
        chain_ab,
        chain_ag,
        antibody_active,
        antigen_active,
        neighbor_radius=2.0,
        neighbor_k=1,
        slack=0.0,
    )
    assert consistent.item() == pytest.approx(0.0)
    assert violated.item() > 0.0


def test_far_patch_triangle_explicitly_pushes_a_distant_patch() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        far_patch_triangle_exclusion_loss,
    )

    # One CDR residue, one selected antigen-patch residue at x=0 and one
    # distant antigen surface residue at x=30.  Pulling both predicted
    # inter-chain distances to 2 A violates 2 + 2 >= 30; keeping the distant
    # prediction at 29 A satisfies the same pose-free constraint.
    antigen_backbone = torch.zeros(1, 2, 3, 3)
    antigen_backbone[0, 1, :, 0] = 30.0
    pair_mask = torch.ones(1, 1, 2, dtype=torch.bool)
    antibody_active = torch.ones(1, 1, dtype=torch.bool)
    antigen_patch = torch.tensor([[True, False]])
    antigen_surface = torch.tensor([[True, True]])
    violated_distances = torch.tensor([[[2.0, 2.0]]], requires_grad=True)
    violated = far_patch_triangle_exclusion_loss(
        violated_distances,
        pair_mask,
        antigen_backbone,
        antibody_active,
        antigen_patch,
        antigen_surface,
        minimum_patch_separation=20.0,
        far_residue_k=1,
        slack=0.0,
    )
    satisfied = far_patch_triangle_exclusion_loss(
        torch.tensor([[[2.0, 29.0]]]),
        pair_mask,
        antigen_backbone,
        antibody_active,
        antigen_patch,
        antigen_surface,
        minimum_patch_separation=20.0,
        far_residue_k=1,
        slack=0.0,
    )
    assert violated.item() == pytest.approx(26.0**2)
    assert satisfied.item() == pytest.approx(0.0)
    violated.backward()
    assert violated_distances.grad is not None
    assert violated_distances.grad[0, 0, 1].item() < 0.0


def test_far_patch_triangle_is_invariant_to_antigen_rigid_transform() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        far_patch_triangle_exclusion_loss,
    )

    antigen_backbone = torch.zeros(1, 3, 3, 3)
    antigen_backbone[0, 1, :, 0] = 25.0
    antigen_backbone[0, 2, :, 1] = 30.0
    pair_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    antibody_active = torch.ones(1, 1, dtype=torch.bool)
    antigen_patch = torch.tensor([[True, False, False]])
    antigen_surface = torch.ones(1, 3, dtype=torch.bool)
    predicted = torch.tensor([[[3.0, 4.0, 5.0]]])
    before = far_patch_triangle_exclusion_loss(
        predicted,
        pair_mask,
        antigen_backbone,
        antibody_active,
        antigen_patch,
        antigen_surface,
        minimum_patch_separation=20.0,
        far_residue_k=2,
        slack=0.5,
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = antigen_backbone @ rotation.T + torch.tensor([7.0, -4.0, 2.0])
    after = far_patch_triangle_exclusion_loss(
        predicted,
        pair_mask,
        transformed,
        antibody_active,
        antigen_patch,
        antigen_surface,
        minimum_patch_separation=20.0,
        far_residue_k=2,
        slack=0.5,
    )
    torch.testing.assert_close(before, after)


def test_far_patch_triangle_returns_zero_when_no_far_patch_exists() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        far_patch_triangle_exclusion_loss,
    )

    predicted = torch.full((1, 1, 2), 4.0, requires_grad=True)
    pair_mask = torch.ones_like(predicted, dtype=torch.bool)
    antigen_backbone = torch.zeros(1, 2, 3, 3)
    antigen_backbone[0, 1, :, 0] = 5.0
    value = far_patch_triangle_exclusion_loss(
        predicted,
        pair_mask,
        antigen_backbone,
        torch.tensor([[True]]),
        torch.tensor([[True, False]]),
        torch.tensor([[True, True]]),
        minimum_patch_separation=20.0,
        far_residue_k=1,
        slack=0.0,
    )
    assert value.item() == 0.0
    value.backward()
    assert predicted.grad is not None
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))


def test_triangle_patch_seed_comes_from_intra_chain_epitope_unary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import interaction_first.framecontact_objective as module

    output, batch = _outputs_and_batch()
    output["distogram_logits"] = torch.zeros(1, 2, 3, 4, requires_grad=True)
    output["distance_bin_boundaries"] = torch.tensor([2.0, 4.5, 8.0])
    output["epitope_unary_logits"] = torch.tensor([[-9.0, 2.0, 1.0]])
    output["antigen_contact_propensity_logits"] = torch.tensor([[8.0, -3.0, -4.0]])
    batch.update(
        {
            "pair_distances": torch.full((1, 2, 3), 6.0),
            "pair_distance_mask": torch.ones(1, 2, 3, dtype=torch.bool),
            "antibody_backbone": torch.zeros(1, 2, 3, 3),
            "antigen_backbone": torch.zeros(1, 3, 3, 3),
            "antibody_chain_ids": torch.zeros(1, 2, dtype=torch.long),
            "antigen_chain_ids": torch.zeros(1, 3, dtype=torch.long),
            "antibody_cdr_mask": torch.ones(1, 2, dtype=torch.bool),
            "antigen_surface_mask": torch.ones(1, 3, dtype=torch.bool),
        }
    )
    captured = {}

    def fake_patch(seed_logits, *args, **kwargs):
        captured["seed_logits"] = seed_logits
        return torch.ones(1, 3, dtype=torch.bool)

    def fake_triangle(predicted_distances, *args, **kwargs):
        return predicted_distances.sum() * 0.0

    monkeypatch.setattr(module, "select_local_antigen_patch", fake_patch)
    monkeypatch.setattr(module, "sparse_triangle_inequality_loss", fake_triangle)
    weights = module.FrameContactLossWeights(
        contact=0.0,
        paratope=0.0,
        epitope=0.0,
        specificity=0.0,
        distogram=0.0,
        triangle=1.0,
    )
    module.FrameContactObjective(weights=weights)(output, batch)
    torch.testing.assert_close(captured["seed_logits"], output["epitope_unary_logits"])


def test_edm_objective_reuses_prediction_routed_surface_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import interaction_first.framecontact_objective as module

    output, batch = _outputs_and_batch()
    output["distogram_logits"] = torch.zeros(1, 2, 3, 4, requires_grad=True)
    output["distance_bin_boundaries"] = torch.tensor([2.0, 4.5, 8.0])
    output["epitope_unary_logits"] = torch.tensor([[-2.0, 3.0, 1.0]])
    batch.update(
        {
            "antibody_backbone": torch.zeros(1, 2, 3, 3),
            "antigen_backbone": torch.zeros(1, 3, 3, 3),
            "antibody_cdr_mask": torch.ones(1, 2, dtype=torch.bool),
            "antigen_surface_mask": torch.ones(1, 3, dtype=torch.bool),
        }
    )
    captured = {}

    def fake_patch(seed_logits, *args, **kwargs):
        captured["seed_logits"] = seed_logits
        return torch.ones(1, 3, dtype=torch.bool)

    def fake_edm(predicted_distances, *args, **kwargs):
        captured["predicted_distances"] = predicted_distances
        return predicted_distances.square().mean()

    monkeypatch.setattr(module, "select_local_antigen_patch", fake_patch)
    monkeypatch.setattr(module, "local_edm_realizability_loss", fake_edm)
    weights = module.FrameContactLossWeights(
        contact=0.0,
        paratope=0.0,
        epitope=0.0,
        specificity=0.0,
        edm=1.0,
    )
    result = module.FrameContactObjective(weights=weights)(output, batch)
    torch.testing.assert_close(captured["seed_logits"], output["epitope_unary_logits"])
    torch.testing.assert_close(result["loss"], result["edm_loss"])
    result["loss"].backward()
    assert output["distogram_logits"].grad is not None
