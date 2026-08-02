"""Contract tests for gauge-routed residue heads and residue-level specificity."""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _model(**overrides):
    from interaction_first.framecontact import FrameContactModel

    arguments = {
        "antibody_input_dim": 12,
        "antigen_input_dim": 10,
        "d_model": 16,
        "num_heads": 4,
        "num_self_layers": 1,
        "neighbor_k": 4,
        "neighbor_radius": None,
        "dropout": 0.0,
    }
    arguments.update(overrides)
    return FrameContactModel(**arguments).eval()


def _batch(batch_size: int = 2):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_framecontact import _batch as build

    return build(batch_size)


def test_the_gauge_never_changes_the_contact_prediction() -> None:
    """Enabling the gauge re-attributes evidence; it must not move the output."""

    import torch

    torch.manual_seed(20260802)
    raw = _model(pair_gauge_mode="raw")
    torch.manual_seed(20260802)
    gauged = _model(pair_gauge_mode="doubly_centered")

    batch = _batch()
    with torch.no_grad():
        raw_output = raw(batch)
        gauged_output = gauged(batch)

    assert torch.equal(raw_output["contact_logits"], gauged_output["contact_logits"])
    assert torch.equal(raw_output["paratope_logits"], gauged_output["paratope_logits"])
    assert raw_output["contact_gauge_interaction_logits"] is None
    assert gauged_output["contact_gauge_interaction_logits"] is not None


def test_gauge_hybrid_starts_exactly_at_the_unary_baseline() -> None:
    import torch

    torch.manual_seed(20260802)
    hybrid = _model(
        residue_output_mode="gauge_hybrid", pair_gauge_mode="doubly_centered"
    )
    torch.manual_seed(20260802)
    unary = _model(residue_output_mode="unary_only")

    batch = _batch()
    with torch.no_grad():
        hybrid_output = hybrid(batch)
        unary_output = unary(batch)

    assert torch.allclose(
        hybrid_output["epitope_logits"], unary_output["epitope_logits"], atol=1e-6
    )
    assert torch.allclose(
        hybrid_output["paratope_logits"], unary_output["paratope_logits"], atol=1e-6
    )


def test_gauge_marginal_is_nonnegative_and_vanishes_on_flat_evidence() -> None:
    """The gauge marginal measures concentration, not level.

    Because the partner-specific channel is centered along the marginalized
    axis, a flat interaction profile contributes exactly zero, and any profile
    contributes a nonnegative amount.
    """

    import torch

    from interaction_first.framecontact import masked_logmeanexp

    torch.manual_seed(20260802)
    model = _model(
        residue_output_mode="gauge_marginal_only", pair_gauge_mode="doubly_centered"
    )
    batch = _batch()
    with torch.no_grad():
        output = model(batch)

    mask = output["pair_candidate_mask"]
    epitope = output["epitope_gauge_logits"]
    assert float(epitope[output["antigen_mask"]].min()) >= -1e-5

    flat = torch.zeros_like(output["contact_gauge_interaction_logits"])
    assert torch.allclose(
        masked_logmeanexp(flat, mask, dim=1),
        torch.zeros_like(epitope),
        atol=1e-6,
    )


def test_gauge_routed_heads_require_an_identifiable_channel() -> None:
    with pytest.raises(ValueError, match="doubly_centered"):
        _model(residue_output_mode="gauge_hybrid", pair_gauge_mode="raw")


def test_gauge_specificity_source_works_for_both_pair_heads() -> None:
    """Binary and distogram heads must expose a comparable specificity channel.

    The distogram collapses bins through logsumexp, so its internal
    ``interaction`` term is not an additive component of the contact logits.
    The gauge is applied to the final map instead, which makes the two heads
    measurable on the same footing.
    """

    import torch

    from interaction_first.framecontact_objective import FrameContactObjective

    objective = FrameContactObjective(specificity_score_source="gauge_interaction")
    batch = _batch()
    for pair_output_mode in ("binary_contact", "distogram"):
        torch.manual_seed(20260802)
        model = _model(
            pair_output_mode=pair_output_mode, pair_gauge_mode="doubly_centered"
        )
        with torch.no_grad():
            output = model(batch)
        score = objective._specificity_score(output)
        assert tuple(score.shape) == (2,)
        assert bool(torch.isfinite(score).all())


def test_gauge_specificity_source_rejects_an_ungauged_model() -> None:
    import torch

    from interaction_first.framecontact_objective import FrameContactObjective

    objective = FrameContactObjective(specificity_score_source="gauge_interaction")
    torch.manual_seed(20260802)
    model = _model(pair_gauge_mode="raw")
    with torch.no_grad():
        output = model(_batch())

    with pytest.raises(ValueError, match="doubly_centered"):
        objective._specificity_score(output)


def test_residue_contrastive_loss_rewards_the_cognate_partner() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        residue_partner_contrastive_loss,
    )

    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    decoy = torch.zeros(1, 4, 2)

    winning = residue_partner_contrastive_loss(
        torch.tensor([[6.0, 0.0, 6.0, 0.0]]), decoy, labels, mask, margin=0.5
    )
    losing = residue_partner_contrastive_loss(
        torch.tensor([[-6.0, 0.0, -6.0, 0.0]]), decoy, labels, mask, margin=0.5
    )

    assert float(winning) < float(losing)
    assert float(winning) < 0.01


def test_residue_contrastive_loss_ignores_non_interface_residues() -> None:
    """Only labelled interface sites carry a partner-specificity requirement."""

    import torch

    from interaction_first.framecontact_objective import (
        residue_partner_contrastive_loss,
    )

    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    decoy = torch.zeros(1, 2, 1)

    baseline = residue_partner_contrastive_loss(
        torch.tensor([[2.0, 2.0]]), decoy, labels, mask
    )
    perturbed = residue_partner_contrastive_loss(
        torch.tensor([[2.0, -9.0]]), decoy, labels, mask
    )

    assert float(baseline) == pytest.approx(float(perturbed))


def test_residue_contrastive_loss_is_zero_without_labelled_positives() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        residue_partner_contrastive_loss,
    )

    loss = residue_partner_contrastive_loss(
        torch.tensor([[1.0, 2.0]]),
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert float(loss) == 0.0


def test_residue_contrastive_loss_validates_the_fixed_residue_axis() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        residue_partner_contrastive_loss,
    )

    labels = torch.ones(1, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="fixed-chain residue axis"):
        residue_partner_contrastive_loss(
            torch.zeros(1, 4), torch.zeros(1, 5, 2), labels, mask
        )


def test_objective_wires_both_contrastive_directions() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        FrameContactLossWeights,
        FrameContactObjective,
    )

    torch.manual_seed(20260802)
    model = _model(pair_gauge_mode="doubly_centered")
    batch = _batch()
    output = model(batch)
    labelled = dict(batch)
    labelled["pair_labels"] = (
        torch.rand(2, 5, 7) > 0.7
    ).float() * output["pair_candidate_mask"]
    labelled["antibody_labels"] = (torch.rand(2, 5) > 0.5).float() * batch[
        "antibody_mask"
    ]
    labelled["antigen_labels"] = (torch.rand(2, 7) > 0.5).float() * batch["antigen_mask"]

    decoys = {
        "fixed_antibody_wrong_antigen": [output],
        "fixed_antigen_wrong_antibody": [output],
    }
    objective = FrameContactObjective(
        weights=FrameContactLossWeights(
            specificity=0.0, residue_partner_contrastive=1.0
        ),
        specificity_score_source="gauge_interaction",
    )
    terms = objective(output, labelled, wrong_partner_outputs=decoys)

    assert set(terms["residue_partner_contrastive_by_side"]) == {"paratope", "epitope"}
    assert bool(torch.isfinite(terms["residue_partner_contrastive_loss"]))
    terms["loss"].backward()


def test_contrastive_weight_requires_a_usable_direction() -> None:
    import torch

    from interaction_first.framecontact_objective import (
        FrameContactLossWeights,
        FrameContactObjective,
    )

    torch.manual_seed(20260802)
    model = _model(pair_gauge_mode="doubly_centered")
    batch = _batch()
    output = model(batch)
    labelled = dict(batch)
    labelled["pair_labels"] = torch.zeros(2, 5, 7)
    labelled["antibody_labels"] = torch.zeros(2, 5)
    labelled["antigen_labels"] = torch.zeros(2, 7)

    objective = FrameContactObjective(
        weights=FrameContactLossWeights(
            specificity=0.0, residue_partner_contrastive=1.0
        ),
    )

    with pytest.raises(ValueError, match="fixed_antibody_wrong_antigen"):
        objective(output, labelled, wrong_partner_outputs={"other": [output]})
