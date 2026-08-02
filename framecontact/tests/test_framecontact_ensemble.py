"""Contracts for validation-frozen FrameContact probability ensembling."""

from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


def _model(contact: float, interaction: float, paratope: float, epitope: float):
    import torch

    class FixedModel(torch.nn.Module):
        def forward(self, batch, *, return_attention=False):
            assert not return_attention
            pair_mask = batch["antibody_mask"][:, :, None] & batch[
                "antigen_mask"
            ][:, None, :]
            return {
                "contact_logits": torch.full(pair_mask.shape, contact),
                "contact_interaction_logits": torch.full(
                    pair_mask.shape, interaction
                ),
                "paratope_logits": torch.full(
                    batch["antibody_mask"].shape, paratope
                ),
                "epitope_logits": torch.full(
                    batch["antigen_mask"].shape, epitope
                ),
                "pair_candidate_mask": pair_mask,
                "antibody_mask": batch["antibody_mask"],
                "antigen_mask": batch["antigen_mask"],
            }

    return FixedModel()


def test_ensemble_averages_member_probabilities_after_validation_calibration() -> None:
    import torch

    from interaction_first.framecontact_ensemble import (
        CalibratedProbabilityEnsemble,
    )

    ensemble = CalibratedProbabilityEnsemble(
        [_model(2.0, 1.0, 3.0, 4.0), _model(-2.0, -1.0, -3.0, -4.0)],
        [
            {"contact": 2.0, "paratope": 3.0, "epitope": 4.0},
            {"contact": 2.0, "paratope": 3.0, "epitope": 4.0},
        ],
    )
    batch = {
        "antibody_mask": torch.ones(1, 2, dtype=torch.bool),
        "antigen_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    result = ensemble(batch)
    for key in (
        "contact_logits",
        "contact_interaction_logits",
        "paratope_logits",
        "epitope_logits",
    ):
        torch.testing.assert_close(result[key], torch.zeros_like(result[key]))


def test_ensemble_rejects_attention_and_invalid_calibration() -> None:
    import torch

    from interaction_first.framecontact_ensemble import (
        CalibratedProbabilityEnsemble,
    )

    models = [_model(0.0, 0.0, 0.0, 0.0), _model(0.0, 0.0, 0.0, 0.0)]
    with pytest.raises(ValueError, match="positive"):
        CalibratedProbabilityEnsemble(
            models,
            [
                {"contact": 0.0, "paratope": 1.0, "epitope": 1.0},
                {"contact": 1.0, "paratope": 1.0, "epitope": 1.0},
            ],
        )
    valid = CalibratedProbabilityEnsemble(
        models,
        [
            {"contact": 1.0, "paratope": 1.0, "epitope": 1.0},
            {"contact": 1.0, "paratope": 1.0, "epitope": 1.0},
        ],
    )
    with pytest.raises(ValueError, match="undefined"):
        valid(
            {
                "antibody_mask": torch.ones(1, 1, dtype=torch.bool),
                "antigen_mask": torch.ones(1, 1, dtype=torch.bool),
            },
            return_attention=True,
        )
