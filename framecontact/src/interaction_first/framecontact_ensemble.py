"""Validation-frozen probability ensemble for FrameContact evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - optional import boundary
    torch = None
    nn = None


_ModuleBase = nn.Module if nn is not None else object


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the FrameContact ensemble")
    return torch


def _positive_temperature(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _probability_logit(probability: Any) -> Any:
    epsilon = torch.finfo(probability.dtype).eps
    clipped = probability.clamp(epsilon, 1.0 - epsilon)
    return torch.logit(clipped)


class CalibratedProbabilityEnsemble(_ModuleBase):
    """Average member probabilities under a predeclared, auditable rule.

    Cognate contact and residue logits are calibrated with each member's
    validation-only temperature before elementwise probability averaging.
    Interaction-only logits deliberately have no unary or calibration path:
    their sigmoid probabilities are averaged directly before partner ranking.
    """

    def __init__(
        self,
        models: Sequence[Any],
        calibrations: Sequence[Mapping[str, float]],
    ) -> None:
        _require_torch()
        super().__init__()
        if len(models) < 2 or len(models) != len(calibrations):
            raise ValueError("ensemble models/calibrations must align and contain >=2 members")
        self.models = nn.ModuleList(models)
        normalized = []
        for index, calibration in enumerate(calibrations):
            normalized.append(
                {
                    "contact": _positive_temperature(
                        calibration["contact"], f"member {index} contact temperature"
                    ),
                    "paratope": _positive_temperature(
                        calibration["paratope"],
                        f"member {index} paratope temperature",
                    ),
                    "epitope": _positive_temperature(
                        calibration["epitope"], f"member {index} epitope temperature"
                    ),
                }
            )
        self.calibrations = tuple(normalized)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        if return_attention:
            raise ValueError(
                "Ensemble attention is undefined; inspect member diagnostics separately"
            )
        outputs = [model(batch, return_attention=False) for model in self.models]
        reference = outputs[0]
        for output in outputs[1:]:
            for mask_name in (
                "pair_candidate_mask",
                "antibody_mask",
                "antigen_mask",
            ):
                if not torch.equal(output[mask_name], reference[mask_name]):
                    raise ValueError("Ensemble members produced different evaluation masks")

        contact_probability = torch.stack(
            [
                torch.sigmoid(output["contact_logits"] / calibration["contact"])
                for output, calibration in zip(
                    outputs, self.calibrations, strict=True
                )
            ]
        ).mean(dim=0)
        interaction_probability = torch.stack(
            [torch.sigmoid(output["contact_interaction_logits"]) for output in outputs]
        ).mean(dim=0)
        paratope_probability = torch.stack(
            [
                torch.sigmoid(output["paratope_logits"] / calibration["paratope"])
                for output, calibration in zip(
                    outputs, self.calibrations, strict=True
                )
            ]
        ).mean(dim=0)
        epitope_probability = torch.stack(
            [
                torch.sigmoid(output["epitope_logits"] / calibration["epitope"])
                for output, calibration in zip(
                    outputs, self.calibrations, strict=True
                )
            ]
        ).mean(dim=0)
        return {
            "contact_logits": _probability_logit(contact_probability),
            "contact_interaction_logits": _probability_logit(
                interaction_probability
            ),
            "paratope_logits": _probability_logit(paratope_probability),
            "epitope_logits": _probability_logit(epitope_probability),
            "pair_candidate_mask": reference["pair_candidate_mask"],
            "antibody_mask": reference["antibody_mask"],
            "antigen_mask": reference["antigen_mask"],
            "self_attention_details": None,
            "cross_attention_details": None,
            "fusion_mode": "validation_frozen_probability_ensemble",
        }


__all__ = ["CalibratedProbabilityEnsemble"]
