"""Executable test of the ``bound_inter_chain_geometry_allowed = false`` contract.

``protocols/sabdab2_partner_geometry_r2_model_contract_v1.json`` forbids the R2
model from consuming bound inter-chain geometry, and
``interaction_first.r2_model_contract`` validates that the manifest says so. It
validates the manifest only. No check reads the model, so a change that feeds the
antibody-to-antigen relative pose into the network satisfies the contract
validator while breaking the property the contract exists to protect -- and the
relative pose in a bound complex is close to the contact label itself.

This module tests the property instead of the declaration, through the behaviour
that defines it:

    Moving one chain rigidly must not change any output.

A rigid transform of the antigen leaves every *intra*-antigen distance and angle
unchanged and alters only the antigen's pose relative to the antibody. A model
restricted to intra-chain geometry is therefore exactly invariant to it, while
any model reading inter-chain distance, displacement, or relative rotation is
not. The test is behavioural, so it does not depend on module names, import
paths, or how the geometry is spelled, and it keeps working against code written
after it.

**A pass is only meaningful once the geometry pathway is live.** Geometry biases
are zero-initialised so that training begins from the topology-only model, which
makes a freshly constructed network invariant to *every* backbone change,
including ones it should react to. The report therefore measures sensitivity
first, under a non-rigid perturbation, and returns ``INCONCLUSIVE`` rather than
a pass when the model ignores the backbone altogether.

Run in float64 for an exact statement: a 100 Å translation costs roughly 1e-6 of
float32 round-off, which is indistinguishable from a small genuine dependence.
The same model in double precision agrees to ~1e-15.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Mapping
from typing import Any

DEFAULT_TOLERANCE = 1e-9
DEFAULT_SENSITIVITY_SCALE = 0.5
DEFAULT_TRANSLATION = (120.0, -60.0, 35.0)


class InvarianceVerdict(enum.Enum):
    """Outcome of a partner-pose probe."""

    INVARIANT = "invariant"
    PARTNER_POSE_DEPENDENT = "partner_pose_dependent"
    INCONCLUSIVE = "inconclusive_no_backbone_sensitivity"


class PartnerPoseInvarianceError(RuntimeError):
    """Raised when a model fails or cannot support the invariance test."""


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise PartnerPoseInvarianceError("PyTorch is required") from error
    return torch


def random_rotation(seed: int, dtype: Any = None) -> Any:
    """Return a deterministic proper rotation matrix (``det == +1``)."""
    torch = _torch()
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(raw)
    if torch.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    return rotation if dtype is None else rotation.to(dtype)


def rigid_transform(
    backbone: Any,
    *,
    rotation: Any | None = None,
    translation: Any | None = None,
) -> Any:
    """Apply a rigid motion to a ``[batch, residue, atom, 3]`` backbone."""
    torch = _torch()
    if not torch.is_tensor(backbone) or backbone.ndim != 4 or backbone.shape[-1] != 3:
        raise PartnerPoseInvarianceError("backbone must have shape [batch, residue, atom, 3]")
    moved = backbone
    if rotation is not None:
        moved = moved @ rotation.to(moved.dtype).transpose(-1, -2)
    if translation is not None:
        offset = torch.as_tensor(translation, dtype=moved.dtype, device=moved.device)
        if offset.shape != (3,):
            raise PartnerPoseInvarianceError("translation must have three components")
        moved = moved + offset
    return moved


@dataclasses.dataclass(frozen=True)
class InvarianceReport:
    """Result of probing one chain's pose against a model's outputs."""

    verdict: InvarianceVerdict
    sensitivity: float
    deviations: Mapping[str, float]
    worst_output: Mapping[str, str]
    tolerance: float

    @property
    def max_deviation(self) -> float:
        """Largest deviation over every rigid perturbation."""
        return max(self.deviations.values(), default=0.0)

    def as_json(self) -> dict[str, object]:
        """Canonical payload for a receipt."""
        return {
            "verdict": self.verdict.value,
            "sensitivity": self.sensitivity,
            "deviations": dict(self.deviations),
            "worst_output_deviation": dict(self.worst_output),
            "max_deviation": self.max_deviation,
            "tolerance": self.tolerance,
        }


def _max_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[float, str]:
    torch = _torch()
    if set(left) != set(right):
        raise PartnerPoseInvarianceError("perturbation changed the set of model outputs")
    worst, where = 0.0, ""
    for name in sorted(left):
        first, second = left[name], right[name]
        if not torch.is_tensor(first) or not first.is_floating_point():
            continue
        if first.shape != second.shape:
            raise PartnerPoseInvarianceError(f"perturbation changed the shape of {name}")
        deviation = float((first - second).abs().max())
        if deviation > worst:
            worst, where = deviation, name
    return worst, where


def probe_partner_pose_invariance(
    run: Callable[[Any], Mapping[str, Any]],
    partner_backbone: Any,
    *,
    seed: int = 20260908,
    tolerance: float = DEFAULT_TOLERANCE,
    sensitivity_scale: float = DEFAULT_SENSITIVITY_SCALE,
    translation: tuple[float, float, float] = DEFAULT_TRANSLATION,
) -> InvarianceReport:
    """Probe whether ``run`` depends on one chain's pose relative to the other.

    ``run`` takes a replacement backbone for the moved chain and returns the
    model outputs. Everything else -- features, masks, the other chain -- is held
    fixed by the caller's closure.
    """
    torch = _torch()
    if not torch.is_tensor(partner_backbone):
        raise PartnerPoseInvarianceError("partner_backbone must be a tensor")
    if tolerance <= 0.0:
        raise PartnerPoseInvarianceError("tolerance must be positive")

    baseline = run(partner_backbone)
    if not isinstance(baseline, Mapping) or not baseline:
        raise PartnerPoseInvarianceError("run must return a non-empty mapping of outputs")

    # Sensitivity control: a non-rigid change must move the outputs, otherwise
    # the model is ignoring the backbone and invariance proves nothing.
    generator = torch.Generator().manual_seed(seed)
    jitter = torch.randn(
        partner_backbone.shape, generator=generator, dtype=torch.float64
    ).to(partner_backbone.dtype)
    sensitivity, _ = _max_difference(
        baseline, run(partner_backbone + sensitivity_scale * jitter)
    )

    rotation = random_rotation(seed, dtype=partner_backbone.dtype)
    perturbations = {
        "translation": {"translation": translation},
        "rotation": {"rotation": rotation},
        "rotation_and_translation": {"rotation": rotation, "translation": translation},
    }
    deviations: dict[str, float] = {}
    worst_output: dict[str, str] = {}
    for name, keyword in perturbations.items():
        moved = rigid_transform(partner_backbone, **keyword)
        deviation, output = _max_difference(baseline, run(moved))
        deviations[name] = deviation
        if output:
            worst_output[name] = output

    if sensitivity <= tolerance:
        verdict = InvarianceVerdict.INCONCLUSIVE
    elif max(deviations.values()) > tolerance:
        verdict = InvarianceVerdict.PARTNER_POSE_DEPENDENT
    else:
        verdict = InvarianceVerdict.INVARIANT

    return InvarianceReport(
        verdict=verdict,
        sensitivity=sensitivity,
        deviations=deviations,
        worst_output=worst_output,
        tolerance=tolerance,
    )


def assert_partner_pose_invariant(report: InvarianceReport) -> None:
    """Raise unless the probe proved invariance."""
    if report.verdict is InvarianceVerdict.INVARIANT:
        return
    if report.verdict is InvarianceVerdict.INCONCLUSIVE:
        raise PartnerPoseInvarianceError(
            "the model ignored the partner backbone entirely "
            f"(sensitivity {report.sensitivity:.3e} <= tolerance {report.tolerance:.3e}); "
            "activate the geometry pathway before asserting invariance, because an "
            "untrained zero-initialised bias is invariant to everything"
        )
    offenders = ", ".join(
        f"{name}={value:.3e}"
        for name, value in sorted(report.deviations.items())
        if value > report.tolerance
    )
    raise PartnerPoseInvarianceError(
        "model output changed when the partner chain was moved rigidly, so it "
        "consumes bound inter-chain geometry, which "
        "protocols/sabdab2_partner_geometry_r2_model_contract_v1.json forbids "
        f"(tolerance {report.tolerance:.3e}; {offenders})"
    )


def activate_zero_initialised_parameters(model: Any, seed: int = 20260908, scale: float = 0.3) -> int:
    """Give every all-zero parameter a deterministic value; return how many.

    Geometry biases ship zero-initialised so training starts from the
    topology-only model. Invariance measured before they are non-zero is
    vacuous, so a probe on an untrained network must perturb them first.
    """
    torch = _torch()
    generator = torch.Generator().manual_seed(seed)
    changed = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 0 or float(parameter.abs().max()) != 0.0:
                continue
            replacement = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float64
            ).to(parameter.dtype)
            parameter.copy_(replacement * scale)
            changed += 1
    return changed
