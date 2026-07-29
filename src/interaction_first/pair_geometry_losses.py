"""Fail-closed objectives for explicit antibody-antigen pair geometry.

This prospective module does not authorize training, model selection, held-out
evaluation, or performance claims. Pair tensors are always
``[batch, antibody_residue, antigen_residue, ...]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None

from .pair_geometry_schema import (
    PAIR_ORIENTATION,
    FrozenGeometryMetadata,
    PairGeometrySchema,
)


def require_torch() -> None:
    """Fail with a focused message when PyTorch is unavailable."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for pair-geometry losses")


def _finite_nonnegative(value: Any, name: str) -> float:
    """Return a finite nonnegative scalar or reject the configuration."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return converted


def _finite_positive(value: Any, name: str) -> float:
    """Return a finite positive scalar or reject the configuration."""
    converted = _finite_nonnegative(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return converted


@dataclass(frozen=True)
class PairGeometryLossWeights:
    """Explicit weights for every prospective pair-geometry objective."""

    contact_bce: float = 1.0
    distance_ce: float = 1.0
    orientation_ce: float = 1.0
    epitope_consistency: float = 0.0
    paratope_consistency: float = 0.0
    matched_decoy_ranking: float = 0.0
    contact_brier: float = 0.0

    def validate(self) -> None:
        """Reject nonfinite, negative, or entirely disabled configurations."""
        validated = {
            name: _finite_nonnegative(value, name)
            for name, value in vars(self).items()
        }
        if sum(validated.values()) <= 0.0:
            raise ValueError("At least one pair-geometry loss weight must be positive")


def _require_float_tensor(
    value: Any,
    name: str,
    *,
    ndim: int | None = None,
) -> Any:
    """Validate a finite, nonempty floating-point tensor."""
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must use a floating-point dtype")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if value.numel() == 0 or any(int(size) == 0 for size in value.shape):
        raise ValueError(f"{name} cannot contain an empty dimension")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_boolean_mask(
    mask: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    device: Any,
    support_mask: Any | None = None,
) -> Any:
    """Validate a boolean supervision mask and optional support containment."""
    if not torch.is_tensor(mask):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must use a boolean dtype")
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if mask.device != device:
        raise ValueError(f"{name} must share the prediction device")
    if support_mask is not None and bool((mask & ~support_mask).any()):
        raise ValueError(f"{name} cannot select an unsupported entry")
    return mask


def _validate_binary_targets(
    targets: Any,
    mask: Any,
    reference: Any,
    name: str,
) -> Any:
    """Validate hard binary labels on supervised entries only."""
    if not torch.is_tensor(targets):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if tuple(targets.shape) != tuple(reference.shape):
        raise ValueError(f"{name} must align with its predictions")
    if targets.device != reference.device:
        raise ValueError(f"{name} must share the prediction device")
    integral_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        targets.dtype != torch.bool
        and not torch.is_floating_point(targets)
        and targets.dtype not in integral_dtypes
    ):
        raise ValueError(f"{name} must use a boolean, integer, or floating dtype")
    selected = targets[mask]
    if torch.is_floating_point(selected) and not bool(torch.isfinite(selected).all()):
        raise ValueError(f"{name} must be finite on supervised entries")
    if selected.numel() and not bool(((selected == 0) | (selected == 1)).all()):
        raise ValueError(f"{name} must contain hard binary labels on supervised entries")
    return selected.to(dtype=reference.dtype)


def _validate_class_targets(
    targets: Any,
    mask: Any,
    reference_logits: Any,
    name: str,
) -> Any:
    """Validate integer class labels on supervised entries only."""
    expected_shape = tuple(reference_logits.shape[:-1])
    if not torch.is_tensor(targets):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if tuple(targets.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if targets.device != reference_logits.device:
        raise ValueError(f"{name} must share the prediction device")
    if targets.dtype != torch.long:
        raise ValueError(f"{name} must use torch.long class indices")
    selected = targets[mask]
    class_count = int(reference_logits.shape[-1])
    if selected.numel() and bool(
        ((selected < 0) | (selected >= class_count)).any()
    ):
        raise ValueError(
            f"{name} must be in [0, {class_count - 1}] on supervised entries"
        )
    return selected


def _macro_masked_mean(
    per_entry_loss: Any,
    mask: Any,
    *,
    zero_source: Any,
) -> tuple[Any, int]:
    """Average entries within each complex, then average valid complexes."""
    if per_entry_loss.ndim < 2:
        raise ValueError("Macro pair losses require an explicit batch dimension")
    flattened_loss = per_entry_loss.reshape(int(per_entry_loss.shape[0]), -1)
    flattened_mask = mask.reshape(int(mask.shape[0]), -1)
    counts = flattened_mask.sum(dim=1)
    valid = counts > 0
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return zero_source.sum() * 0.0, 0
    per_complex = (
        flattened_loss.masked_fill(~flattened_mask, 0.0).sum(dim=1)
        / counts.clamp_min(1)
    )
    return per_complex[valid].mean(), valid_count


def masked_contact_bce_with_logits(
    logits: Any,
    targets: Any,
    mask: Any,
    *,
    positive_weight: float = 1.0,
    return_valid_complex_count: bool = False,
) -> Any:
    """Compute positive-class-weighted BCE with equal complex weighting.

    ``positive_weight`` changes discrimination training by upweighting positive
    labels. It is intentionally not used by the Brier calibration regularizer.
    """
    require_torch()
    logits = _require_float_tensor(logits, "contact_logits")
    if logits.ndim < 2:
        raise ValueError("contact_logits must include batch and pair dimensions")
    mask = _validate_boolean_mask(
        mask,
        "contact_mask",
        shape=tuple(logits.shape),
        device=logits.device,
    )
    selected_targets = _validate_binary_targets(
        targets,
        mask,
        logits,
        "contact_targets",
    )
    positive_weight = _finite_positive(positive_weight, "positive_weight")
    dense_targets = torch.zeros_like(logits)
    dense_targets[mask] = selected_targets
    per_entry = F.binary_cross_entropy_with_logits(
        logits,
        dense_targets,
        pos_weight=logits.new_tensor(positive_weight),
        reduction="none",
    )
    loss, count = _macro_masked_mean(per_entry, mask, zero_source=logits)
    return (loss, count) if return_valid_complex_count else loss


def masked_bin_cross_entropy(
    logits: Any,
    targets: Any,
    mask: Any,
    *,
    prediction_name: str = "bin_logits",
    target_name: str = "bin_targets",
    mask_name: str = "bin_mask",
    return_valid_complex_count: bool = False,
) -> Any:
    """Compute categorical cross-entropy with equal complex weighting."""
    require_torch()
    logits = _require_float_tensor(logits, prediction_name)
    if logits.ndim < 3:
        raise ValueError(
            f"{prediction_name} must include batch, item, and class dimensions"
        )
    if int(logits.shape[-1]) < 2:
        raise ValueError(f"{prediction_name} must contain at least two classes")
    prefix_shape = tuple(logits.shape[:-1])
    mask = _validate_boolean_mask(
        mask,
        mask_name,
        shape=prefix_shape,
        device=logits.device,
    )
    selected_targets = _validate_class_targets(
        targets,
        mask,
        logits,
        target_name,
    )
    dense_targets = torch.zeros(prefix_shape, dtype=torch.long, device=logits.device)
    dense_targets[mask] = selected_targets
    per_entry = F.cross_entropy(
        logits.movedim(-1, 1),
        dense_targets,
        reduction="none",
    )
    loss, count = _macro_masked_mean(per_entry, mask, zero_source=logits)
    return (loss, count) if return_valid_complex_count else loss


def masked_probability_consistency_loss(
    node_probabilities: Any,
    pair_derived_probabilities: Any,
    mask: Any,
) -> Any:
    """Align node probabilities only on candidate-supported residues."""
    require_torch()
    node_probabilities = _require_float_tensor(
        node_probabilities,
        "node_probabilities",
    )
    pair_derived_probabilities = _require_float_tensor(
        pair_derived_probabilities,
        "pair_derived_probabilities",
    )
    if tuple(node_probabilities.shape) != tuple(pair_derived_probabilities.shape):
        raise ValueError("Node and pair-derived probabilities must have equal shapes")
    if node_probabilities.device != pair_derived_probabilities.device:
        raise ValueError("Consistency probabilities must share a device")
    if node_probabilities.dtype != pair_derived_probabilities.dtype:
        raise ValueError("Consistency probabilities must share a dtype")
    for value, name in (
        (node_probabilities, "node_probabilities"),
        (pair_derived_probabilities, "pair_derived_probabilities"),
    ):
        if bool(((value < 0.0) | (value > 1.0)).any()):
            raise ValueError(f"{name} must lie in [0, 1]")
    mask = _validate_boolean_mask(
        mask,
        "node_consistency_mask",
        shape=tuple(node_probabilities.shape),
        device=node_probabilities.device,
    )
    per_entry = (node_probabilities - pair_derived_probabilities).square()
    loss, _ = _macro_masked_mean(
        per_entry,
        mask,
        zero_source=node_probabilities + pair_derived_probabilities,
    )
    return loss


def matched_decoy_margin_ranking_loss(
    matched_scores: Any,
    decoy_scores: Any,
    decoy_mask: Any,
    *,
    margin: float = 1.0,
) -> Any:
    """Rank each matched partner above explicit decoys with equal complex weight."""
    require_torch()
    matched_scores = _require_float_tensor(
        matched_scores,
        "matched_scores",
        ndim=1,
    )
    decoy_scores = _require_float_tensor(decoy_scores, "decoy_scores", ndim=2)
    if int(matched_scores.shape[0]) != int(decoy_scores.shape[0]):
        raise ValueError("Matched and decoy scores must have aligned batches")
    if matched_scores.device != decoy_scores.device:
        raise ValueError("Matched and decoy scores must share a device")
    if matched_scores.dtype != decoy_scores.dtype:
        raise ValueError("Matched and decoy scores must share a dtype")
    decoy_mask = _validate_boolean_mask(
        decoy_mask,
        "decoy_mask",
        shape=tuple(decoy_scores.shape),
        device=decoy_scores.device,
    )
    margin = _finite_positive(margin, "margin")
    pairwise = F.relu(
        margin - matched_scores.unsqueeze(1) + decoy_scores
    )
    loss, _ = _macro_masked_mean(
        pairwise,
        decoy_mask,
        zero_source=matched_scores.sum() + decoy_scores.sum(),
    )
    return loss


def masked_contact_brier_loss(
    logits: Any,
    targets: Any,
    mask: Any,
    *,
    return_valid_complex_count: bool = False,
) -> Any:
    """Compute an unweighted Brier calibration regularizer per complex."""
    require_torch()
    logits = _require_float_tensor(logits, "contact_logits")
    if logits.ndim < 2:
        raise ValueError("contact_logits must include batch and pair dimensions")
    mask = _validate_boolean_mask(
        mask,
        "contact_mask",
        shape=tuple(logits.shape),
        device=logits.device,
    )
    selected_targets = _validate_binary_targets(
        targets,
        mask,
        logits,
        "contact_targets",
    )
    dense_targets = torch.zeros_like(logits)
    dense_targets[mask] = selected_targets
    per_entry = (torch.sigmoid(logits) - dense_targets).square()
    loss, count = _macro_masked_mean(per_entry, mask, zero_source=logits)
    return (loss, count) if return_valid_complex_count else loss


def _validate_base_outputs(
    outputs: Mapping[str, Any],
) -> tuple[Any, PairGeometrySchema, frozenset[str], Any, Any]:
    """Validate orientation, immutable schema, candidates, and node supports."""
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must be a mapping")
    required = {
        "pair_orientation",
        "executed_heads",
        "geometry_schema",
        "geometry_schema_metadata",
        "geometry_schema_sha256",
        "candidate_pair_mask",
    }
    missing = required - set(outputs)
    if missing:
        raise KeyError(f"Pair-geometry outputs are missing keys: {sorted(missing)}")
    if outputs["pair_orientation"] != PAIR_ORIENTATION:
        raise ValueError(
            f"pair_orientation must equal {PAIR_ORIENTATION!r}"
        )
    schema = outputs["geometry_schema"]
    if not isinstance(schema, PairGeometrySchema):
        raise TypeError("geometry_schema must be PairGeometrySchema")
    if outputs["geometry_schema_sha256"] != schema.sha256:
        raise ValueError("geometry_schema_sha256 does not match geometry_schema")
    metadata = outputs["geometry_schema_metadata"]
    if not isinstance(metadata, FrozenGeometryMetadata) or metadata != (
        schema.canonical_metadata()
    ):
        raise ValueError("geometry_schema_metadata does not match geometry_schema")

    executed_value = outputs["executed_heads"]
    if not isinstance(executed_value, tuple):
        raise ValueError("executed_heads must be an immutable tuple")
    canonical_order = ("contact", "distance", "orientation", "interaction")
    if len(executed_value) != len(set(executed_value)):
        raise ValueError("executed_heads cannot contain duplicates")
    if any(name not in canonical_order for name in executed_value):
        raise ValueError("executed_heads contains an unknown head")
    expected_order = tuple(name for name in canonical_order if name in executed_value)
    if executed_value != expected_order:
        raise ValueError("executed_heads must use canonical order")
    executed = frozenset(executed_value)
    if "contact" in executed and "distance" not in executed:
        raise ValueError("Executed contact head requires executed distance head")

    candidate_pair_mask = outputs["candidate_pair_mask"]
    if not torch.is_tensor(candidate_pair_mask):
        raise TypeError("candidate_pair_mask must be a PyTorch tensor")
    if candidate_pair_mask.dtype != torch.bool or candidate_pair_mask.ndim != 3:
        raise ValueError(
            "candidate_pair_mask must be a three-dimensional boolean tensor"
        )
    if any(int(size) == 0 for size in candidate_pair_mask.shape):
        raise ValueError("candidate_pair_mask cannot contain an empty dimension")
    if bool((candidate_pair_mask.flatten(1).sum(dim=1) == 0).any()):
        raise ValueError("Every batch item must contain candidate-pair support")

    epitope_support = candidate_pair_mask.any(dim=1)
    paratope_support = candidate_pair_mask.any(dim=2)
    for side, derived in (
        ("epitope", epitope_support),
        ("paratope", paratope_support),
    ):
        key = f"{side}_mask"
        if key in outputs:
            supplied = _validate_boolean_mask(
                outputs[key],
                key,
                shape=tuple(derived.shape),
                device=candidate_pair_mask.device,
            )
            if not torch.equal(supplied, derived):
                raise ValueError(
                    f"{key} must equal support derived from candidate_pair_mask"
                )
    return candidate_pair_mask, schema, executed, epitope_support, paratope_support


def _require_head_tensor(
    outputs: Mapping[str, Any],
    executed: frozenset[str],
    head: str,
    key: str,
    *,
    ndim: int,
) -> Any:
    """Require one output tensor and an honest executed-head declaration."""
    if head not in executed:
        raise ValueError(f"{head} head was not executed")
    if key not in outputs:
        raise KeyError(f"Pair-geometry outputs are missing key: {key}")
    return _require_float_tensor(outputs[key], key, ndim=ndim)


def _validate_schema_equality(
    output_schema: PairGeometrySchema,
    target_schema: Any,
) -> PairGeometrySchema:
    """Require exact semantic equality, not only matching class counts."""
    if not isinstance(target_schema, PairGeometrySchema):
        raise TypeError("target_schema must be PairGeometrySchema")
    if (
        target_schema != output_schema
        or target_schema.sha256 != output_schema.sha256
    ):
        raise ValueError("target_schema must exactly equal the output geometry schema")
    return target_schema


def _validate_contact_distance_coherence(
    contact_targets: Any,
    contact_mask: Any,
    distance_targets: Any,
    distance_mask: Any,
    *,
    contact_distance_bin_count: int,
) -> None:
    """Reject contact labels that contradict distance bins on shared support."""
    overlap = contact_mask & distance_mask
    if not bool(overlap.any()):
        return
    expected = distance_targets[overlap] < contact_distance_bin_count
    observed = contact_targets[overlap].to(dtype=torch.bool)
    if not torch.equal(observed, expected):
        raise ValueError(
            "Contact labels contradict distance-bin targets on shared supervision"
        )


def _optional_node_consistency(
    outputs: Mapping[str, Any],
    *,
    side: str,
    derived_support: Any,
    node_probabilities: Any,
    consistency_mask: Any,
) -> Any:
    """Compute one enabled node term against exact candidate-derived support."""
    probability_key = f"{side}_probability"
    if probability_key not in outputs:
        raise KeyError(f"Pair-geometry outputs are missing key: {probability_key}")
    if node_probabilities is None or consistency_mask is None:
        raise ValueError(
            f"{side} node probabilities and consistency mask are required"
        )
    pair_derived = _require_float_tensor(outputs[probability_key], probability_key)
    if tuple(pair_derived.shape) != tuple(derived_support.shape):
        raise ValueError(f"{probability_key} must align with candidate-derived support")
    consistency_mask = _validate_boolean_mask(
        consistency_mask,
        f"{side}_consistency_mask",
        shape=tuple(derived_support.shape),
        device=pair_derived.device,
        support_mask=derived_support,
    )
    return masked_probability_consistency_loss(
        node_probabilities,
        pair_derived,
        consistency_mask,
    )


def pair_geometry_objective(
    outputs: Mapping[str, Any],
    *,
    contact_targets: Any | None = None,
    contact_mask: Any | None = None,
    distance_bin_targets: Any | None = None,
    distance_mask: Any | None = None,
    orientation_bin_targets: Any | None = None,
    orientation_mask: Any | None = None,
    target_schema: PairGeometrySchema | None = None,
    weights: PairGeometryLossWeights = PairGeometryLossWeights(),
    contact_positive_weight: float = 1.0,
    epitope_node_probabilities: Any | None = None,
    epitope_consistency_mask: Any | None = None,
    paratope_node_probabilities: Any | None = None,
    paratope_consistency_mask: Any | None = None,
    decoy_scores: Any | None = None,
    decoy_mask: Any | None = None,
    ranking_margin: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Compose only positive-weight objectives under an exact schema contract.

    Contact BCE is a positive-class-weighted discrimination loss. Contact Brier
    is a separate, unweighted calibration regularizer. Contact, distance,
    orientation, and Brier terms reduce within each complex before averaging
    valid complexes. Zero-weight heads are never accessed or validated.
    """
    require_torch()
    if not isinstance(weights, PairGeometryLossWeights):
        raise TypeError("weights must be PairGeometryLossWeights")
    weights.validate()
    (
        candidate_pair_mask,
        output_schema,
        executed,
        epitope_support,
        paratope_support,
    ) = _validate_base_outputs(outputs)
    zero = torch.zeros((), device=candidate_pair_mask.device)
    components: dict[str, Any] = {
        name: zero
        for name in (
            "contact_bce",
            "distance_ce",
            "orientation_ce",
            "epitope_consistency",
            "paratope_consistency",
            "matched_decoy_ranking",
            "contact_brier",
        )
    }
    valid_complex_counts = {
        "contact_bce": 0,
        "distance_ce": 0,
        "orientation_ce": 0,
        "contact_brier": 0,
    }

    needs_contact = weights.contact_bce > 0.0 or weights.contact_brier > 0.0
    needs_distance = weights.distance_ce > 0.0
    needs_orientation = weights.orientation_ce > 0.0
    needs_schema = needs_contact or needs_distance or needs_orientation
    if needs_schema:
        _validate_schema_equality(output_schema, target_schema)

    pair_shape = tuple(candidate_pair_mask.shape)
    validated_contact_targets = None
    validated_contact_mask = None
    validated_distance_targets = None
    validated_distance_mask = None

    if needs_contact:
        contact_logits = _require_head_tensor(
            outputs,
            executed,
            "contact",
            "contact_logits",
            ndim=3,
        )
        distance_logits_for_contact = _require_head_tensor(
            outputs,
            executed,
            "distance",
            "distance_bin_logits",
            ndim=4,
        )
        if tuple(contact_logits.shape) != pair_shape:
            raise ValueError("contact_logits must align with candidate_pair_mask")
        if tuple(distance_logits_for_contact.shape[:3]) != pair_shape:
            raise ValueError("distance_bin_logits must align with candidate_pair_mask")
        if (
            contact_logits.device != candidate_pair_mask.device
            or distance_logits_for_contact.device != candidate_pair_mask.device
        ):
            raise ValueError("Contact and distance outputs must share the mask device")
        if distance_logits_for_contact.dtype != contact_logits.dtype:
            raise ValueError("Contact and distance outputs must share a dtype")
        if int(distance_logits_for_contact.shape[-1]) != output_schema.num_distance_bins:
            raise ValueError("distance_bin_logits class count violates geometry schema")
        near = torch.logsumexp(
            distance_logits_for_contact[
                ..., : output_schema.contact_distance_bin_count
            ],
            dim=-1,
        )
        far = torch.logsumexp(
            distance_logits_for_contact[
                ..., output_schema.contact_distance_bin_count :
            ],
            dim=-1,
        )
        expected_contact_logits = near - far
        if not torch.allclose(
            contact_logits[candidate_pair_mask],
            expected_contact_logits[candidate_pair_mask],
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(
                "contact_logits do not match the schema-defined distance-bin mapping"
            )
        validated_contact_mask = _validate_boolean_mask(
            contact_mask,
            "contact_mask",
            shape=pair_shape,
            device=candidate_pair_mask.device,
            support_mask=candidate_pair_mask,
        )
        _validate_binary_targets(
            contact_targets,
            validated_contact_mask,
            contact_logits,
            "contact_targets",
        )
        validated_contact_targets = contact_targets
        if weights.contact_bce > 0.0:
            (
                components["contact_bce"],
                valid_complex_counts["contact_bce"],
            ) = masked_contact_bce_with_logits(
                contact_logits,
                contact_targets,
                validated_contact_mask,
                positive_weight=contact_positive_weight,
                return_valid_complex_count=True,
            )
        if weights.contact_brier > 0.0:
            (
                components["contact_brier"],
                valid_complex_counts["contact_brier"],
            ) = masked_contact_brier_loss(
                contact_logits,
                contact_targets,
                validated_contact_mask,
                return_valid_complex_count=True,
            )

    if needs_distance:
        distance_logits = _require_head_tensor(
            outputs,
            executed,
            "distance",
            "distance_bin_logits",
            ndim=4,
        )
        if tuple(distance_logits.shape[:3]) != pair_shape:
            raise ValueError("distance_bin_logits must align with candidate_pair_mask")
        if distance_logits.device != candidate_pair_mask.device:
            raise ValueError("distance_bin_logits must share the mask device")
        if int(distance_logits.shape[-1]) != output_schema.num_distance_bins:
            raise ValueError("distance_bin_logits class count violates geometry schema")
        validated_distance_mask = _validate_boolean_mask(
            distance_mask,
            "distance_mask",
            shape=pair_shape,
            device=candidate_pair_mask.device,
            support_mask=candidate_pair_mask,
        )
        _validate_class_targets(
            distance_bin_targets,
            validated_distance_mask,
            distance_logits,
            "distance_bin_targets",
        )
        validated_distance_targets = distance_bin_targets
        (
            components["distance_ce"],
            valid_complex_counts["distance_ce"],
        ) = masked_bin_cross_entropy(
            distance_logits,
            distance_bin_targets,
            validated_distance_mask,
            prediction_name="distance_bin_logits",
            target_name="distance_bin_targets",
            mask_name="distance_mask",
            return_valid_complex_count=True,
        )

    if needs_orientation:
        orientation_logits = _require_head_tensor(
            outputs,
            executed,
            "orientation",
            "orientation_bin_logits",
            ndim=5,
        )
        expected_shape = (
            *pair_shape,
            output_schema.num_orientation_axes,
            output_schema.num_orientation_bins,
        )
        if tuple(orientation_logits.shape) != expected_shape:
            raise ValueError(
                "orientation_bin_logits dimensions violate geometry schema"
            )
        if orientation_logits.device != candidate_pair_mask.device:
            raise ValueError("orientation_bin_logits must share the mask device")
        orientation_prefix = tuple(orientation_logits.shape[:-1])
        expanded_candidate = candidate_pair_mask.unsqueeze(-1).expand(
            orientation_prefix
        )
        validated_orientation_mask = _validate_boolean_mask(
            orientation_mask,
            "orientation_mask",
            shape=orientation_prefix,
            device=candidate_pair_mask.device,
            support_mask=expanded_candidate,
        )
        _validate_class_targets(
            orientation_bin_targets,
            validated_orientation_mask,
            orientation_logits,
            "orientation_bin_targets",
        )
        (
            components["orientation_ce"],
            valid_complex_counts["orientation_ce"],
        ) = masked_bin_cross_entropy(
            orientation_logits,
            orientation_bin_targets,
            validated_orientation_mask,
            prediction_name="orientation_bin_logits",
            target_name="orientation_bin_targets",
            mask_name="orientation_mask",
            return_valid_complex_count=True,
        )

    if (
        validated_contact_mask is not None
        and validated_distance_mask is not None
    ):
        _validate_contact_distance_coherence(
            validated_contact_targets,
            validated_contact_mask,
            validated_distance_targets,
            validated_distance_mask,
            contact_distance_bin_count=output_schema.contact_distance_bin_count,
        )

    if weights.epitope_consistency > 0.0:
        components["epitope_consistency"] = _optional_node_consistency(
            outputs,
            side="epitope",
            derived_support=epitope_support,
            node_probabilities=epitope_node_probabilities,
            consistency_mask=epitope_consistency_mask,
        )
    if weights.paratope_consistency > 0.0:
        components["paratope_consistency"] = _optional_node_consistency(
            outputs,
            side="paratope",
            derived_support=paratope_support,
            node_probabilities=paratope_node_probabilities,
            consistency_mask=paratope_consistency_mask,
        )
    if weights.matched_decoy_ranking > 0.0:
        if "interaction" not in executed:
            raise ValueError("interaction head was not executed")
        if "partner_score" not in outputs:
            raise KeyError("Pair-geometry outputs are missing key: partner_score")
        if decoy_scores is None or decoy_mask is None:
            raise ValueError(
                "decoy_scores and decoy_mask are required for matched-decoy ranking"
            )
        components["matched_decoy_ranking"] = matched_decoy_margin_ranking_loss(
            outputs["partner_score"],
            decoy_scores,
            decoy_mask,
            margin=ranking_margin,
        )

    loss_names = tuple(vars(weights))
    total = zero
    for name in loss_names:
        total = total + getattr(weights, name) * components[name]
    if total.ndim != 0 or not bool(torch.isfinite(total)):
        raise RuntimeError("Pair-geometry objective produced a nonfinite scalar")
    components["valid_complex_counts"] = valid_complex_counts
    return total, components
