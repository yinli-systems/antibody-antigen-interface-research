"""Leakage-safe multi-task objectives for FrameContact."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, Mapping

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None
    functional = None


_ModuleBase = nn.Module if nn is not None else object


def _require_torch() -> Any:
    if torch is None or nn is None or functional is None:
        raise RuntimeError("PyTorch is required for the FrameContact objective")
    return torch


def balanced_binary_loss_per_complex(logits: Any, labels: Any, mask: Any) -> Any:
    """Give every complex and each class equal mass before averaging."""
    torch_module = _require_torch()
    if (
        tuple(logits.shape) != tuple(labels.shape) != tuple(mask.shape)
        or tuple(logits.shape) != tuple(mask.shape)
    ):
        raise ValueError("logits, labels, and mask must have identical shapes")
    if mask.dtype != torch_module.bool:
        raise ValueError("mask must be boolean")
    if not bool((((labels == 0) | (labels == 1)) | ~mask).all()):
        raise ValueError("observed labels must be binary")
    flattened_logits = logits.flatten(start_dim=1)
    flattened_labels = labels.flatten(start_dim=1).to(logits.dtype)
    flattened_mask = mask.flatten(start_dim=1)
    losses = []
    for batch_index in range(int(logits.shape[0])):
        observed_logits = flattened_logits[batch_index][flattened_mask[batch_index]]
        observed_labels = flattened_labels[batch_index][flattened_mask[batch_index]]
        positive = observed_labels == 1
        negative = observed_labels == 0
        class_losses = []
        if bool(positive.any()):
            class_losses.append(functional.softplus(-observed_logits[positive]).mean())
        if bool(negative.any()):
            class_losses.append(functional.softplus(observed_logits[negative]).mean())
        if not class_losses:
            raise ValueError("each complex must contain at least one observed label")
        # Average over classes that are actually observed in this complex. Most
        # complexes retain the original equal positive/negative weighting; a
        # rare zero-contact complex receives a well-defined negative-only BCE
        # instead of crashing or fabricating a positive residue pair.
        losses.append(torch_module.stack(class_losses).mean())
    return torch_module.stack(losses).mean()


def distogram_bin_centers(boundaries: Any) -> Any:
    """Return finite representatives for bins ``[0,b0], ... , (bn, inf)``."""
    torch_module = _require_torch()
    if not torch_module.is_tensor(boundaries) or boundaries.ndim != 1:
        raise ValueError("distance bin boundaries must be a one-dimensional tensor")
    values = boundaries.float()
    if values.numel() == 0 or not bool(torch_module.isfinite(values).all()):
        raise ValueError("distance bin boundaries must be nonempty and finite")
    if not bool((values > 0).all()) or (
        values.numel() > 1 and not bool((values[1:] > values[:-1]).all())
    ):
        raise ValueError("distance bin boundaries must be positive and increasing")
    lower = torch_module.cat((values.new_zeros(1), values))
    final_width = values[-1] - (values[-2] if values.numel() > 1 else 0.0)
    upper = torch_module.cat((values, values[-1:] + final_width.clamp_min(1.0)))
    return (lower + upper) * 0.5


def expected_distance_from_distogram(
    distogram_logits: Any, boundaries: Any
) -> Any:
    """Compute an SE(3)-invariant expected inter-chain distance matrix."""
    _require_torch()
    if distogram_logits.ndim != 4:
        raise ValueError("distogram_logits must have shape [batch, ab, ag, bin]")
    centers = distogram_bin_centers(boundaries).to(
        device=distogram_logits.device, dtype=torch.float32
    )
    if int(distogram_logits.shape[-1]) != int(centers.numel()):
        raise ValueError("distogram bin count differs from its boundaries")
    probabilities = torch.softmax(distogram_logits.float(), dim=-1)
    return torch.sum(probabilities * centers, dim=-1)


def distogram_cross_entropy_per_complex(
    logits: Any,
    distances: Any,
    mask: Any,
    boundaries: Any,
) -> Any:
    """Average categorical distance NLL per complex, then across complexes."""
    torch_module = _require_torch()
    if logits.ndim != 4 or tuple(logits.shape[:3]) != tuple(distances.shape):
        raise ValueError("distogram logits and pair distances must align")
    if tuple(mask.shape) != tuple(distances.shape) or mask.dtype != torch_module.bool:
        raise ValueError("pair distance mask must be boolean and align with distances")
    if not bool(torch_module.isfinite(distances[mask]).all()) or bool(
        (distances[mask] < 0).any()
    ):
        raise ValueError("observed pair distances must be finite and nonnegative")
    boundaries = boundaries.to(device=logits.device, dtype=torch.float32)
    if int(logits.shape[-1]) != int(boundaries.numel()) + 1:
        raise ValueError("distogram bin count differs from its boundaries")
    safe_distances = distances.float().masked_fill(~mask, 0.0)
    targets = torch_module.bucketize(safe_distances, boundaries)
    losses = []
    for batch_index in range(int(logits.shape[0])):
        observed = mask[batch_index]
        if not bool(observed.any()):
            raise ValueError("every complex must contain an observed pair distance")
        losses.append(
            functional.cross_entropy(
                logits[batch_index][observed].float(),
                targets[batch_index][observed],
            )
        )
    return torch_module.stack(losses).mean()


def bernoulli_js_consistency_per_complex(
    direct_logits: Any,
    comparison_logits: Any,
    mask: Any,
) -> Any:
    """Jensen-Shannon consistency for two aligned Bernoulli predictions.

    The loss is symmetric, bounded, and averaged within each complex before
    averaging across complexes so long antigens cannot dominate the objective.
    It is valid only when both logits describe the same aligned entities.
    """

    torch_module = _require_torch()
    if (
        tuple(direct_logits.shape) != tuple(comparison_logits.shape)
        or tuple(direct_logits.shape) != tuple(mask.shape)
        or mask.dtype != torch_module.bool
    ):
        raise ValueError("Bernoulli consistency logits and mask must align")
    p = torch_module.sigmoid(direct_logits.float()).clamp(1e-7, 1.0 - 1e-7)
    q = torch_module.sigmoid(comparison_logits.float()).clamp(
        1e-7, 1.0 - 1e-7
    )
    midpoint = 0.5 * (p + q)

    def bernoulli_kl(left: Any, right: Any) -> Any:
        return left * (left.log() - right.log()) + (1.0 - left) * (
            (1.0 - left).log() - (1.0 - right).log()
        )

    pair_loss = 0.5 * (
        bernoulli_kl(p, midpoint) + bernoulli_kl(q, midpoint)
    )
    losses = []
    for batch_index in range(int(mask.shape[0])):
        observed = mask[batch_index]
        if not bool(observed.any()):
            raise ValueError("every complex must contain an observed entity")
        losses.append(pair_loss[batch_index][observed].mean())
    return torch_module.stack(losses).mean()


def pair_node_consistency_per_complex(
    *,
    paratope_unary_logits: Any,
    paratope_contact_logits: Any,
    antibody_mask: Any,
    epitope_unary_logits: Any,
    epitope_contact_logits: Any,
    antigen_mask: Any,
) -> tuple[Any, Any, Any]:
    """Couple independent residue heads to contact-derived marginals.

    Each side is averaged per complex by
    :func:`bernoulli_js_consistency_per_complex`, then the antibody and antigen
    losses receive equal weight.  This prevents the typically longer antigen
    axis from dominating and makes the coupling auditable as two explicit
    terms.  The function deliberately accepts *unary* node heads rather than
    final hybrid residue logits, avoiding a circular comparison when the final
    logits already contain the contact marginal.
    """

    paratope = bernoulli_js_consistency_per_complex(
        paratope_unary_logits,
        paratope_contact_logits,
        antibody_mask,
    )
    epitope = bernoulli_js_consistency_per_complex(
        epitope_unary_logits,
        epitope_contact_logits,
        antigen_mask,
    )
    return 0.5 * (paratope + epitope), paratope, epitope


def _node_spatial_smoothness_one_side(
    logits: Any,
    backbone: Any,
    residue_mask: Any,
    chain_ids: Any,
    *,
    neighbor_radius: float,
    neighbor_k: int,
) -> Any:
    """Penalize local probability discontinuities on one monomer graph."""

    torch_module = _require_torch()
    if logits.ndim != 2:
        raise ValueError("node spatial logits must have shape [batch, residue]")
    expected = tuple(logits.shape)
    if (
        tuple(backbone.shape) != (*expected, 3, 3)
        or tuple(residue_mask.shape) != expected
        or tuple(chain_ids.shape) != expected
        or residue_mask.dtype != torch_module.bool
    ):
        raise ValueError("node spatial inputs must align with the residue axis")
    if not math.isfinite(neighbor_radius) or neighbor_radius <= 0:
        raise ValueError("node spatial neighbor radius must be finite and positive")
    if isinstance(neighbor_k, bool) or not isinstance(neighbor_k, int) or neighbor_k <= 0:
        raise ValueError("node spatial neighbor_k must be positive")

    probabilities = torch_module.sigmoid(logits.float())
    losses = []
    for batch_index in range(int(logits.shape[0])):
        frame_valid = torch_module.isfinite(backbone[batch_index]).all(
            dim=(-1, -2)
        )
        active = residue_mask[batch_index] & frame_valid
        source, target, _ = _sparse_neighbor_edges(
            backbone[batch_index, :, 1, :].nan_to_num(),
            active,
            chain_ids[batch_index],
            radius=neighbor_radius,
            neighbor_k=neighbor_k,
        )
        if source.numel() == 0:
            raise ValueError(
                "node spatial supervision selected no valid intrachain edges"
            )
        # Radius-kNN can contain both orientations of one physical edge.  Make
        # the graph undirected before averaging so directed duplication cannot
        # change a complex's weight.
        undirected = torch_module.stack(
            (torch_module.minimum(source, target), torch_module.maximum(source, target)),
            dim=1,
        ).unique(dim=0)
        difference = (
            probabilities[batch_index, undirected[:, 0]]
            - probabilities[batch_index, undirected[:, 1]]
        )
        losses.append(difference.square().mean())
    return torch_module.stack(losses).mean()


def node_spatial_smoothness_per_complex(
    *,
    paratope_unary_logits: Any,
    antibody_backbone: Any,
    antibody_mask: Any,
    antibody_chain_ids: Any,
    epitope_unary_logits: Any,
    antigen_backbone: Any,
    antigen_mask: Any,
    antigen_chain_ids: Any,
    neighbor_radius: float = 12.0,
    neighbor_k: int = 8,
) -> tuple[Any, Any, Any]:
    """Regularize coarse epitope/paratope probabilities on monomer geometry.

    This consumes only within-partner C-alpha neighborhoods.  It therefore
    remains invariant to the unrelated antibody and antigen input coordinate
    frames and does not invent cross-chain distance supervision.  The two
    partner losses receive equal mass after per-complex averaging.
    """

    paratope = _node_spatial_smoothness_one_side(
        paratope_unary_logits,
        antibody_backbone,
        antibody_mask,
        antibody_chain_ids,
        neighbor_radius=neighbor_radius,
        neighbor_k=neighbor_k,
    )
    epitope = _node_spatial_smoothness_one_side(
        epitope_unary_logits,
        antigen_backbone,
        antigen_mask,
        antigen_chain_ids,
        neighbor_radius=neighbor_radius,
        neighbor_k=neighbor_k,
    )
    return 0.5 * (paratope + epitope), paratope, epitope


def select_local_antigen_patch(
    scores: Any,
    backbone: Any,
    residue_mask: Any,
    surface_mask: Any,
    *,
    top_seed_count: int,
    patch_radius: float,
) -> Any:
    """Select a detached, prediction-routed surface patch without label access."""
    torch_module = _require_torch()
    expected = tuple(residue_mask.shape)
    if (
        tuple(scores.shape) != expected
        or tuple(backbone.shape) != (*expected, 3, 3)
        or tuple(surface_mask.shape) != expected
        or residue_mask.dtype != torch_module.bool
        or surface_mask.dtype != torch_module.bool
    ):
        raise ValueError("antigen patch inputs must share a boolean residue axis")
    if isinstance(top_seed_count, bool) or not isinstance(top_seed_count, int) or top_seed_count <= 0:
        raise ValueError("top_seed_count must be positive")
    if not math.isfinite(patch_radius) or patch_radius <= 0:
        raise ValueError("patch_radius must be finite and positive")
    ca = backbone[..., 1, :].float()
    frame_valid = torch_module.isfinite(backbone).all(dim=(-1, -2))
    eligible = residue_mask & surface_mask & frame_valid
    patches = torch_module.zeros_like(residue_mask)
    detached_scores = scores.detach().float()
    for batch_index in range(int(scores.shape[0])):
        candidate_indices = torch_module.nonzero(
            eligible[batch_index], as_tuple=False
        ).flatten()
        if candidate_indices.numel() == 0:
            raise ValueError("every complex requires an eligible antigen surface residue")
        seed_count = min(top_seed_count, int(candidate_indices.numel()))
        selected = candidate_indices[
            torch_module.topk(
                detached_scores[batch_index, candidate_indices], seed_count
            ).indices
        ]
        distances = torch_module.cdist(
            ca[batch_index, selected], ca[batch_index, candidate_indices]
        )
        local = (distances <= patch_radius).any(dim=0)
        patches[batch_index, candidate_indices[local]] = True
    return patches


def _sparse_neighbor_edges(
    ca: Any,
    active_mask: Any,
    chain_ids: Any,
    *,
    radius: float,
    neighbor_k: int,
) -> tuple[Any, Any, Any]:
    torch_module = _require_torch()
    distances = torch_module.cdist(ca.float(), ca.float())
    count = int(ca.shape[0])
    eye = torch_module.eye(count, dtype=torch_module.bool, device=ca.device)
    eligible = (
        active_mask[:, None]
        & active_mask[None, :]
        & (chain_ids[:, None] == chain_ids[None, :])
        & ~eye
        & (distances <= radius)
    )
    masked = distances.masked_fill(~eligible, float("inf"))
    k = min(neighbor_k, max(count - 1, 1))
    nearest_distance, nearest_index = torch_module.topk(
        masked, k, dim=-1, largest=False
    )
    source = torch_module.arange(count, device=ca.device)[:, None].expand(-1, k)
    valid = torch_module.isfinite(nearest_distance)
    return source[valid], nearest_index[valid], nearest_distance[valid]


def sparse_triangle_inequality_loss(
    predicted_distances: Any,
    pair_mask: Any,
    antibody_backbone: Any,
    antigen_backbone: Any,
    antibody_chain_ids: Any,
    antigen_chain_ids: Any,
    antibody_active_mask: Any,
    antigen_active_mask: Any,
    *,
    neighbor_radius: float = 12.0,
    neighbor_k: int = 8,
    slack: float = 0.5,
) -> Any:
    """Penalize pose-free local triangle violations on sparse active patches.

    For known intra-chain distance ``d(i,k)`` and two predicted inter-chain
    distances to the same partner residue, the reverse triangle inequality
    requires ``|D(i,j)-D(k,j)| <= d(i,k)``.  The symmetric antigen-side term is
    evaluated as well.  No cross-chain input coordinates are consumed.
    """
    torch_module = _require_torch()
    if predicted_distances.ndim != 3 or tuple(pair_mask.shape) != tuple(
        predicted_distances.shape
    ):
        raise ValueError("predicted distances and pair mask must align")
    if pair_mask.dtype != torch_module.bool:
        raise ValueError("pair mask must be boolean")
    if isinstance(neighbor_k, bool) or not isinstance(neighbor_k, int) or neighbor_k <= 0:
        raise ValueError("neighbor_k must be positive")
    if not math.isfinite(neighbor_radius) or neighbor_radius <= 0:
        raise ValueError("neighbor_radius must be finite and positive")
    if not math.isfinite(slack) or slack < 0:
        raise ValueError("triangle slack must be finite and nonnegative")
    losses = []
    for batch_index in range(int(predicted_distances.shape[0])):
        local_prediction = predicted_distances[batch_index]
        local_pair_mask = pair_mask[batch_index]
        antibody_ca = antibody_backbone[batch_index, :, 1, :]
        antigen_ca = antigen_backbone[batch_index, :, 1, :]
        antibody_valid = (
            antibody_active_mask[batch_index]
            & torch_module.isfinite(antibody_backbone[batch_index]).all(dim=(-1, -2))
        )
        antigen_valid = (
            antigen_active_mask[batch_index]
            & torch_module.isfinite(antigen_backbone[batch_index]).all(dim=(-1, -2))
        )
        ab_source, ab_target, ab_distance = _sparse_neighbor_edges(
            antibody_ca.nan_to_num(),
            antibody_valid,
            antibody_chain_ids[batch_index],
            radius=neighbor_radius,
            neighbor_k=neighbor_k,
        )
        if ab_source.numel():
            valid = (
                local_pair_mask[ab_source]
                & local_pair_mask[ab_target]
                & antigen_valid[None, :]
            )
            violation = (
                (local_prediction[ab_source] - local_prediction[ab_target]).abs()
                - ab_distance[:, None]
                - slack
            )
            if bool(valid.any()):
                losses.append(functional.relu(violation[valid]).square().mean())
        ag_source, ag_target, ag_distance = _sparse_neighbor_edges(
            antigen_ca.nan_to_num(),
            antigen_valid,
            antigen_chain_ids[batch_index],
            radius=neighbor_radius,
            neighbor_k=neighbor_k,
        )
        if ag_source.numel():
            valid = (
                local_pair_mask[:, ag_source]
                & local_pair_mask[:, ag_target]
                & antibody_valid[:, None]
            )
            violation = (
                (
                    local_prediction[:, ag_source]
                    - local_prediction[:, ag_target]
                ).abs()
                - ag_distance[None, :]
                - slack
            )
            if bool(valid.any()):
                losses.append(functional.relu(violation[valid]).square().mean())
    if not losses:
        raise ValueError("triangle supervision selected no valid local constraints")
    return torch_module.stack(losses).mean()


def far_patch_triangle_exclusion_loss(
    predicted_distances: Any,
    pair_mask: Any,
    antigen_backbone: Any,
    antibody_active_mask: Any,
    antigen_patch_mask: Any,
    antigen_surface_mask: Any,
    *,
    minimum_patch_separation: float = 20.0,
    far_residue_k: int = 16,
    slack: float = 0.5,
) -> Any:
    """Make a selected antigen patch compete with distant surface patches.

    For antibody residue ``i``, selected-patch residue ``j`` and a distant
    antigen surface residue ``k``, the ordinary triangle inequality requires

    ``D(i,j) + D(i,k) >= d(j,k)``.

    Unlike the existing reverse-triangle term, this lower bound explicitly
    couples attraction to the selected patch with exclusion of geometrically
    distant patches: decreasing ``D(i,j)`` increases the required lower bound on
    ``D(i,k)``.  It consumes only monomer intra-chain distances and predicted
    inter-chain distances, so the two input coordinate frames never need to be
    aligned.  Candidate selection is deterministic and label-free.
    """

    torch_module = _require_torch()
    if predicted_distances.ndim != 3 or tuple(pair_mask.shape) != tuple(
        predicted_distances.shape
    ):
        raise ValueError("predicted distances and pair mask must align")
    batch_size, antibody_count, antigen_count = predicted_distances.shape
    if tuple(antigen_backbone.shape) != (batch_size, antigen_count, 3, 3):
        raise ValueError("antigen backbone must align with predicted distances")
    if (
        tuple(antibody_active_mask.shape) != (batch_size, antibody_count)
        or tuple(antigen_patch_mask.shape) != (batch_size, antigen_count)
        or tuple(antigen_surface_mask.shape) != (batch_size, antigen_count)
        or pair_mask.dtype != torch_module.bool
        or antibody_active_mask.dtype != torch_module.bool
        or antigen_patch_mask.dtype != torch_module.bool
        or antigen_surface_mask.dtype != torch_module.bool
    ):
        raise ValueError("far-patch masks must be boolean and axis-aligned")
    if not math.isfinite(minimum_patch_separation) or minimum_patch_separation <= 0:
        raise ValueError("minimum patch separation must be finite and positive")
    if isinstance(far_residue_k, bool) or not isinstance(far_residue_k, int) or far_residue_k <= 0:
        raise ValueError("far_residue_k must be positive")
    if not math.isfinite(slack) or slack < 0:
        raise ValueError("far-patch slack must be finite and nonnegative")

    losses = []
    for batch_index in range(int(batch_size)):
        frame_valid = torch_module.isfinite(
            antigen_backbone[batch_index]
        ).all(dim=(-1, -2))
        antibody_indices = torch_module.nonzero(
            antibody_active_mask[batch_index], as_tuple=False
        ).flatten()
        patch_indices = torch_module.nonzero(
            antigen_patch_mask[batch_index] & frame_valid, as_tuple=False
        ).flatten()
        surface_indices = torch_module.nonzero(
            antigen_surface_mask[batch_index]
            & frame_valid
            & ~antigen_patch_mask[batch_index],
            as_tuple=False,
        ).flatten()
        if (
            antibody_indices.numel() == 0
            or patch_indices.numel() == 0
            or surface_indices.numel() == 0
        ):
            continue
        ca = antigen_backbone[batch_index, :, 1, :].float()
        intra = torch_module.cdist(ca[patch_indices], ca[surface_indices])
        eligible = intra >= float(minimum_patch_separation)
        masked = intra.masked_fill(~eligible, float("-inf"))
        k = min(far_residue_k, int(surface_indices.numel()))
        known_distance, local_far_index = torch_module.topk(
            masked, k, dim=-1, largest=True
        )
        valid_far = torch_module.isfinite(known_distance)
        if not bool(valid_far.any()):
            continue
        far_indices = surface_indices[local_far_index]
        local_prediction = predicted_distances[batch_index]
        local_pair_mask = pair_mask[batch_index]
        near_prediction = local_prediction[antibody_indices][:, patch_indices]
        near_valid = local_pair_mask[antibody_indices][:, patch_indices]
        flat_far = far_indices.flatten()
        far_prediction = local_prediction[antibody_indices][:, flat_far].reshape(
            int(antibody_indices.numel()), int(patch_indices.numel()), k
        )
        far_valid = local_pair_mask[antibody_indices][:, flat_far].reshape(
            int(antibody_indices.numel()), int(patch_indices.numel()), k
        )
        valid = near_valid[:, :, None] & far_valid & valid_far[None, :, :]
        violation = (
            known_distance[None, :, :]
            - near_prediction[:, :, None]
            - far_prediction
            - float(slack)
        )
        if bool(valid.any()):
            losses.append(functional.relu(violation[valid]).square().mean())
    if not losses:
        # Short antigens or compact surface patches can legitimately have no
        # residue beyond the preregistered separation in a particular batch.
        # Keep those examples trainable; the epoch audit reports coverage.
        return predicted_distances.sum() * 0.0
    return torch_module.stack(losses).mean()


def local_edm_realizability_loss(
    predicted_distances: Any,
    pair_mask: Any,
    antibody_backbone: Any,
    antigen_backbone: Any,
    antibody_active_mask: Any,
    antigen_active_mask: Any,
    *,
    numerical_tolerance: float = 1e-6,
) -> Any:
    """Test whether local intra/inter-chain distances form a 3D Euclidean EDM.

    This diagnostic consumes *only* invariant distances.  It combines the two
    known intra-chain C-alpha distance blocks with the predicted cross-chain
    block, double-centers the squared distance matrix, and penalizes negative
    Gram eigenvalues plus positive eigenvalues outside the largest three.  It
    therefore does not require the antibody and antigen input coordinate frames
    to be aligned.

    The active masks are expected to have already selected a small CDR-by-
    antigen-surface patch.  The routine deliberately has no learned or label-
    driven subset selection of its own so it can be audited independently of
    the routing policy used by a future exploratory protocol.
    """
    torch_module = _require_torch()
    if predicted_distances.ndim != 3 or tuple(pair_mask.shape) != tuple(
        predicted_distances.shape
    ):
        raise ValueError("predicted distances and pair mask must align")
    if pair_mask.dtype != torch_module.bool:
        raise ValueError("pair mask must be boolean")
    batch_size, antibody_count, antigen_count = predicted_distances.shape
    if tuple(antibody_backbone.shape) != (batch_size, antibody_count, 3, 3):
        raise ValueError("antibody backbone must align with predicted distances")
    if tuple(antigen_backbone.shape) != (batch_size, antigen_count, 3, 3):
        raise ValueError("antigen backbone must align with predicted distances")
    if (
        tuple(antibody_active_mask.shape) != (batch_size, antibody_count)
        or tuple(antigen_active_mask.shape) != (batch_size, antigen_count)
        or antibody_active_mask.dtype != torch_module.bool
        or antigen_active_mask.dtype != torch_module.bool
    ):
        raise ValueError("active masks must be boolean and align with residue axes")
    if not math.isfinite(numerical_tolerance) or numerical_tolerance < 0:
        raise ValueError("numerical_tolerance must be finite and nonnegative")

    losses = []
    for batch_index in range(int(batch_size)):
        antibody_valid = antibody_active_mask[batch_index] & torch_module.isfinite(
            antibody_backbone[batch_index]
        ).all(dim=(-1, -2))
        antigen_valid = antigen_active_mask[batch_index] & torch_module.isfinite(
            antigen_backbone[batch_index]
        ).all(dim=(-1, -2))
        antibody_indices = torch_module.nonzero(
            antibody_valid, as_tuple=False
        ).flatten()
        antigen_indices = torch_module.nonzero(antigen_valid, as_tuple=False).flatten()
        if antibody_indices.numel() < 2 or antigen_indices.numel() < 2:
            raise ValueError(
                "EDM supervision requires at least two active residues on each partner"
            )
        local_pair_mask = pair_mask[batch_index][antibody_indices][
            :, antigen_indices
        ]
        if not bool(local_pair_mask.all()):
            raise ValueError("EDM supervision requires a complete active cross block")

        antibody_ca = antibody_backbone[
            batch_index, antibody_indices, 1, :
        ].float()
        antigen_ca = antigen_backbone[batch_index, antigen_indices, 1, :].float()
        cross = predicted_distances[batch_index][antibody_indices][
            :, antigen_indices
        ].float()
        if not bool(torch_module.isfinite(cross).all()) or bool((cross < 0).any()):
            raise ValueError("active predicted distances must be finite and nonnegative")

        # CUDA eigvalsh does not accept BF16, and allowing autocast to downcast
        # the Gram matmuls also makes the small signed eigenvalues unstable.
        # Keep this local spectral calculation in differentiable FP32 while the
        # surrounding model remains eligible for mixed-precision execution.
        with torch_module.autocast(
            device_type=predicted_distances.device.type, enabled=False
        ):
            antibody_within = torch_module.cdist(antibody_ca, antibody_ca)
            antigen_within = torch_module.cdist(antigen_ca, antigen_ca)
            full_distance = torch_module.cat(
                (
                    torch_module.cat((antibody_within, cross), dim=1),
                    torch_module.cat(
                        (cross.transpose(0, 1), antigen_within), dim=1
                    ),
                ),
                dim=0,
            )
            count = int(full_distance.shape[0])
            off_diagonal = ~torch_module.eye(
                count, dtype=torch_module.bool, device=full_distance.device
            )
            # Normalize without allowing the model to shrink the denominator
            # as an escape route.  Scale normalization also makes the tolerance
            # and loss comparable across complexes of different physical size.
            scale = (
                full_distance[off_diagonal]
                .square()
                .mean()
                .clamp_min(1e-6)
                .detach()
            )
            squared = full_distance.square() / scale
            centering = torch_module.eye(
                count, dtype=squared.dtype, device=squared.device
            ) - squared.new_full((count, count), 1.0 / count)
            gram = -0.5 * centering @ squared @ centering
            gram = 0.5 * (gram + gram.transpose(0, 1))
            eigenvalues = torch_module.linalg.eigvalsh(gram)
            negative = functional.relu(
                -eigenvalues - numerical_tolerance
            ).square().mean()
            excess_rank = (
                functional.relu(
                    eigenvalues[:-3] - numerical_tolerance
                ).square().mean()
                if count > 3
                else eigenvalues.new_zeros(())
            )
        losses.append(negative + excess_rank)
    if not losses:
        raise ValueError("EDM supervision selected no complexes")
    return torch_module.stack(losses).mean()


def top_fraction_contact_score(
    contact_logits: Any,
    pair_mask: Any,
    *,
    fraction: float = 0.01,
    temperature: float | Any = 1.0,
) -> Any:
    """Length-normalized differentiable compatibility score per complex."""
    torch_module = _require_torch()
    if tuple(contact_logits.shape) != tuple(pair_mask.shape) or pair_mask.dtype != torch_module.bool:
        raise ValueError("contact_logits and pair_mask must align")
    if not math.isfinite(float(fraction)) or not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    if torch_module.is_tensor(temperature):
        if temperature.numel() != 1:
            raise ValueError("temperature must be scalar")
        scaled_logits = contact_logits / temperature.clamp_min(1e-6)
    else:
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        scaled_logits = contact_logits / float(temperature)
    scores = []
    for batch_index in range(int(contact_logits.shape[0])):
        observed = scaled_logits[batch_index][pair_mask[batch_index]]
        if observed.numel() == 0:
            raise ValueError("each complex must contain candidate pairs")
        count = max(1, math.ceil(fraction * int(observed.numel())))
        scores.append(torch_module.sigmoid(torch_module.topk(observed, count).values).mean())
    return torch_module.stack(scores)


def residue_partner_contrastive_loss(
    cognate_logits: Any,
    decoy_logits: Any,
    labels: Any,
    mask: Any,
    *,
    margin: float = 0.5,
    mode: str = "pairwise_softplus",
    temperature: float = 1.0,
) -> Any:
    """Require the cognate partner to raise interface logits at true sites.

    The complex-level compatibility score supervises one scalar per complex,
    which is far too weak to teach residue-level partner dependence.  This term
    supervises the quantity the evaluation actually measures: at every labelled
    interface residue, the logit computed against the cognate partner must beat
    the logit computed against each decoy partner by ``margin``.

    ``decoy_logits`` has shape ``[batch, residue, decoy]`` and must be indexed
    over the *fixed* chain, so a decoy swaps only the opposite partner and the
    residue axis stays aligned with ``labels``.
    """

    torch_module = _require_torch()
    if cognate_logits.ndim != 2 or decoy_logits.ndim != 3:
        raise ValueError("cognate logits must be [batch, residue] and decoys [batch, residue, decoy]")
    if tuple(decoy_logits.shape[:2]) != tuple(cognate_logits.shape):
        raise ValueError("decoy logits must align with the fixed-chain residue axis")
    if tuple(labels.shape) != tuple(cognate_logits.shape) or tuple(mask.shape) != tuple(
        cognate_logits.shape
    ):
        raise ValueError("labels and mask must align with the cognate logits")
    if mask.dtype != torch_module.bool:
        raise ValueError("mask must be boolean")
    if not math.isfinite(float(margin)) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if mode not in {"pairwise_softplus", "listwise_contrastive"}:
        raise ValueError("mode must be pairwise_softplus or listwise_contrastive")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    positive = mask & (labels > 0.5)
    differences = float(margin) + decoy_logits - cognate_logits.unsqueeze(-1)
    if mode == "pairwise_softplus":
        per_residue = functional.softplus(differences).mean(dim=-1)
    else:
        scaled = differences / float(temperature)
        per_residue = torch.logsumexp(
            torch.cat((torch.zeros_like(scaled[..., :1]), scaled), dim=-1), dim=-1
        )
    per_residue = per_residue * positive
    positive_count = positive.sum(dim=1)
    complex_loss = per_residue.sum(dim=1) / positive_count.clamp_min(1)
    eligible = positive_count > 0
    if not bool(eligible.any()):
        return cognate_logits.sum() * 0.0
    return (complex_loss * eligible).sum() / eligible.sum()


def compute_local_motif_hard_negative_loss(
    contact_logits: Any,
    positive_mask: Any,
    native_local_noncontact_mask: Any,
    matched_local_decoy_mask: Any,
    *,
    margin: float = 0.25,
    temperature: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Rank a native local motif above a geometry-matched local surface decoy.

    ``positive_mask`` and ``matched_local_decoy_mask`` are generated by
    :mod:`local_interface_decoys`; they never contain a frozen/official test
    item.  Native local non-contact is reported as a separate diagnostic tier
    and is *not* silently treated as an inter-complex or batch off-diagonal
    negative.  Each complex receives equal mass, and each motif may have a
    different number of residue pairs.
    """

    torch_module = _require_torch()
    if contact_logits.ndim != 3:
        raise ValueError("contact_logits must have shape [batch, antibody, antigen]")
    shape = tuple(contact_logits.shape)
    masks = (positive_mask, native_local_noncontact_mask, matched_local_decoy_mask)
    if any(tuple(mask.shape) != shape or mask.dtype != torch_module.bool for mask in masks):
        raise ValueError("local motif masks must be boolean and align with contact_logits")
    if not math.isfinite(float(margin)) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    positive_scores: list[Any] = []
    decoy_scores: list[Any] = []
    native_noncontact_scores: list[Any] = []
    losses: list[Any] = []
    for batch_index in range(int(contact_logits.shape[0])):
        positive = contact_logits[batch_index][positive_mask[batch_index]]
        decoy = contact_logits[batch_index][matched_local_decoy_mask[batch_index]]
        native_noncontact = contact_logits[batch_index][
            native_local_noncontact_mask[batch_index]
        ]
        if positive.numel() == 0 or decoy.numel() == 0:
            continue
        positive_score = functional.logsigmoid(positive).mean()
        decoy_score = functional.logsigmoid(decoy).mean()
        losses.append(
            functional.softplus(
                (float(margin) + decoy_score - positive_score) / float(temperature)
            )
        )
        positive_scores.append(positive_score)
        decoy_scores.append(decoy_score)
        if native_noncontact.numel():
            native_noncontact_scores.append(torch_module.sigmoid(native_noncontact).mean())
    if not losses:
        zero = contact_logits.sum() * 0.0
        return zero, {
            "eligible_complexes": 0,
            "positive_score": None,
            "matched_decoy_score": None,
            "native_local_noncontact_score": None,
        }
    loss = torch_module.stack(losses).mean()
    return loss, {
        "eligible_complexes": len(losses),
        "positive_score": torch_module.stack(positive_scores).mean(),
        "matched_decoy_score": torch_module.stack(decoy_scores).mean(),
        "native_local_noncontact_score": (
            torch_module.stack(native_noncontact_scores).mean()
            if native_noncontact_scores
            else None
        ),
    }


@dataclass(frozen=True)
class FrameContactLossWeights:
    contact: float = 1.0
    paratope: float = 0.25
    epitope: float = 0.25
    specificity: float = 0.25
    distogram: float = 0.0
    contact_distance_consistency: float = 0.0
    pair_node_consistency: float = 0.0
    node_spatial_smoothness: float = 0.0
    triangle: float = 0.0
    far_patch_exclusion: float = 0.0
    edm: float = 0.0
    residue_partner_contrastive: float = 0.0
    local_motif_hard_negative: float = 0.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} loss weight must be finite and nonnegative")
        if sum(self.__dict__.values()) <= 0:
            raise ValueError("at least one loss weight must be positive")


class FrameContactObjective(_ModuleBase):
    """Contact, residue-marginal, and non-cognate ranking objective."""

    def __init__(
        self,
        *,
        weights: FrameContactLossWeights = FrameContactLossWeights(),
        specificity_margin: float = 0.05,
        score_fraction: float = 0.01,
        specificity_score_source: str = "interaction_only",
        specificity_loss_mode: str = "pairwise_softplus",
        specificity_temperature: float = 1.0,
        specificity_direction_aggregation: str = "mean",
        specificity_direction_temperature: float = 0.1,
        triangle_neighbor_radius: float = 12.0,
        triangle_neighbor_k: int = 8,
        triangle_slack: float = 0.5,
        node_spatial_neighbor_radius: float = 12.0,
        node_spatial_neighbor_k: int = 8,
        antigen_patch_top_seeds: int = 1,
        antigen_patch_radius: float = 12.0,
        far_patch_minimum_separation: float = 20.0,
        far_patch_residue_k: int = 16,
        far_patch_slack: float = 0.5,
        edm_numerical_tolerance: float = 1e-6,
        residue_contrastive_margin: float = 0.5,
        residue_contrastive_mode: str = "pairwise_softplus",
        residue_contrastive_temperature: float = 1.0,
        local_motif_hard_negative_margin: float = 0.25,
        local_motif_hard_negative_temperature: float = 1.0,
    ) -> None:
        _require_torch()
        super().__init__()
        weights.validate()
        if not math.isfinite(specificity_margin) or specificity_margin < 0:
            raise ValueError("specificity_margin must be finite and nonnegative")
        if not math.isfinite(score_fraction) or not 0 < score_fraction <= 1:
            raise ValueError("score_fraction must lie in (0, 1]")
        if specificity_score_source not in {
            "interaction_only",
            "gauge_interaction",
            "total_contact",
            "contact_bottleneck_cosine",
        }:
            raise ValueError(
                "specificity_score_source must be interaction_only, gauge_interaction, "
                "total_contact, or contact_bottleneck_cosine"
            )
        if specificity_loss_mode not in {
            "pairwise_softplus",
            "listwise_contrastive",
        }:
            raise ValueError(
                "specificity_loss_mode must be pairwise_softplus or "
                "listwise_contrastive"
            )
        if not math.isfinite(specificity_temperature) or specificity_temperature <= 0:
            raise ValueError("specificity_temperature must be finite and positive")
        if specificity_direction_aggregation not in {"mean", "smooth_max"}:
            raise ValueError(
                "specificity_direction_aggregation must be mean or smooth_max"
            )
        if (
            not math.isfinite(specificity_direction_temperature)
            or specificity_direction_temperature <= 0
        ):
            raise ValueError(
                "specificity_direction_temperature must be finite and positive"
            )
        if not math.isfinite(triangle_neighbor_radius) or triangle_neighbor_radius <= 0:
            raise ValueError("triangle_neighbor_radius must be finite and positive")
        if (
            isinstance(triangle_neighbor_k, bool)
            or not isinstance(triangle_neighbor_k, int)
            or triangle_neighbor_k <= 0
        ):
            raise ValueError("triangle_neighbor_k must be positive")
        if not math.isfinite(triangle_slack) or triangle_slack < 0:
            raise ValueError("triangle_slack must be finite and nonnegative")
        if (
            not math.isfinite(node_spatial_neighbor_radius)
            or node_spatial_neighbor_radius <= 0
        ):
            raise ValueError(
                "node_spatial_neighbor_radius must be finite and positive"
            )
        if (
            isinstance(node_spatial_neighbor_k, bool)
            or not isinstance(node_spatial_neighbor_k, int)
            or node_spatial_neighbor_k <= 0
        ):
            raise ValueError("node_spatial_neighbor_k must be positive")
        if (
            isinstance(antigen_patch_top_seeds, bool)
            or not isinstance(antigen_patch_top_seeds, int)
            or antigen_patch_top_seeds <= 0
        ):
            raise ValueError("antigen_patch_top_seeds must be positive")
        if not math.isfinite(antigen_patch_radius) or antigen_patch_radius <= 0:
            raise ValueError("antigen_patch_radius must be finite and positive")
        if (
            not math.isfinite(far_patch_minimum_separation)
            or far_patch_minimum_separation <= 0
        ):
            raise ValueError(
                "far_patch_minimum_separation must be finite and positive"
            )
        if (
            isinstance(far_patch_residue_k, bool)
            or not isinstance(far_patch_residue_k, int)
            or far_patch_residue_k <= 0
        ):
            raise ValueError("far_patch_residue_k must be positive")
        if not math.isfinite(far_patch_slack) or far_patch_slack < 0:
            raise ValueError("far_patch_slack must be finite and nonnegative")
        if not math.isfinite(edm_numerical_tolerance) or edm_numerical_tolerance < 0:
            raise ValueError("edm_numerical_tolerance must be finite and nonnegative")
        if not math.isfinite(residue_contrastive_margin) or residue_contrastive_margin < 0:
            raise ValueError(
                "residue_contrastive_margin must be finite and nonnegative"
            )
        if residue_contrastive_mode not in {
            "pairwise_softplus",
            "listwise_contrastive",
        }:
            raise ValueError(
                "residue_contrastive_mode must be pairwise_softplus or "
                "listwise_contrastive"
            )
        if (
            not math.isfinite(residue_contrastive_temperature)
            or residue_contrastive_temperature <= 0
        ):
            raise ValueError(
                "residue_contrastive_temperature must be finite and positive"
            )
        if (
            not math.isfinite(local_motif_hard_negative_margin)
            or local_motif_hard_negative_margin < 0
        ):
            raise ValueError(
                "local_motif_hard_negative_margin must be finite and nonnegative"
            )
        if (
            not math.isfinite(local_motif_hard_negative_temperature)
            or local_motif_hard_negative_temperature <= 0
        ):
            raise ValueError(
                "local_motif_hard_negative_temperature must be finite and positive"
            )
        self.residue_contrastive_margin = float(residue_contrastive_margin)
        self.residue_contrastive_mode = residue_contrastive_mode
        self.residue_contrastive_temperature = float(residue_contrastive_temperature)
        self.local_motif_hard_negative_margin = float(
            local_motif_hard_negative_margin
        )
        self.local_motif_hard_negative_temperature = float(
            local_motif_hard_negative_temperature
        )
        self.weights = weights
        self.specificity_margin = float(specificity_margin)
        self.score_fraction = float(score_fraction)
        self.specificity_score_source = specificity_score_source
        self.specificity_loss_mode = specificity_loss_mode
        self.specificity_temperature = float(specificity_temperature)
        self.specificity_direction_aggregation = specificity_direction_aggregation
        self.specificity_direction_temperature = float(
            specificity_direction_temperature
        )
        self.triangle_neighbor_radius = float(triangle_neighbor_radius)
        self.triangle_neighbor_k = int(triangle_neighbor_k)
        self.triangle_slack = float(triangle_slack)
        self.node_spatial_neighbor_radius = float(node_spatial_neighbor_radius)
        self.node_spatial_neighbor_k = int(node_spatial_neighbor_k)
        self.antigen_patch_top_seeds = int(antigen_patch_top_seeds)
        self.antigen_patch_radius = float(antigen_patch_radius)
        self.far_patch_minimum_separation = float(
            far_patch_minimum_separation
        )
        self.far_patch_residue_k = int(far_patch_residue_k)
        self.far_patch_slack = float(far_patch_slack)
        self.edm_numerical_tolerance = float(edm_numerical_tolerance)

    def _specificity_logits(self, output: Mapping[str, Any]) -> Any:
        if self.specificity_score_source == "contact_bottleneck_cosine":
            raise ValueError(
                "contact_bottleneck_cosine is a complex-level score, not pair logits"
            )
        if self.specificity_score_source == "total_contact":
            return output["contact_logits"]
        if self.specificity_score_source == "gauge_interaction":
            logits = output.get("contact_gauge_interaction_logits")
            if logits is None:
                raise ValueError(
                    "gauge_interaction specificity requires a model built with "
                    "pair_gauge_mode='doubly_centered'"
                )
            return logits
        if "contact_interaction_logits" not in output:
            raise ValueError(
                "interaction_only specificity requires contact_interaction_logits"
            )
        return output["contact_interaction_logits"]

    def _specificity_score(self, output: Mapping[str, Any]) -> Any:
        if self.specificity_score_source != "contact_bottleneck_cosine":
            return top_fraction_contact_score(
                self._specificity_logits(output),
                output["pair_candidate_mask"],
                fraction=self.score_fraction,
            )
        score = output.get("contact_bottleneck_compatibility")
        if score is None:
            raise ValueError(
                "contact_bottleneck_cosine specificity requires "
                "contact_bottleneck_compatibility"
            )
        pair_mask = output.get("pair_candidate_mask")
        if (
            not torch.is_tensor(score)
            or score.ndim != 1
            or pair_mask is None
            or int(score.shape[0]) != int(pair_mask.shape[0])
            or not bool(torch.isfinite(score).all())
        ):
            raise ValueError(
                "contact_bottleneck_compatibility must be finite with shape [batch]"
            )
        return score

    def _residue_partner_contrastive(
        self,
        cognate_output: Mapping[str, Any],
        cognate_batch: Mapping[str, Any],
        wrong_partner_outputs: Mapping[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        """Contrast cognate against decoy partners at labelled interface residues.

        Each side keeps one chain fixed so the residue axis of the decoy output
        still indexes the same residues as the labels.
        """

        zero = cognate_output["paratope_logits"].sum() * 0.0
        if self.weights.residue_partner_contrastive == 0:
            return zero, {}
        if not wrong_partner_outputs:
            raise ValueError(
                "residue partner contrastive training requires wrong-partner output"
            )
        sides = {
            # Swapping the antigen keeps the antibody axis fixed, so this side
            # supervises paratope logits; the antibody side is the mirror image.
            "paratope": (
                "fixed_antibody_wrong_antigen",
                "paratope_logits",
                "antibody_labels",
                "antibody_mask",
            ),
            "epitope": (
                "fixed_antigen_wrong_antibody",
                "epitope_logits",
                "antigen_labels",
                "antigen_mask",
            ),
        }
        terms: dict[str, Any] = {}
        losses = []
        for side, (direction, logit_key, label_key, mask_key) in sides.items():
            outputs = wrong_partner_outputs.get(direction)
            if outputs is None:
                continue
            local_outputs = (outputs,) if isinstance(outputs, Mapping) else tuple(outputs)
            if not local_outputs:
                continue
            decoy = torch.stack(
                [output[logit_key] for output in local_outputs], dim=-1
            )
            side_loss = residue_partner_contrastive_loss(
                cognate_output[logit_key],
                decoy,
                cognate_batch[label_key],
                cognate_output[mask_key],
                margin=self.residue_contrastive_margin,
                mode=self.residue_contrastive_mode,
                temperature=self.residue_contrastive_temperature,
            )
            terms[side] = side_loss
            losses.append(side_loss)
        if not losses:
            raise ValueError(
                "residue partner contrastive training requires at least one of "
                "fixed_antibody_wrong_antigen or fixed_antigen_wrong_antibody"
            )
        return torch.stack(losses).mean(), terms

    def forward(
        self,
        cognate_output: Mapping[str, Any],
        cognate_batch: Mapping[str, Any],
        *,
        wrong_partner_output: Mapping[str, Any] | None = None,
        wrong_partner_outputs: Mapping[
            str, Mapping[str, Any] | Sequence[Mapping[str, Any]]
        ]
        | None = None,
    ) -> dict[str, Any]:
        required_labels = ("pair_labels", "antibody_labels", "antigen_labels")
        if any(key not in cognate_batch for key in required_labels):
            raise ValueError("cognate batch lacks authenticated training labels")
        contact_loss = balanced_binary_loss_per_complex(
            cognate_output["contact_logits"],
            cognate_batch["pair_labels"],
            cognate_output["pair_candidate_mask"],
        )
        paratope_loss = balanced_binary_loss_per_complex(
            cognate_output["paratope_logits"],
            cognate_batch["antibody_labels"],
            cognate_output["antibody_mask"],
        )
        epitope_loss = balanced_binary_loss_per_complex(
            cognate_output["epitope_logits"],
            cognate_batch["antigen_labels"],
            cognate_output["antigen_mask"],
        )
        if (
            self.weights.distogram != 0
            or self.weights.contact_distance_consistency != 0
            or self.weights.triangle != 0
            or self.weights.far_patch_exclusion != 0
            or self.weights.edm != 0
        ):
            distogram_logits = cognate_output.get("distogram_logits")
            boundaries = cognate_output.get("distance_bin_boundaries")
            if distogram_logits is None or boundaries is None:
                raise ValueError("distance objectives require a distogram pair head")
        else:
            distogram_logits = None
            boundaries = None
        if self.weights.distogram != 0:
            if "pair_distances" not in cognate_batch or "pair_distance_mask" not in cognate_batch:
                raise ValueError("distogram loss requires authenticated pair distances")
            distogram_loss = distogram_cross_entropy_per_complex(
                distogram_logits,
                cognate_batch["pair_distances"],
                cognate_batch["pair_distance_mask"]
                & cognate_output["pair_candidate_mask"],
                boundaries,
            )
        else:
            distogram_loss = contact_loss.new_zeros(())
        if self.weights.contact_distance_consistency != 0:
            distogram_contact_logits = cognate_output.get(
                "distogram_contact_logits"
            )
            direct_contact_logits = cognate_output.get("direct_contact_logits")
            if distogram_contact_logits is None or direct_contact_logits is None:
                raise ValueError(
                    "contact-distance consistency requires a joint contact-distogram head"
                )
            contact_distance_consistency_loss = (
                bernoulli_js_consistency_per_complex(
                    direct_contact_logits,
                    distogram_contact_logits,
                    cognate_output["pair_candidate_mask"],
                )
            )
        else:
            contact_distance_consistency_loss = contact_loss.new_zeros(())
        if self.weights.pair_node_consistency != 0:
            pair_node_keys = (
                "paratope_unary_logits",
                "paratope_contact_logits",
                "antibody_mask",
                "epitope_unary_logits",
                "epitope_contact_logits",
                "antigen_mask",
            )
            missing_pair_node_keys = [
                key for key in pair_node_keys if cognate_output.get(key) is None
            ]
            if missing_pair_node_keys:
                raise ValueError(
                    "pair-node consistency requires independent unary heads and "
                    "contact-derived marginals; missing "
                    + ", ".join(missing_pair_node_keys)
                )
            (
                pair_node_consistency_loss,
                pair_node_consistency_paratope_loss,
                pair_node_consistency_epitope_loss,
            ) = pair_node_consistency_per_complex(
                paratope_unary_logits=cognate_output["paratope_unary_logits"],
                paratope_contact_logits=cognate_output["paratope_contact_logits"],
                antibody_mask=cognate_output["antibody_mask"],
                epitope_unary_logits=cognate_output["epitope_unary_logits"],
                epitope_contact_logits=cognate_output["epitope_contact_logits"],
                antigen_mask=cognate_output["antigen_mask"],
            )
        else:
            pair_node_consistency_loss = contact_loss.new_zeros(())
            pair_node_consistency_paratope_loss = contact_loss.new_zeros(())
            pair_node_consistency_epitope_loss = contact_loss.new_zeros(())
        if self.weights.node_spatial_smoothness != 0:
            required_spatial_outputs = (
                "paratope_unary_logits",
                "antibody_mask",
                "epitope_unary_logits",
                "antigen_mask",
            )
            required_spatial_batch = (
                "antibody_backbone",
                "antibody_chain_ids",
                "antigen_backbone",
                "antigen_chain_ids",
            )
            missing_spatial = [
                key
                for key in required_spatial_outputs
                if cognate_output.get(key) is None
            ] + [
                key for key in required_spatial_batch if cognate_batch.get(key) is None
            ]
            if missing_spatial:
                raise ValueError(
                    "node spatial smoothness requires independent unary heads and "
                    "authenticated monomer geometry; missing "
                    + ", ".join(missing_spatial)
                )
            (
                node_spatial_smoothness_loss,
                node_spatial_smoothness_paratope_loss,
                node_spatial_smoothness_epitope_loss,
            ) = node_spatial_smoothness_per_complex(
                paratope_unary_logits=cognate_output["paratope_unary_logits"],
                antibody_backbone=cognate_batch["antibody_backbone"],
                antibody_mask=cognate_output["antibody_mask"],
                antibody_chain_ids=cognate_batch["antibody_chain_ids"],
                epitope_unary_logits=cognate_output["epitope_unary_logits"],
                antigen_backbone=cognate_batch["antigen_backbone"],
                antigen_mask=cognate_output["antigen_mask"],
                antigen_chain_ids=cognate_batch["antigen_chain_ids"],
                neighbor_radius=self.node_spatial_neighbor_radius,
                neighbor_k=self.node_spatial_neighbor_k,
            )
        else:
            node_spatial_smoothness_loss = contact_loss.new_zeros(())
            node_spatial_smoothness_paratope_loss = contact_loss.new_zeros(())
            node_spatial_smoothness_epitope_loss = contact_loss.new_zeros(())
        if (
            self.weights.triangle != 0
            or self.weights.far_patch_exclusion != 0
            or self.weights.edm != 0
        ):
            if "antibody_cdr_mask" not in cognate_batch:
                raise ValueError(
                    "local geometry loss requires a precomputed antibody CDR mask"
                )
            if "antigen_surface_mask" not in cognate_batch:
                raise ValueError(
                    "local geometry loss requires a precomputed antigen surface mask"
                )
            antigen_patch = select_local_antigen_patch(
                cognate_output["epitope_unary_logits"],
                cognate_batch["antigen_backbone"],
                cognate_output["antigen_mask"],
                cognate_batch["antigen_surface_mask"],
                top_seed_count=self.antigen_patch_top_seeds,
                patch_radius=self.antigen_patch_radius,
            )
            predicted_distances = expected_distance_from_distogram(
                distogram_logits, boundaries
            )
        else:
            antigen_patch = None
            predicted_distances = None
        if self.weights.triangle != 0:
            triangle_loss = sparse_triangle_inequality_loss(
                predicted_distances,
                cognate_output["pair_candidate_mask"],
                cognate_batch["antibody_backbone"],
                cognate_batch["antigen_backbone"],
                cognate_batch["antibody_chain_ids"],
                cognate_batch["antigen_chain_ids"],
                cognate_batch["antibody_cdr_mask"],
                antigen_patch,
                neighbor_radius=self.triangle_neighbor_radius,
                neighbor_k=self.triangle_neighbor_k,
                slack=self.triangle_slack,
            )
        else:
            triangle_loss = contact_loss.new_zeros(())
        if self.weights.far_patch_exclusion != 0:
            assert antigen_patch is not None and predicted_distances is not None
            far_patch_exclusion_loss = far_patch_triangle_exclusion_loss(
                predicted_distances,
                cognate_output["pair_candidate_mask"],
                cognate_batch["antigen_backbone"],
                cognate_batch["antibody_cdr_mask"],
                antigen_patch,
                cognate_batch["antigen_surface_mask"],
                minimum_patch_separation=(
                    self.far_patch_minimum_separation
                ),
                far_residue_k=self.far_patch_residue_k,
                slack=self.far_patch_slack,
            )
        else:
            far_patch_exclusion_loss = contact_loss.new_zeros(())
        if self.weights.edm != 0:
            assert antigen_patch is not None and predicted_distances is not None
            edm_loss = local_edm_realizability_loss(
                predicted_distances,
                cognate_output["pair_candidate_mask"],
                cognate_batch["antibody_backbone"],
                cognate_batch["antigen_backbone"],
                cognate_batch["antibody_cdr_mask"],
                antigen_patch,
                numerical_tolerance=self.edm_numerical_tolerance,
            )
        else:
            edm_loss = contact_loss.new_zeros(())
        if wrong_partner_output is not None and wrong_partner_outputs is not None:
            raise ValueError(
                "provide wrong_partner_output or wrong_partner_outputs, not both"
            )
        if wrong_partner_output is not None:
            wrong_partner_outputs = {"legacy": wrong_partner_output}
        cognate_score = self._specificity_score(cognate_output)
        direction_margins: dict[str, Any] = {}
        if wrong_partner_outputs is None:
            if self.weights.specificity != 0:
                raise ValueError("specificity-weighted training requires wrong-partner output")
            specificity_loss = cognate_score.new_zeros(())
            wrong_score = None
        else:
            if not wrong_partner_outputs:
                raise ValueError("wrong_partner_outputs cannot be empty")
            wrong_scores = []
            directional_losses = []
            negative_counts: dict[str, int] = {}
            for direction, outputs in sorted(wrong_partner_outputs.items()):
                local_outputs = (
                    (outputs,)
                    if isinstance(outputs, Mapping)
                    else tuple(outputs)
                )
                if not local_outputs:
                    raise ValueError(
                        f"wrong-partner direction {direction!r} cannot be empty"
                    )
                local_scores = []
                for output in local_outputs:
                    local_score = self._specificity_score(output)
                    if tuple(local_score.shape) != tuple(cognate_score.shape):
                        raise ValueError(
                            "cognate and wrong-partner batches must align by focal complex"
                        )
                    local_scores.append(local_score)
                local_score_matrix = torch.stack(local_scores, dim=-1)
                negative_counts[str(direction)] = len(local_outputs)
                wrong_scores.extend(local_scores)
                direction_margins[str(direction)] = (
                    cognate_score - local_score_matrix.max(dim=-1).values
                ).mean()
                differences = (
                    self.specificity_margin
                    + local_score_matrix
                    - cognate_score.unsqueeze(-1)
                )
                if self.specificity_loss_mode == "pairwise_softplus":
                    direction_loss = functional.softplus(differences).mean(dim=-1)
                else:
                    scaled = differences / self.specificity_temperature
                    direction_loss = torch.logsumexp(
                        torch.cat(
                            (torch.zeros_like(scaled[:, :1]), scaled), dim=-1
                        ),
                        dim=-1,
                    )
                directional_losses.append(direction_loss)
            wrong_score = torch.stack(wrong_scores).mean(dim=0)
            directional_loss_matrix = torch.stack(directional_losses, dim=-1)
            if self.specificity_direction_aggregation == "mean":
                per_complex_specificity_loss = directional_loss_matrix.mean(dim=-1)
            else:
                temperature = self.specificity_direction_temperature
                per_complex_specificity_loss = temperature * (
                    torch.logsumexp(
                        directional_loss_matrix / temperature, dim=-1
                    )
                    - math.log(directional_loss_matrix.shape[-1])
                )
            specificity_loss = per_complex_specificity_loss.mean()
        residue_contrastive_loss, residue_contrastive_terms = (
            self._residue_partner_contrastive(
                cognate_output, cognate_batch, wrong_partner_outputs
            )
        )
        if self.weights.local_motif_hard_negative != 0:
            local_motif_masks = (
                "local_motif_positive_mask",
                "local_motif_native_noncontact_mask",
                "local_motif_matched_decoy_mask",
            )
            missing_local_motif = [
                key
                for key in local_motif_masks
                if cognate_batch.get(key) is None
            ]
            if missing_local_motif:
                raise ValueError(
                    "local motif hard-negative loss requires development masks; missing "
                    + ", ".join(missing_local_motif)
                )
            local_motif_hard_negative_loss, local_motif_terms = (
                compute_local_motif_hard_negative_loss(
                    cognate_output["contact_logits"],
                    cognate_batch["local_motif_positive_mask"],
                    cognate_batch["local_motif_native_noncontact_mask"],
                    cognate_batch["local_motif_matched_decoy_mask"],
                    margin=self.local_motif_hard_negative_margin,
                    temperature=self.local_motif_hard_negative_temperature,
                )
            )
        else:
            local_motif_hard_negative_loss = contact_loss.new_zeros(())
            local_motif_terms = {
                "eligible_complexes": 0,
                "positive_score": None,
                "matched_decoy_score": None,
                "native_local_noncontact_score": None,
            }
        total = (
            self.weights.contact * contact_loss
            + self.weights.paratope * paratope_loss
            + self.weights.epitope * epitope_loss
            + self.weights.specificity * specificity_loss
            + self.weights.distogram * distogram_loss
            + self.weights.contact_distance_consistency
            * contact_distance_consistency_loss
            + self.weights.pair_node_consistency * pair_node_consistency_loss
            + self.weights.node_spatial_smoothness
            * node_spatial_smoothness_loss
            + self.weights.triangle * triangle_loss
            + self.weights.far_patch_exclusion * far_patch_exclusion_loss
            + self.weights.edm * edm_loss
            + self.weights.residue_partner_contrastive * residue_contrastive_loss
            + self.weights.local_motif_hard_negative
            * local_motif_hard_negative_loss
        )
        return {
            "loss": total,
            "contact_loss": contact_loss,
            "paratope_loss": paratope_loss,
            "epitope_loss": epitope_loss,
            "specificity_loss": specificity_loss,
            "distogram_loss": distogram_loss,
            "contact_distance_consistency_loss": (
                contact_distance_consistency_loss
            ),
            "pair_node_consistency_loss": pair_node_consistency_loss,
            "pair_node_consistency_paratope_loss": (
                pair_node_consistency_paratope_loss
            ),
            "pair_node_consistency_epitope_loss": (
                pair_node_consistency_epitope_loss
            ),
            "node_spatial_smoothness_loss": node_spatial_smoothness_loss,
            "node_spatial_smoothness_paratope_loss": (
                node_spatial_smoothness_paratope_loss
            ),
            "node_spatial_smoothness_epitope_loss": (
                node_spatial_smoothness_epitope_loss
            ),
            "triangle_loss": triangle_loss,
            "far_patch_exclusion_loss": far_patch_exclusion_loss,
            "edm_loss": edm_loss,
            "residue_partner_contrastive_loss": residue_contrastive_loss,
            "residue_partner_contrastive_by_side": residue_contrastive_terms,
            "local_motif_hard_negative_loss": local_motif_hard_negative_loss,
            "local_motif_hard_negative_diagnostics": local_motif_terms,
            "mean_cognate_compatibility": cognate_score.mean(),
            "mean_wrong_compatibility": (
                None if wrong_score is None else wrong_score.mean()
            ),
            "mean_compatibility_margin": (
                None if wrong_score is None else (cognate_score - wrong_score).mean()
            ),
            "specificity_margin_by_direction": direction_margins,
            "specificity_score_source": self.specificity_score_source,
            "specificity_loss_mode": self.specificity_loss_mode,
            "specificity_direction_aggregation": (
                self.specificity_direction_aggregation
            ),
            "specificity_negative_count_by_direction": (
                {} if wrong_partner_outputs is None else negative_counts
            ),
        }


class ValidationTemperatureScaler(_ModuleBase):
    """One positive temperature fitted only from validation contact logits."""

    def __init__(self) -> None:
        _require_torch()
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> Any:
        return self.log_temperature.exp().clamp(1e-3, 1e3)

    def forward(self, logits: Any) -> Any:
        return logits / self.temperature

    def fit(self, logits: Any, labels: Any, mask: Any, *, max_iter: int = 50) -> float:
        if tuple(logits.shape) != tuple(labels.shape) or tuple(logits.shape) != tuple(mask.shape):
            raise ValueError("validation logits, labels, and mask must align")
        observed_logits = logits.detach()[mask].float()
        observed_labels = labels.detach()[mask].float()
        if observed_logits.numel() == 0 or not bool(
            ((observed_labels == 0) | (observed_labels == 1)).all()
        ):
            raise ValueError("validation calibration labels are empty or non-binary")
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> Any:
            optimizer.zero_grad()
            loss = functional.binary_cross_entropy_with_logits(
                observed_logits / self.temperature, observed_labels
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach())


__all__ = [
    "FrameContactLossWeights",
    "FrameContactObjective",
    "ValidationTemperatureScaler",
    "balanced_binary_loss_per_complex",
    "bernoulli_js_consistency_per_complex",
    "distogram_bin_centers",
    "distogram_cross_entropy_per_complex",
    "expected_distance_from_distogram",
    "far_patch_triangle_exclusion_loss",
    "local_edm_realizability_loss",
    "compute_local_motif_hard_negative_loss",
    "node_spatial_smoothness_per_complex",
    "pair_node_consistency_per_complex",
    "residue_partner_contrastive_loss",
    "select_local_antigen_patch",
    "sparse_triangle_inequality_loss",
    "top_fraction_contact_score",
]
