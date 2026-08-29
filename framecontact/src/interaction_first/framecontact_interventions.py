"""Deterministic causal controls for FrameContact experiments."""

from __future__ import annotations

from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from .intra_chain_geometry import (
    build_backbone_frames,
    build_intra_chain_relative_geometry,
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for FrameContact interventions")
    return torch


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    torch_module = _require_torch()
    return {
        key: value.clone() if torch_module.is_tensor(value) else value
        for key, value in batch.items()
    }


def _random_rotations(
    shape: tuple[int, ...],
    *,
    device: Any,
    dtype: Any,
    generator: Any,
) -> Any:
    torch_module = _require_torch()
    matrices = torch_module.randn(
        *shape, 3, 3, device=device, dtype=dtype, generator=generator
    )
    rotations, _ = torch_module.linalg.qr(matrices)
    determinants = torch_module.linalg.det(rotations)
    correction = torch_module.ones(*shape, 3, device=device, dtype=dtype)
    correction[..., -1] = torch_module.where(
        determinants < 0,
        -torch_module.ones_like(determinants),
        torch_module.ones_like(determinants),
    )
    return rotations * correction[..., None, :]


def _refresh_frame_masks(batch: dict[str, Any], partner: str) -> None:
    residue_mask = batch[f"{partner}_mask"]
    valid = build_backbone_frames(batch[f"{partner}_backbone"], residue_mask).valid_mask
    batch[f"{partner}_frame_valid_mask"] = valid
    batch[f"{partner}_missing_frame_mask"] = residue_mask & ~valid


def independent_monomer_rigid_transform(
    batch: Mapping[str, Any],
    *,
    generator: Any,
    translation_scale: float = 50.0,
) -> dict[str, Any]:
    """Apply different SE(3) transforms to antibody and antigen monomers."""
    torch_module = _require_torch()
    if translation_scale < 0:
        raise ValueError("translation_scale must be nonnegative")
    result = _clone_batch(batch)
    for partner in ("antibody", "antigen"):
        backbone = result[f"{partner}_backbone"]
        batch_size = int(backbone.shape[0])
        rotation = _random_rotations(
            (batch_size,),
            device=backbone.device,
            dtype=backbone.dtype,
            generator=generator,
        )
        translation = torch_module.randn(
            batch_size,
            3,
            device=backbone.device,
            dtype=backbone.dtype,
            generator=generator,
        ) * float(translation_scale)
        result[f"{partner}_backbone"] = (
            torch_module.einsum("bij,brkj->brki", rotation, backbone)
            + translation[:, None, None, :]
        )
        _refresh_frame_masks(result, partner)
    result["intervention"] = "independent_monomer_rigid_transform"
    return result


def geometry_shuffle(
    batch: Mapping[str, Any],
    *,
    generator: Any,
    partners: tuple[str, ...] = ("antibody", "antigen"),
) -> dict[str, Any]:
    """Permute residue frames while retaining sequence, identity, and targets."""
    torch_module = _require_torch()
    result = _clone_batch(batch)
    for partner in partners:
        if partner not in {"antibody", "antigen"}:
            raise ValueError("partners may contain only antibody and antigen")
        backbone = result[f"{partner}_backbone"]
        original = backbone.clone()
        residue_mask = result[f"{partner}_mask"]
        for batch_index in range(int(backbone.shape[0])):
            indices = torch_module.nonzero(
                residue_mask[batch_index], as_tuple=False
            ).flatten()
            permutation = indices[
                torch_module.randperm(
                    len(indices), device=indices.device, generator=generator
                )
            ]
            backbone[batch_index, indices] = original[batch_index, permutation]
        _refresh_frame_masks(result, partner)
    result["intervention"] = "geometry_shuffle"
    return result


def corrupt_local_frames(
    batch: Mapping[str, Any],
    *,
    generator: Any,
    partners: tuple[str, ...] = ("antibody", "antigen"),
) -> dict[str, Any]:
    """Rotate N and C offsets independently around each fixed CA origin."""
    torch_module = _require_torch()
    result = _clone_batch(batch)
    for partner in partners:
        if partner not in {"antibody", "antigen"}:
            raise ValueError("partners may contain only antibody and antigen")
        backbone = result[f"{partner}_backbone"]
        ca_atom = backbone[..., 1, :]
        rotations = _random_rotations(
            tuple(backbone.shape[:2]) + (2,),
            device=backbone.device,
            dtype=backbone.dtype,
            generator=generator,
        )
        offsets = torch_module.stack(
            (backbone[..., 0, :] - ca_atom, backbone[..., 2, :] - ca_atom), dim=-2
        )
        rotated = torch_module.einsum("...ij,...j->...i", rotations, offsets)
        backbone[..., 0, :] = ca_atom + rotated[..., 0, :]
        backbone[..., 2, :] = ca_atom + rotated[..., 1, :]
        _refresh_frame_masks(result, partner)
    result["intervention"] = "local_frame_corruption"
    return result


def perturb_coordinates(
    batch: Mapping[str, Any],
    *,
    generator: Any,
    standard_deviation: float,
    partners: tuple[str, ...] = ("antibody", "antigen"),
) -> dict[str, Any]:
    """Add sequence-preserving Gaussian noise to every observed backbone atom."""
    torch_module = _require_torch()
    if standard_deviation < 0:
        raise ValueError("standard_deviation must be nonnegative")
    result = _clone_batch(batch)
    for partner in partners:
        if partner not in {"antibody", "antigen"}:
            raise ValueError("partners may contain only antibody and antigen")
        backbone = result[f"{partner}_backbone"]
        finite = torch_module.isfinite(backbone)
        noise = torch_module.randn(
            backbone.shape,
            device=backbone.device,
            dtype=backbone.dtype,
            generator=generator,
        ) * float(standard_deviation)
        result[f"{partner}_backbone"] = torch_module.where(
            finite, backbone + noise, backbone
        )
        _refresh_frame_masks(result, partner)
    result["intervention"] = f"coordinate_noise_{standard_deviation:g}_angstrom"
    return result


def derangement(batch_size: int, *, device: Any, generator: Any) -> Any:
    """Return a deterministic-with-generator permutation with no fixed points."""
    torch_module = _require_torch()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
        raise ValueError("partner shuffle requires batch_size >= 2")
    base = torch_module.arange(batch_size, device=device)
    for _ in range(128):
        permutation = torch_module.randperm(
            batch_size, device=device, generator=generator
        )
        if bool((permutation != base).all()):
            return permutation
    # Always valid and deterministic even for an adversarial generator stream.
    return torch_module.roll(base, shifts=1)


def partner_shuffle(
    batch: Mapping[str, Any],
    *,
    partner: str,
    permutation: Any,
) -> dict[str, Any]:
    """Swap one partner and invalidate only the now-undefined pair targets."""
    torch_module = _require_torch()
    if partner not in {"antibody", "antigen"}:
        raise ValueError("partner must be antibody or antigen")
    batch_size = int(batch[f"{partner}_mask"].shape[0])
    if (
        not torch_module.is_tensor(permutation)
        or permutation.dtype not in (torch_module.int32, torch_module.int64)
        or tuple(permutation.shape) != (batch_size,)
        or set(permutation.detach().cpu().tolist()) != set(range(batch_size))
    ):
        raise ValueError("permutation must contain every batch index exactly once")
    result = _clone_batch(batch)
    prefix = f"{partner}_"
    for key, value in tuple(result.items()):
        if key.startswith(prefix) and torch_module.is_tensor(value) and int(value.shape[0]) == batch_size:
            result[key] = value.index_select(0, permutation.to(value.device))
    result.pop("pair_labels", None)
    result.pop("pair_candidate_mask", None)
    result["intervention"] = f"{partner}_partner_shuffle"
    result["partner_permutation"] = permutation.clone()
    return result


def rewire_neighbor_mask(
    neighbor_mask: Any,
    residue_mask: Any,
    chain_ids: Any,
    *,
    generator: Any,
) -> Any:
    """Randomize directed off-diagonal edges while preserving every row degree."""
    torch_module = _require_torch()
    if (
        neighbor_mask.dtype != torch_module.bool
        or residue_mask.dtype != torch_module.bool
        or tuple(neighbor_mask.shape) != (*residue_mask.shape, residue_mask.shape[1])
        or tuple(chain_ids.shape) != tuple(residue_mask.shape)
    ):
        raise ValueError("neighbor graph tensors have inconsistent shapes")
    rewired = torch_module.zeros_like(neighbor_mask)
    batch_size, residue_count = residue_mask.shape
    for batch_index in range(int(batch_size)):
        for query_index in range(int(residue_count)):
            if not bool(residue_mask[batch_index, query_index]):
                continue
            rewired[batch_index, query_index, query_index] = True
            off_diagonal_degree = int(neighbor_mask[batch_index, query_index].sum()) - 1
            candidates = torch_module.nonzero(
                residue_mask[batch_index]
                & (chain_ids[batch_index] == chain_ids[batch_index, query_index])
                & (
                    torch_module.arange(residue_count, device=residue_mask.device)
                    != query_index
                ),
                as_tuple=False,
            ).flatten()
            if off_diagonal_degree < 0 or off_diagonal_degree > len(candidates):
                raise ValueError("neighbor_mask has an invalid same-chain row degree")
            if off_diagonal_degree:
                order = torch_module.randperm(
                    len(candidates), device=candidates.device, generator=generator
                )[:off_diagonal_degree]
                rewired[batch_index, query_index, candidates[order]] = True
    return rewired


def build_rewired_neighbor_override(
    backbone: Any,
    residue_mask: Any,
    chain_ids: Any,
    *,
    generator: Any,
    neighbor_k: int | None = 32,
    neighbor_radius: float | None = 16.0,
) -> Any:
    """Build the reference spatial graph and return its degree-matched null."""
    geometry = build_intra_chain_relative_geometry(
        build_backbone_frames(backbone, residue_mask),
        residue_mask,
        chain_ids=chain_ids,
        attention_graph_mode="spatial",
        neighbor_k=neighbor_k,
        neighbor_radius=neighbor_radius,
    )
    return rewire_neighbor_mask(
        geometry.neighbor_mask,
        residue_mask,
        chain_ids,
        generator=generator,
    )


def degree_matched_graph_rewiring(
    batch: Mapping[str, Any],
    *,
    generator: Any,
    neighbor_k: int | None = 32,
    neighbor_radius: float | None = 16.0,
    partners: tuple[str, ...] = ("antibody", "antigen"),
) -> dict[str, Any]:
    """Attach degree-matched same-chain null graphs to a copied batch."""
    result = _clone_batch(batch)
    for partner in partners:
        if partner not in {"antibody", "antigen"}:
            raise ValueError("partners may contain only antibody and antigen")
        result[f"{partner}_neighbor_mask_override"] = build_rewired_neighbor_override(
            result[f"{partner}_backbone"],
            result[f"{partner}_mask"],
            result[f"{partner}_chain_ids"],
            generator=generator,
            neighbor_k=neighbor_k,
            neighbor_radius=neighbor_radius,
        )
    result["intervention"] = "degree_matched_graph_rewiring"
    return result


__all__ = [
    "build_rewired_neighbor_override",
    "corrupt_local_frames",
    "degree_matched_graph_rewiring",
    "derangement",
    "geometry_shuffle",
    "independent_monomer_rigid_transform",
    "partner_shuffle",
    "perturb_coordinates",
    "rewire_neighbor_mask",
]
