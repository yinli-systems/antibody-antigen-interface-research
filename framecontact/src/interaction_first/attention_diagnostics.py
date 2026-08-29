"""Streaming, head-resolved diagnostics for geometry-aware self-attention."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch


def _finite_mean(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-head sums and counts for ``[B,H,L]`` values."""
    selected = mask[:, None, :].expand_as(values) & torch.isfinite(values)
    return (
        values.masked_fill(~selected, 0.0).sum(dim=(0, 2)).double().cpu(),
        selected.sum(dim=(0, 2)).double().cpu(),
    )


class AttentionDiagnosticsAccumulator:
    """Accumulate comparable attention statistics without retaining pair matrices.

    Results are reported per partner, layer, and head. Geometry-conditioned
    distance statistics are normalized only over pairs with valid local frames;
    missing-frame residues remain visible through ``valid_geometry_mass``.
    """

    def __init__(
        self,
        *,
        distance_cutoffs: Sequence[float] = (8.0, 12.0, 16.0),
        sequence_cutoffs: Sequence[int] = (8, 16, 32),
    ) -> None:
        if not distance_cutoffs or any(value <= 0 for value in distance_cutoffs):
            raise ValueError("distance_cutoffs must contain positive values")
        if not sequence_cutoffs or any(value <= 0 for value in sequence_cutoffs):
            raise ValueError("sequence_cutoffs must contain positive values")
        self.distance_cutoffs = tuple(float(value) for value in distance_cutoffs)
        self.sequence_cutoffs = tuple(int(value) for value in sequence_cutoffs)
        self._sums: dict[tuple[str, int, str], torch.Tensor] = defaultdict(
            lambda: torch.empty(0, dtype=torch.float64)
        )
        self._counts: dict[tuple[str, int, str], torch.Tensor] = defaultdict(
            lambda: torch.empty(0, dtype=torch.float64)
        )

    def _add(
        self,
        partner: str,
        layer_index: int,
        metric: str,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        sums, counts = _finite_mean(values, mask)
        key = (partner, layer_index, metric)
        if self._sums[key].numel() == 0:
            self._sums[key] = sums
            self._counts[key] = counts
        else:
            if self._sums[key].shape != sums.shape:
                raise ValueError("Attention head count changed between batches")
            self._sums[key] += sums
            self._counts[key] += counts

    def update(
        self,
        partner: str,
        details: Mapping[str, Any],
        residue_mask: torch.Tensor,
    ) -> None:
        if partner not in {"antibody", "antigen"}:
            raise ValueError("partner must be antibody or antigen")
        distance = details["distance"].float()
        neighbor_mask = details["neighbor_mask"].bool()
        residue_count = int(residue_mask.shape[-1])
        positions = torch.arange(residue_count, device=residue_mask.device)
        sequence_delta = (positions[None, :] - positions[:, None]).abs()
        sequence_delta = sequence_delta[None, :, :]

        for layer_index, layer in enumerate(details["layers"]):
            weights = layer["weights"].float()
            content_only = layer["content_only_weights"].float()
            geometry_valid = layer["geometry_valid_mask"].bool()
            query_mask = residue_mask.bool()
            degree = neighbor_mask.sum(dim=-1).float()

            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
            entropy_denominator = degree.clamp_min(2.0).log()[:, None, :]
            normalized_entropy = entropy / entropy_denominator
            entropy_mask = query_mask & (degree > 1)
            self._add(
                partner, layer_index, "normalized_entropy", normalized_entropy, entropy_mask
            )

            total_variation = 0.5 * (weights - content_only).abs().sum(dim=-1)
            self._add(
                partner,
                layer_index,
                "geometry_counterfactual_total_variation",
                total_variation,
                query_mask,
            )

            valid_geometry_mass = (
                weights * geometry_valid[:, None, :, :].to(weights.dtype)
            ).sum(dim=-1)
            self._add(
                partner,
                layer_index,
                "valid_geometry_attention_mass",
                valid_geometry_mass,
                query_mask,
            )

            valid_mass = valid_geometry_mass.clamp_min(1e-12)
            valid_query = query_mask & geometry_valid.any(dim=-1)
            expected_distance = (
                weights
                * distance[:, None, :, :]
                * geometry_valid[:, None, :, :].to(weights.dtype)
            ).sum(dim=-1) / valid_mass
            self._add(
                partner,
                layer_index,
                "expected_ca_distance_angstrom",
                expected_distance,
                valid_query,
            )

            valid_candidate_count = geometry_valid.sum(dim=-1).clamp_min(1)
            for cutoff in self.distance_cutoffs:
                close = geometry_valid & (distance <= cutoff)
                conditional_mass = (
                    weights * close[:, None, :, :].to(weights.dtype)
                ).sum(dim=-1) / valid_mass
                uniform_fraction = (
                    close.sum(dim=-1).float() / valid_candidate_count.float()
                )[:, None, :]
                label = f"ca_within_{cutoff:g}a"
                self._add(
                    partner,
                    layer_index,
                    f"{label}_conditional_attention_mass",
                    conditional_mass,
                    valid_query,
                )
                self._add(
                    partner,
                    layer_index,
                    f"{label}_mass_lift_over_uniform",
                    conditional_mass - uniform_fraction,
                    valid_query,
                )

            for cutoff in self.sequence_cutoffs:
                close = neighbor_mask & (sequence_delta <= cutoff)
                mass = (weights * close[:, None, :, :].to(weights.dtype)).sum(dim=-1)
                self._add(
                    partner,
                    layer_index,
                    f"sequence_within_{cutoff}_attention_mass",
                    mass,
                    query_mask,
                )

            permitted = neighbor_mask[:, None, :, :].expand_as(weights)
            bias = layer["geometry_bias"].float()
            bias_count = permitted.sum(dim=(0, 2, 3)).double().cpu()
            bias_abs_sum = bias.abs().masked_fill(~permitted, 0.0).sum(
                dim=(0, 2, 3)
            ).double().cpu()
            bias_square_sum = bias.square().masked_fill(~permitted, 0.0).sum(
                dim=(0, 2, 3)
            ).double().cpu()
            for metric, value in (
                ("geometry_bias_abs", bias_abs_sum),
                ("geometry_bias_square", bias_square_sum),
            ):
                key = (partner, layer_index, metric)
                if self._sums[key].numel() == 0:
                    self._sums[key] = value
                    self._counts[key] = bias_count.clone()
                else:
                    self._sums[key] += value
                    self._counts[key] += bias_count

    def result(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "definition": {
                "distance_conditioning": (
                    "conditioned on attention mass over pairs with two valid N/CA/C frames"
                ),
                "mass_lift": (
                    "conditional attention mass minus the uniform fraction on the same graph"
                ),
                "geometry_counterfactual": (
                    "total variation from removing only additive geometry bias on the same graph"
                ),
            },
            "partners": {},
        }
        partners = sorted({key[0] for key in self._sums})
        for partner in partners:
            layer_indices = sorted({key[1] for key in self._sums if key[0] == partner})
            layers = []
            for layer_index in layer_indices:
                metrics: dict[str, list[float | None]] = {}
                keys = sorted(
                    key for key in self._sums if key[:2] == (partner, layer_index)
                )
                for key in keys:
                    sums = self._sums[key]
                    counts = self._counts[key]
                    if key[2] == "geometry_bias_square":
                        continue
                    if key[2] == "geometry_bias_abs":
                        square_key = (partner, layer_index, "geometry_bias_square")
                        mean_square = self._sums[square_key] / self._counts[square_key].clamp_min(1)
                        metrics["geometry_bias_rms"] = [
                            float(value) for value in mean_square.sqrt().tolist()
                        ]
                    means = sums / counts.clamp_min(1)
                    metrics[f"mean_{key[2]}"] = [
                        float(value) if count > 0 else None
                        for value, count in zip(means.tolist(), counts.tolist())
                    ]
                layers.append(
                    {
                        "layer_index": layer_index,
                        "head_count": int(next(iter(self._sums[key].shape for key in keys))[0]),
                        "metrics_per_head": metrics,
                    }
                )
            output["partners"][partner] = {"layers": layers}
        return output


class CrossAttentionDiagnosticsAccumulator:
    """Streaming diagnostics for late or layer-interleaved cross-attention."""

    def __init__(self) -> None:
        self._sums: dict[tuple[str, int, str], torch.Tensor] = {}
        self._counts: dict[tuple[str, int, str], torch.Tensor] = {}
        self._parameters: dict[tuple[str, int], dict[str, Any]] = {}

    def _add(
        self,
        direction: str,
        layer_index: int,
        metric: str,
        values: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> None:
        selected = query_mask[:, None, :].expand_as(values) & torch.isfinite(values)
        sums = values.masked_fill(~selected, 0.0).sum(dim=(0, 2)).double().cpu()
        counts = selected.sum(dim=(0, 2)).double().cpu()
        key = (direction, layer_index, metric)
        if key not in self._sums:
            self._sums[key] = sums
            self._counts[key] = counts
        else:
            if self._sums[key].shape != sums.shape:
                raise ValueError("Cross-attention head count changed between batches")
            self._sums[key] += sums
            self._counts[key] += counts

    def update(
        self,
        details: Mapping[str, Any],
        antibody_mask: torch.Tensor,
        antigen_mask: torch.Tensor,
    ) -> None:
        if details.get("fusion_mode") == "interleaved":
            layers = details["layers"]
        else:
            layers = (details,)
        directions = (
            ("antibody_from_antigen", antibody_mask),
            ("antigen_from_antibody", antigen_mask),
        )
        for layer_index, layer in enumerate(layers):
            for direction, receiver_mask in directions:
                directional = layer[direction]
                weights = directional["weights"].float()
                content_only = directional["content_only_weights"].float()
                pair_mask = directional["pair_mask"].bool()
                candidate_count = pair_mask.sum(dim=-1)
                query_mask = receiver_mask.bool() & (candidate_count > 0)
                entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
                normalized_entropy = entropy / candidate_count.clamp_min(2).log()[:, None, :]
                self._add(
                    direction,
                    layer_index,
                    "normalized_entropy",
                    normalized_entropy,
                    query_mask & (candidate_count > 1),
                )
                total_variation = 0.5 * (weights - content_only).abs().sum(dim=-1)
                self._add(
                    direction,
                    layer_index,
                    "contact_gate_total_variation",
                    total_variation,
                    query_mask,
                )
                self._parameters[(direction, layer_index)] = {
                    "contact_scale_per_head": [
                        float(value)
                        for value in directional["contact_scale_per_head"]
                        .detach()
                        .float()
                        .cpu()
                        .tolist()
                    ],
                    "residual_gate": float(
                        directional["residual_gate"].detach().float().cpu()
                    ),
                }

    def result(self) -> dict[str, Any]:
        output: dict[str, Any] = {"directions": {}}
        directions = sorted({key[0] for key in self._sums})
        for direction in directions:
            layer_indices = sorted(
                {key[1] for key in self._sums if key[0] == direction}
            )
            layers = []
            for layer_index in layer_indices:
                metrics = {}
                for key in sorted(
                    key
                    for key in self._sums
                    if key[:2] == (direction, layer_index)
                ):
                    means = self._sums[key] / self._counts[key].clamp_min(1)
                    metrics[f"mean_{key[2]}_per_head"] = [
                        float(value) if count > 0 else None
                        for value, count in zip(
                            means.tolist(), self._counts[key].tolist()
                        )
                    ]
                parameters = self._parameters[(direction, layer_index)]
                layers.append(
                    {
                        "layer_index": layer_index,
                        "metrics": metrics,
                        **parameters,
                    }
                )
            output["directions"][direction] = {"layers": layers}
        return output


__all__ = [
    "AttentionDiagnosticsAccumulator",
    "CrossAttentionDiagnosticsAccumulator",
]
