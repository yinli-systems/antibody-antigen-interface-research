"""Asymmetric CALM-style partner retrieval and masked contrastive loss."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, Tuple

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    """Fail with a focused message when PyTorch is unavailable."""
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for asymmetric retrieval operations")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_dropout(value: float) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("dropout must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return result


def _validate_residue_batch(features: Any, mask: Any, input_dim: int) -> None:
    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError("features must have shape [batch, residues, channels]")
    if not torch.is_floating_point(features) or int(features.shape[-1]) != input_dim:
        raise ValueError(f"features must be floating point with dimension {input_dim}")
    if any(int(size) == 0 for size in features.shape):
        raise ValueError("features cannot contain an empty dimension")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional boolean tensor")
    if tuple(mask.shape) != tuple(features.shape[:2]) or mask.device != features.device:
        raise ValueError("mask must align with features and use the same device")
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("Every item must contain at least one valid residue")
    valid = features.masked_select(mask.unsqueeze(-1))
    if not bool(torch.isfinite(valid).all()):
        raise ValueError("Valid residue features must be finite")


def _l2_normalize(values: Any) -> Any:
    if values.ndim != 2 or not torch.is_floating_point(values):
        raise ValueError("Projected embeddings must be a floating-point matrix")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Projected embeddings must be finite")
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    threshold = max(1e-12, float(torch.finfo(values.dtype).tiny))
    if bool((norms <= threshold).any()):
        raise ValueError("Projected embeddings cannot contain zero-norm rows")
    return values / norms


class CALMMeanPooler(_ModuleBase):
    """Project residues before strict CALM mean pooling or an attention ablation."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        chain_balanced: bool = False,
        pooling: str = "mean",
    ) -> None:
        require_torch()
        super().__init__()
        self.input_dim = _positive_integer(input_dim, "input_dim")
        embedding_dim = _positive_integer(embedding_dim, "embedding_dim")
        hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        dropout = _validate_dropout(dropout)
        if not isinstance(chain_balanced, bool):
            raise ValueError("chain_balanced must be boolean")
        if pooling not in {"mean", "attention"}:
            raise ValueError("pooling must be 'mean' or 'attention'")
        self.chain_balanced = chain_balanced
        self.pooling = pooling
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.attention_score = (
            nn.Linear(embedding_dim, 1, bias=False)
            if pooling == "attention"
            else None
        )

    @staticmethod
    def _masked_mean(values: Any, mask: Any) -> Any:
        weights = mask.unsqueeze(-1).to(dtype=values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _validate_chain_ids(mask: Any, chain_ids: Any) -> None:
        if not torch.is_tensor(chain_ids) or chain_ids.dtype != torch.long:
            raise ValueError("chain_ids must be a two-dimensional int64 tensor")
        if tuple(chain_ids.shape) != tuple(mask.shape) or chain_ids.device != mask.device:
            raise ValueError("chain_ids must align with mask and use the same device")
        if bool(((chain_ids[mask] < 0) | (chain_ids[mask] > 1)).any()):
            raise ValueError("Valid antibody residues must use chain ID 0 or 1")
        if bool((chain_ids[~mask] != -1).any()):
            raise ValueError("Padded antibody residues must use chain ID -1")

    def _chain_balanced_mean(self, values: Any, mask: Any, chain_ids: Any) -> Any:
        self._validate_chain_ids(mask, chain_ids)
        chain_means = []
        chain_present = []
        for chain_id in (0, 1):
            chain_mask = mask & (chain_ids == chain_id)
            chain_means.append(self._masked_mean(values, chain_mask))
            chain_present.append(chain_mask.any(dim=1))
        stacked_means = torch.stack(chain_means, dim=1)
        present = torch.stack(chain_present, dim=1)
        if bool((present.sum(dim=1) == 0).any()):
            raise ValueError("Every antibody must contain at least one represented chain")
        weights = present.unsqueeze(-1).to(dtype=values.dtype)
        return (stacked_means * weights).sum(dim=1) / weights.sum(dim=1)

    def _attention_weights(self, projected: Any, mask: Any, chain_ids: Any | None) -> Any:
        if self.attention_score is None:
            raise AssertionError("Attention weights requested from a mean pooler")
        logits = self.attention_score(projected).squeeze(-1)
        if not self.chain_balanced:
            weights = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
            return weights.masked_fill(~mask, 0.0)
        if chain_ids is None:
            raise ValueError("chain_ids are required for chain-balanced pooling")
        self._validate_chain_ids(mask, chain_ids)
        present = torch.stack(
            [(mask & (chain_ids == value)).any(dim=1) for value in (0, 1)], dim=1
        )
        chain_weights = present.to(projected.dtype)
        chain_weights = chain_weights / chain_weights.sum(dim=1, keepdim=True)
        weights = torch.zeros_like(mask, dtype=projected.dtype)
        for chain_id in (0, 1):
            chain_mask = mask & (chain_ids == chain_id)
            local = torch.softmax(logits.masked_fill(~chain_mask, -torch.inf), dim=1)
            local = local.masked_fill(~chain_mask, 0.0)
            weights = weights + local * chain_weights[:, chain_id].unsqueeze(1)
        return weights

    def forward(
        self,
        features: Any,
        mask: Any,
        chain_ids: Any | None = None,
    ) -> Tuple[Any, Any]:
        """Return unit-normalized embeddings and residue averaging weights."""
        _validate_residue_batch(features, mask, self.input_dim)
        safe_features = torch.where(
            mask.unsqueeze(-1), features, torch.zeros_like(features)
        )
        projected = self.residue_projection(safe_features)
        if self.pooling == "attention":
            residue_weights = self._attention_weights(projected, mask, chain_ids)
            pooled = (projected * residue_weights.unsqueeze(-1)).sum(dim=1)
        elif self.chain_balanced:
            if chain_ids is None:
                raise ValueError("chain_ids are required for chain-balanced pooling")
            pooled = self._chain_balanced_mean(projected, mask, chain_ids)
            chain_sizes = torch.stack(
                [(mask & (chain_ids == value)).sum(dim=1) for value in (0, 1)],
                dim=1,
            )
            present = chain_sizes > 0
            chain_weight = present.to(dtype=projected.dtype)
            chain_weight = chain_weight / chain_weight.sum(dim=1, keepdim=True)
            residue_weights = torch.zeros_like(mask, dtype=projected.dtype)
            for chain_id in (0, 1):
                denominator = chain_sizes[:, chain_id].clamp_min(1).to(projected.dtype)
                per_residue = chain_weight[:, chain_id] / denominator
                residue_weights = torch.where(
                    mask & (chain_ids == chain_id),
                    per_residue.unsqueeze(1),
                    residue_weights,
                )
        else:
            pooled = self._masked_mean(projected, mask)
            residue_weights = mask.to(dtype=projected.dtype)
            residue_weights = residue_weights / residue_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        return _l2_normalize(pooled), residue_weights


class BoundedTemperature(_ModuleBase):
    """Learn a positive temperature inside fixed numerical bounds."""

    def __init__(
        self,
        initial: float = 0.2,
        minimum: float = 0.01,
        maximum: float = 1.0,
    ) -> None:
        require_torch()
        super().__init__()
        values = (initial, minimum, maximum)
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in values):
            raise ValueError("Temperature values must be real numbers")
        initial, minimum, maximum = map(float, values)
        if not all(math.isfinite(value) for value in (initial, minimum, maximum)):
            raise ValueError("Temperature values must be finite")
        if minimum <= 0.0 or not minimum < initial < maximum:
            raise ValueError("Temperature values must satisfy 0 < minimum < initial < maximum")
        fraction = (initial - minimum) / (maximum - minimum)
        self.raw_temperature = nn.Parameter(
            torch.tensor(math.log(fraction) - math.log1p(-fraction), dtype=torch.float32)
        )
        self.minimum = minimum
        self.maximum = maximum

    def forward(self) -> Any:
        """Return the bounded differentiable scalar temperature."""
        return self.minimum + (self.maximum - self.minimum) * torch.sigmoid(
            self.raw_temperature
        )


def cosine_similarity_matrix(antibody: Any, antigen: Any) -> Any:
    """Return all pairwise cosine similarities between normalized libraries."""
    require_torch()
    if not torch.is_tensor(antibody) or not torch.is_tensor(antigen):
        raise ValueError("Partner embeddings must be tensors")
    if antibody.ndim != 2 or antigen.ndim != 2:
        raise ValueError("Partner embeddings must be two-dimensional")
    if antibody.device != antigen.device or antibody.dtype != antigen.dtype:
        raise ValueError("Partner embeddings must share a device and dtype")
    if antibody.shape[1] != antigen.shape[1]:
        raise ValueError("Partner embedding dimensions must match")
    return torch.matmul(_l2_normalize(antibody), _l2_normalize(antigen).transpose(0, 1))


class AsymmetricPartnerRetrievalModel(_ModuleBase):
    """Map AntiBERTy antibodies and ESM-2 antigens into one retrieval space."""

    def __init__(
        self,
        antibody_input_dim: int = 512,
        antigen_input_dim: int = 1280,
        embedding_dim: int = 128,
        antibody_hidden_dim: int = 256,
        antigen_hidden_dim: int = 256,
        dropout: float = 0.1,
        chain_balanced_antibody_pooling: bool = True,
        antibody_pooling: str = "mean",
        antigen_pooling: str = "mean",
        initial_temperature: float = 0.2,
        minimum_temperature: float = 0.01,
        maximum_temperature: float = 1.0,
    ) -> None:
        require_torch()
        super().__init__()
        self.antibody_pooler = CALMMeanPooler(
            antibody_input_dim,
            embedding_dim,
            antibody_hidden_dim,
            dropout,
            chain_balanced=chain_balanced_antibody_pooling,
            pooling=antibody_pooling,
        )
        self.antigen_pooler = CALMMeanPooler(
            antigen_input_dim,
            embedding_dim,
            antigen_hidden_dim,
            dropout,
            chain_balanced=False,
            pooling=antigen_pooling,
        )
        self.temperature_head = BoundedTemperature(
            initial_temperature,
            minimum_temperature,
            maximum_temperature,
        )

    def encode_antibody(
        self, features: Any, mask: Any, chain_ids: Any | None
    ) -> Tuple[Any, Any]:
        """Encode a padded antibody batch with optional chain balancing."""
        return self.antibody_pooler(features, mask, chain_ids)

    def encode_antigen(self, features: Any, mask: Any) -> Tuple[Any, Any]:
        """Encode a padded antigen batch with projected masked averaging."""
        return self.antigen_pooler(features, mask)

    def forward(
        self,
        antibody_features: Any,
        antigen_features: Any,
        antibody_mask: Any,
        antigen_mask: Any,
        antibody_chain_ids: Any | None,
    ) -> Dict[str, Any]:
        """Return partner embeddings, similarities, weights, and temperature."""
        antibody_embeddings, antibody_weights = self.encode_antibody(
            antibody_features, antibody_mask, antibody_chain_ids
        )
        antigen_embeddings, antigen_weights = self.encode_antigen(
            antigen_features, antigen_mask
        )
        similarities = cosine_similarity_matrix(antibody_embeddings, antigen_embeddings)
        temperature = self.temperature_head()
        return {
            "antibody_embeddings": antibody_embeddings,
            "antigen_embeddings": antigen_embeddings,
            "antibody_weights": antibody_weights,
            "antigen_weights": antigen_weights,
            "similarity_matrix": similarities,
            "scaled_similarity_matrix": similarities / temperature,
            "temperature": temperature,
        }


def _temperature_tensor(value: Any, similarities: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() != 1 or not torch.is_floating_point(value):
            raise ValueError("temperature must be a floating-point scalar")
        if value.device != similarities.device:
            raise ValueError("temperature and similarities must share a device")
        result = value.reshape(()).to(dtype=similarities.dtype)
    else:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError("temperature must be a real scalar")
        result = similarities.new_tensor(float(value))
    if not bool(torch.isfinite(result)) or bool(result <= 0.0):
        raise ValueError("temperature must be finite and positive")
    return result


def _directional_loss(logits: Any, positives: Any, candidates: Any) -> Any:
    denominator = torch.logsumexp(logits.masked_fill(~candidates, -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    if not bool(torch.isfinite(denominator).all()) or not bool(torch.isfinite(numerator).all()):
        raise ValueError("Every anchor must retain a known positive candidate")
    return (denominator - numerator).mean()


def symmetric_masked_multi_positive_info_nce(
    similarities: Any,
    positive_mask: Any,
    temperature: Any = 0.2,
    denominator_mask: Any | None = None,
) -> Any:
    """Compute symmetric multi-positive InfoNCE with optional uncertain-edge masking."""
    require_torch()
    if not torch.is_tensor(similarities) or similarities.ndim != 2:
        raise ValueError("similarities must be a two-dimensional tensor")
    if not torch.is_floating_point(similarities) or not bool(torch.isfinite(similarities).all()):
        raise ValueError("similarities must contain finite floating-point values")
    if not torch.is_tensor(positive_mask) or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be boolean")
    if tuple(positive_mask.shape) != tuple(similarities.shape):
        raise ValueError("positive_mask must align with similarities")
    if positive_mask.device != similarities.device:
        raise ValueError("positive_mask and similarities must share a device")
    if not bool(positive_mask.any(dim=1).all()) or not bool(positive_mask.any(dim=0).all()):
        raise ValueError("Every partner candidate must participate in a known positive")
    if denominator_mask is None:
        denominator_mask = torch.ones_like(positive_mask)
    if not torch.is_tensor(denominator_mask) or denominator_mask.dtype != torch.bool:
        raise ValueError("denominator_mask must be boolean")
    if tuple(denominator_mask.shape) != tuple(similarities.shape):
        raise ValueError("denominator_mask must align with similarities")
    if denominator_mask.device != similarities.device:
        raise ValueError("denominator_mask and similarities must share a device")
    if bool((positive_mask & ~denominator_mask).any()):
        raise ValueError("Known positives cannot be excluded from the denominator")

    safe_temperature = _temperature_tensor(temperature, similarities)
    calculation_dtype = (
        torch.float32
        if similarities.dtype in (torch.float16, torch.bfloat16)
        else similarities.dtype
    )
    logits = similarities.to(calculation_dtype) / safe_temperature.to(calculation_dtype)
    forward = _directional_loss(logits, positive_mask, denominator_mask)
    reverse = _directional_loss(
        logits.transpose(0, 1),
        positive_mask.transpose(0, 1),
        denominator_mask.transpose(0, 1),
    )
    return 0.5 * (forward + reverse)
