"""Partner-retrieval heads and multi-positive contrastive learning utilities."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    """Fail with a focused message when retrieval code is used without PyTorch."""
    if torch is None or nn is None or F is None:
        raise RuntimeError("PyTorch is required for partner-retrieval operations")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_residue_batch(features: Any, mask: Any, name: str) -> None:
    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError(f"{name} features must have shape [batch, residues, channels]")
    if not torch.is_floating_point(features):
        raise ValueError(f"{name} features must use a floating-point dtype")
    if features.shape[0] == 0 or features.shape[1] == 0 or features.shape[2] == 0:
        raise ValueError(f"{name} features cannot contain an empty dimension")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError(f"{name} mask must be a two-dimensional boolean tensor")
    if tuple(mask.shape) != tuple(features.shape[:2]):
        raise ValueError(f"{name} mask must align with the batch and residue axes")
    if mask.device != features.device:
        raise ValueError(f"{name} features and mask must use the same device")
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError(f"Every {name} item must contain at least one valid residue")
    valid_features = features.masked_select(mask.unsqueeze(-1)).reshape(-1, features.shape[-1])
    if not torch.isfinite(valid_features).all():
        raise ValueError(f"Valid {name} features must be finite")


def _l2_normalize(embeddings: Any, name: str) -> Any:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"{name} embeddings must have shape [batch, channels]")
    if not torch.is_floating_point(embeddings) or not torch.isfinite(embeddings).all():
        raise ValueError(f"{name} embeddings must be finite floating-point values")
    norms = torch.linalg.vector_norm(embeddings, ord=2, dim=-1, keepdim=True)
    epsilon = max(1e-12, float(torch.finfo(embeddings.dtype).tiny))
    if torch.any(norms <= epsilon):
        raise ValueError(f"{name} embeddings cannot contain a zero-norm vector")
    return embeddings / norms


class MaskedAttentionPooler(_ModuleBase):
    """Pool variable-length residue features and project them onto a unit sphere."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        attention_hidden_dim: int = 128,
        projection_hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        require_torch()
        super().__init__()
        input_dim = _positive_integer(input_dim, "input_dim")
        embedding_dim = _positive_integer(embedding_dim, "embedding_dim")
        attention_hidden_dim = _positive_integer(
            attention_hidden_dim, "attention_hidden_dim"
        )
        projection_hidden_dim = _positive_integer(
            projection_hidden_dim, "projection_hidden_dim"
        )
        if not isinstance(dropout, Real) or isinstance(dropout, bool):
            raise ValueError("dropout must be a real number")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

        self.input_dim = input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1, bias=False),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(projection_hidden_dim, embedding_dim),
        )

    def forward(self, features: Any, mask: Any) -> Tuple[Any, Any]:
        """Return normalized complex embeddings and residue attention weights."""
        _validate_residue_batch(features, mask, "residue")
        if int(features.shape[-1]) != self.input_dim:
            raise ValueError(
                f"Expected residue feature dimension {self.input_dim}, "
                f"received {int(features.shape[-1])}"
            )

        safe_features = torch.where(
            mask.unsqueeze(-1), features, torch.zeros_like(features)
        )
        normalized_features = self.input_norm(safe_features)
        attention_logits = self.attention(normalized_features).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, -torch.inf)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = attention_weights.masked_fill(~mask, 0.0)
        attention_weights = attention_weights / attention_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(torch.finfo(attention_weights.dtype).tiny)
        pooled = torch.sum(
            attention_weights.unsqueeze(-1) * safe_features,
            dim=1,
        )
        embeddings = _l2_normalize(self.projection(pooled), "projected")
        return embeddings, attention_weights


class BoundedTemperature(_ModuleBase):
    """Learn a positive temperature constrained to a numerically safe interval."""

    def __init__(
        self,
        initial_temperature: float = 0.07,
        minimum_temperature: float = 0.01,
        maximum_temperature: float = 1.0,
    ) -> None:
        require_torch()
        super().__init__()
        values = (initial_temperature, minimum_temperature, maximum_temperature)
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in values):
            raise ValueError("Temperature bounds must be real numbers")
        initial = float(initial_temperature)
        minimum = float(minimum_temperature)
        maximum = float(maximum_temperature)
        if not all(math.isfinite(value) for value in (initial, minimum, maximum)):
            raise ValueError("Temperature bounds must be finite")
        if minimum <= 0.0 or not minimum < initial < maximum:
            raise ValueError(
                "Temperature bounds must satisfy 0 < minimum < initial < maximum"
            )

        fraction = (initial - minimum) / (maximum - minimum)
        raw_initial = math.log(fraction) - math.log1p(-fraction)
        self.raw_temperature = nn.Parameter(torch.tensor(raw_initial, dtype=torch.float32))
        self.minimum_temperature = minimum
        self.maximum_temperature = maximum

    def forward(self) -> Any:
        """Return the bounded differentiable scalar temperature."""
        interval = self.maximum_temperature - self.minimum_temperature
        return self.minimum_temperature + interval * torch.sigmoid(self.raw_temperature)

    def extra_repr(self) -> str:
        return (
            f"minimum_temperature={self.minimum_temperature}, "
            f"maximum_temperature={self.maximum_temperature}"
        )


def cosine_similarity_matrix(antibody_embeddings: Any, antigen_embeddings: Any) -> Any:
    """Return all antibody-by-antigen cosine similarities."""
    require_torch()
    if not torch.is_tensor(antibody_embeddings) or not torch.is_tensor(
        antigen_embeddings
    ):
        raise ValueError("Antibody and antigen embeddings must be tensors")
    if antibody_embeddings.device != antigen_embeddings.device:
        raise ValueError("Antibody and antigen embeddings must use the same device")
    if antibody_embeddings.dtype != antigen_embeddings.dtype:
        raise ValueError("Antibody and antigen embeddings must use the same dtype")
    if antibody_embeddings.ndim != 2 or antigen_embeddings.ndim != 2:
        raise ValueError("Antibody and antigen embeddings must be two-dimensional")
    if antibody_embeddings.shape[1] != antigen_embeddings.shape[1]:
        raise ValueError("Antibody and antigen embedding dimensions must match")
    antibody = _l2_normalize(antibody_embeddings, "antibody")
    antigen = _l2_normalize(antigen_embeddings, "antigen")
    return torch.matmul(antibody, antigen.transpose(0, 1))


class PartnerRetrievalModel(_ModuleBase):
    """Encode both partners independently for symmetric contrastive retrieval."""

    def __init__(
        self,
        antigen_input_dim: int,
        antibody_input_dim: int,
        embedding_dim: int = 128,
        attention_hidden_dim: int = 128,
        projection_hidden_dim: int = 256,
        dropout: float = 0.0,
        initial_temperature: float = 0.07,
        minimum_temperature: float = 0.01,
        maximum_temperature: float = 1.0,
    ) -> None:
        require_torch()
        super().__init__()
        self.antigen_pooler = MaskedAttentionPooler(
            antigen_input_dim,
            embedding_dim,
            attention_hidden_dim,
            projection_hidden_dim,
            dropout,
        )
        self.antibody_pooler = MaskedAttentionPooler(
            antibody_input_dim,
            embedding_dim,
            attention_hidden_dim,
            projection_hidden_dim,
            dropout,
        )
        self.temperature_head = BoundedTemperature(
            initial_temperature,
            minimum_temperature,
            maximum_temperature,
        )

    def encode_antigen(self, features: Any, mask: Any) -> Tuple[Any, Any]:
        """Encode an antigen residue batch into normalized complex embeddings."""
        return self.antigen_pooler(features, mask)

    def encode_antibody(self, features: Any, mask: Any) -> Tuple[Any, Any]:
        """Encode an antibody residue batch into normalized complex embeddings."""
        return self.antibody_pooler(features, mask)

    def forward(
        self,
        antigen_features: Any,
        antibody_features: Any,
        antigen_mask: Any,
        antibody_mask: Any,
    ) -> Dict[str, Any]:
        """Return normalized embeddings, cosine logits, and attention weights."""
        antigen_embeddings, antigen_attention = self.encode_antigen(
            antigen_features, antigen_mask
        )
        antibody_embeddings, antibody_attention = self.encode_antibody(
            antibody_features, antibody_mask
        )
        similarities = cosine_similarity_matrix(
            antibody_embeddings, antigen_embeddings
        )
        temperature = self.temperature_head()
        return {
            "antigen_embeddings": antigen_embeddings,
            "antibody_embeddings": antibody_embeddings,
            "antigen_attention": antigen_attention,
            "antibody_attention": antibody_attention,
            "similarity_matrix": similarities,
            "scaled_similarity_matrix": similarities / temperature,
            "temperature": temperature,
        }


def _temperature_tensor(temperature: Any, similarities: Any) -> Any:
    if torch.is_tensor(temperature):
        if temperature.numel() != 1 or not torch.is_floating_point(temperature):
            raise ValueError("temperature must be a floating-point scalar")
        if temperature.device != similarities.device:
            raise ValueError("temperature and similarities must use the same device")
        result = temperature.reshape(()).to(dtype=similarities.dtype)
    else:
        if not isinstance(temperature, Real) or isinstance(temperature, bool):
            raise ValueError("temperature must be a real scalar")
        result = similarities.new_tensor(float(temperature))
    if not torch.isfinite(result) or result <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return result


def _directional_multi_positive_loss(logits: Any, positive_mask: Any) -> Any:
    valid_anchors = positive_mask.any(dim=1)
    valid_logits = logits[valid_anchors]
    valid_positives = positive_mask[valid_anchors]
    denominator = torch.logsumexp(valid_logits, dim=1)
    numerator = torch.logsumexp(
        valid_logits.masked_fill(~valid_positives, -torch.inf),
        dim=1,
    )
    return (denominator - numerator).mean()


def symmetric_multi_positive_info_nce(
    similarities: Any,
    positive_mask: Any,
    temperature: Any = 0.07,
) -> Any:
    """Compute bidirectional InfoNCE while allowing multiple known positives."""
    require_torch()
    if not torch.is_tensor(similarities) or similarities.ndim != 2:
        raise ValueError("similarities must be a two-dimensional tensor")
    if not torch.is_floating_point(similarities):
        raise ValueError("similarities must use a floating-point dtype")
    if similarities.shape[0] == 0 or similarities.shape[1] == 0:
        raise ValueError("similarities cannot contain an empty dimension")
    if not torch.isfinite(similarities).all():
        raise ValueError("similarities must be finite")
    if not torch.is_tensor(positive_mask) or positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be a boolean tensor")
    if tuple(positive_mask.shape) != tuple(similarities.shape):
        raise ValueError("positive_mask must have the same shape as similarities")
    if positive_mask.device != similarities.device:
        raise ValueError("positive_mask and similarities must use the same device")
    if not torch.any(positive_mask):
        raise ValueError("positive_mask must contain at least one known positive")

    safe_temperature = _temperature_tensor(temperature, similarities)
    calculation_dtype = (
        torch.float32
        if similarities.dtype in (torch.float16, torch.bfloat16)
        else similarities.dtype
    )
    logits = similarities.to(dtype=calculation_dtype) / safe_temperature.to(
        dtype=calculation_dtype
    )
    if not torch.isfinite(logits).all():
        raise ValueError("Scaled similarities must be finite")

    antibody_to_antigen = _directional_multi_positive_loss(logits, positive_mask)
    antigen_to_antibody = _directional_multi_positive_loss(
        logits.transpose(0, 1), positive_mask.transpose(0, 1)
    )
    return 0.5 * (antibody_to_antigen + antigen_to_antibody)
