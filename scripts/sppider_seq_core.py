#!/usr/bin/env python3
"""Faithful SPPIDER-seq model and frozen-ESM embedding utilities.

The public SPPIDER-seq repository exposes inference code but not its training
entrypoint.  This module preserves the released v2.1.0 architecture and its
long-sequence chunking behavior so that a separate benchmark adapter can train
the same head on precomputed SAbDab2 embeddings.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
EMBED_DIM = 1280
NUM_HEADS = 16
MAX_TOKENS = 1024
STRIDE = 512

_ModuleBase = nn.Module if nn is not None else object


def require_torch() -> None:
    """Fail with a focused message when a torch-only operation is requested."""
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for this command")


def sequence_sha256(sequence: str) -> str:
    """Return the stable content key used for deduplicated sequence embeddings."""
    return hashlib.sha256(sequence.strip().upper().encode("ascii")).hexdigest()


def embedding_path(root: Path, sequence_key: str) -> Path:
    """Map a sequence digest to a bounded-fanout artifact path."""
    return root / sequence_key[:2] / f"{sequence_key}.pt"


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


class CrossAttentionLayer(_ModuleBase):
    """Released SPPIDER-seq unilateral cross-attention block."""

    def __init__(self, embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS) -> None:
        require_torch()
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: Any,
        context: Any,
        context_mask: Any | None = None,
    ) -> Any:
        """Condition query residues on one context chunk and retain the residual."""
        output, _ = self.cross_attn(
            query,
            context,
            context,
            key_padding_mask=context_mask,
        )
        return self.norm(query + output)


class ChunkwiseInteractionModel(_ModuleBase):
    """Released 8.2M-parameter SPPIDER-seq residue-classification head."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        initial_bias: float | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        self.cross = CrossAttentionLayer(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(embed_dim, 1),
        )
        if initial_bias is not None:
            with torch.no_grad():
                self.mlp[-1].bias.fill_(initial_bias)

    def forward(self, query_chunks: Sequence[Any], context_chunks: Sequence[Any]) -> list[Any]:
        """Return one max-pooled logit vector for every query chunk."""
        if not query_chunks or not context_chunks:
            raise ValueError("Query and context must each contain at least one chunk")
        pooled_per_query: list[Any] = []
        for query in query_chunks:
            if query.ndim == 2:
                query = query.unsqueeze(0)
            context_logits: list[Any] = []
            for context in context_chunks:
                if context.ndim == 2:
                    context = context.unsqueeze(0)
                context_mask = context.abs().sum(dim=-1) == 0
                attended = self.cross(query, context, context_mask=context_mask)
                logits = self.mlp(attended).squeeze(0).squeeze(-1)
                context_logits.append(logits)
            pooled = torch.max(torch.stack(context_logits, dim=0), dim=0).values
            pooled_per_query.append(pooled)
        return pooled_per_query


def merge_chunk_logits_to_full(
    pooled_per_query: Sequence[Any],
    query_starts: Sequence[int],
    full_length: int,
    device: Any,
) -> Any:
    """Merge overlapping query windows by the released elementwise-max rule."""
    require_torch()
    if len(pooled_per_query) != len(query_starts):
        raise ValueError("Query logits and chunk starts have different lengths")
    output = torch.full(
        (full_length,),
        -1e9,
        dtype=torch.float32,
        device=device,
    )
    for logits, start in zip(pooled_per_query, query_starts):
        logits = logits.to(device=device, dtype=torch.float32)
        end = start + int(logits.shape[0])
        if start < 0 or end > full_length:
            raise ValueError(
                f"Chunk interval [{start}, {end}) exceeds sequence length {full_length}"
            )
        placed = torch.full_like(output, -1e9)
        placed[start:end] = logits
        output = torch.maximum(output, placed)
    if torch.any(output <= -1e8):
        raise ValueError("Chunk merge left at least one residue without a prediction")
    return output


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Persist a torch artifact through an adjacent temporary file."""
    require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def safe_torch_load(path: Path, map_location: str | Any = "cpu") -> Any:
    """Load trusted tensor-only artifacts across PyTorch serialization versions."""
    require_torch()
    try:
        try:
            from torch.torch_version import TorchVersion
        except (AttributeError, ImportError):
            TorchVersion = None
        if TorchVersion is not None and hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([TorchVersion]):
                return torch.load(path, map_location=map_location, weights_only=True)
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def validate_embedding_artifact(
    artifact: Mapping[str, Any],
    expected_key: str | None = None,
) -> None:
    """Validate the structural contract of one precomputed embedding artifact."""
    required = {
        "format_version",
        "sequence_sha256",
        "sequence_length",
        "model_name",
        "starts",
        "ends",
        "chunks",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Embedding artifact is missing keys: {sorted(missing)}")
    if artifact["format_version"] != "sppider_seq_embedding_v1":
        raise ValueError(f"Unsupported embedding format: {artifact['format_version']!r}")
    if expected_key is not None and artifact["sequence_sha256"] != expected_key:
        raise ValueError("Embedding sequence digest does not match its requested key")
    starts = [int(value) for value in artifact["starts"]]
    ends = [int(value) for value in artifact["ends"]]
    chunks = list(artifact["chunks"])
    if not chunks or len(starts) != len(ends) or len(starts) != len(chunks):
        raise ValueError("Embedding chunk metadata is empty or inconsistent")
    sequence_length = int(artifact["sequence_length"])
    covered = np.zeros(sequence_length, dtype=np.bool_)
    for start, end, chunk in zip(starts, ends, chunks):
        if start < 0 or end <= start or end > sequence_length:
            raise ValueError(f"Invalid embedding interval [{start}, {end})")
        if tuple(chunk.shape) != (end - start, EMBED_DIM):
            raise ValueError(
                f"Embedding shape {tuple(chunk.shape)} does not match interval "
                f"[{start}, {end}) and dimension {EMBED_DIM}"
            )
        if not torch.isfinite(chunk).all():
            raise ValueError("Embedding artifact contains non-finite values")
        covered[start:end] = True
    if not covered.all():
        raise ValueError("Embedding chunks do not cover the complete sequence")


def embed_sequence(
    sequence: str,
    tokenizer: Any,
    esm_model: Any,
    device: Any,
    max_tokens: int = MAX_TOKENS,
    stride: int = STRIDE,
) -> dict[str, Any]:
    """Generate the exact overlapping ESM-2 chunks used by released inference."""
    require_torch()
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("Cannot embed an empty sequence")
    if max_tokens < 3:
        raise ValueError("max_tokens must accommodate BOS, one residue, and EOS")
    window = max_tokens - 2
    if stride <= 0 or stride > window:
        raise ValueError("stride must be positive and no larger than the residue window")

    starts: list[int] = []
    ends: list[int] = []
    chunks: list[Any] = []
    for start in range(0, len(sequence), stride):
        end = min(start + window, len(sequence))
        encoded = tokenizer(
            sequence[start:end],
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(device)
        with torch.inference_mode():
            output = esm_model(input_ids)
            embedding = (
                output.last_hidden_state[:, 1:-1, :]
                .squeeze(0)
                .to(dtype=torch.float32, device="cpu")
                .contiguous()
            )
        if tuple(embedding.shape) != (end - start, EMBED_DIM):
            raise ValueError(
                f"Unexpected ESM embedding shape {tuple(embedding.shape)} for "
                f"interval [{start}, {end})"
            )
        starts.append(start)
        ends.append(end)
        chunks.append(embedding)
        if end == len(sequence):
            break

    artifact = {
        "format_version": "sppider_seq_embedding_v1",
        "sequence_sha256": sequence_sha256(sequence),
        "sequence_length": len(sequence),
        "model_name": ESM_MODEL_NAME,
        "max_tokens": int(max_tokens),
        "stride": int(stride),
        "starts": starts,
        "ends": ends,
        "chunks": chunks,
    }
    validate_embedding_artifact(artifact, expected_key=artifact["sequence_sha256"])
    return artifact


def load_embedding_for_device(path: Path, device: Any) -> tuple[list[Any], list[int], int]:
    """Load and validate one embedding artifact and transfer its chunks to a device."""
    artifact = safe_torch_load(path, map_location="cpu")
    validate_embedding_artifact(artifact, expected_key=path.stem)
    chunks = [chunk.to(device=device, dtype=torch.float32) for chunk in artifact["chunks"]]
    starts = [int(value) for value in artifact["starts"]]
    return chunks, starts, int(artifact["sequence_length"])


def extract_model_state(payload: Any) -> Mapping[str, Any]:
    """Accept either an official bare state dict or a resumable training checkpoint."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(payload)!r}")
    if "model_state_dict" in payload:
        state = payload["model_state_dict"]
    else:
        state = payload
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint does not contain a non-empty model state dict")
    return state


def predict_residue_scores(
    model: Any,
    query_chunks: Sequence[Any],
    query_starts: Sequence[int],
    query_length: int,
    context_chunks: Sequence[Any],
    device: Any,
) -> np.ndarray:
    """Run one query-context pair and return finite per-residue probabilities."""
    require_torch()
    model.eval()
    with torch.inference_mode():
        pooled = model(query_chunks, context_chunks)
        logits = merge_chunk_logits_to_full(
            pooled,
            query_starts,
            query_length,
            device,
        )
        scores = torch.sigmoid(logits).to(device="cpu", dtype=torch.float32).numpy()
    if scores.shape != (query_length,):
        raise ValueError(f"Unexpected residue score shape: {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("Model emitted non-finite residue scores")
    return scores
