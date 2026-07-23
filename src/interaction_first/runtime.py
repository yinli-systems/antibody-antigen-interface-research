"""Self-contained runtime helpers for frozen embeddings and checkpoints."""

from __future__ import annotations

import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EMBEDDING_DIM = 1280
EMBEDDING_FORMAT = "sppider_seq_embedding_v1"


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for this operation") from error
    return torch


def embedding_path(root: Path, sequence_key: str) -> Path:
    """Map a sequence digest to the frozen embedding artifact path."""
    return root / sequence_key[:2] / f"{sequence_key}.pt"


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic CUDA."""
    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Persist a PyTorch artifact through an adjacent temporary file."""
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def safe_torch_load(path: Path, map_location: str | Any = "cpu") -> Any:
    """Load trusted tensor-only artifacts across PyTorch versions."""
    torch = _require_torch()
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
    """Validate one frozen ESM embedding before it enters the model."""
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
    if artifact["format_version"] != EMBEDDING_FORMAT:
        raise ValueError(f"Unsupported embedding format: {artifact['format_version']!r}")
    if expected_key is not None and artifact["sequence_sha256"] != expected_key:
        raise ValueError("Embedding sequence digest does not match its requested key")
    starts = [int(value) for value in artifact["starts"]]
    ends = [int(value) for value in artifact["ends"]]
    chunks = list(artifact["chunks"])
    if not chunks or len(starts) != len(ends) or len(starts) != len(chunks):
        raise ValueError("Embedding chunk metadata is empty or inconsistent")
    sequence_length = int(artifact["sequence_length"])
    if sequence_length <= 0:
        raise ValueError("Embedding sequence length must be positive")
    covered = np.zeros(sequence_length, dtype=np.bool_)
    for start, end, chunk in zip(starts, ends, chunks):
        if start < 0 or end <= start or end > sequence_length:
            raise ValueError(f"Invalid embedding interval [{start}, {end})")
        if tuple(chunk.shape) != (end - start, EMBEDDING_DIM):
            raise ValueError("Embedding chunk shape does not match its interval")
        if not bool(chunk.is_floating_point()) or not bool(chunk.isfinite().all()):
            raise ValueError("Embedding chunk contains invalid tensor values")
        covered[start:end] = True
    if not covered.all():
        raise ValueError("Embedding chunks do not cover the complete sequence")


class EmbeddingStore:
    """Bounded CPU LRU cache for validated frozen embedding artifacts."""

    def __init__(self, root: Path, max_cache_gib: float) -> None:
        if max_cache_gib < 0:
            raise ValueError("max_cache_gib cannot be negative")
        self.root = root
        self.max_cache_bytes = int(max_cache_gib * 1024**3)
        self.cache: OrderedDict[str, tuple[Mapping[str, Any], int]] = OrderedDict()
        self.cache_bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def artifact_bytes(artifact: Mapping[str, Any]) -> int:
        """Return the exact tensor payload retained by an artifact."""
        return sum(int(chunk.numel() * chunk.element_size()) for chunk in artifact["chunks"])

    def load(self, sequence_key: str, device: Any) -> tuple[list[Any], list[int], int]:
        """Load, validate, cache, and transfer an embedding to a device."""
        torch = _require_torch()
        if sequence_key in self.cache:
            artifact, artifact_size = self.cache.pop(sequence_key)
            self.cache[sequence_key] = (artifact, artifact_size)
            self.hits += 1
        else:
            artifact = safe_torch_load(
                embedding_path(self.root, sequence_key),
                map_location="cpu",
            )
            validate_embedding_artifact(artifact, expected_key=sequence_key)
            artifact_size = self.artifact_bytes(artifact)
            self.misses += 1
            if artifact_size <= self.max_cache_bytes:
                while self.cache and self.cache_bytes + artifact_size > self.max_cache_bytes:
                    _, (_, evicted_size) = self.cache.popitem(last=False)
                    self.cache_bytes -= evicted_size
                self.cache[sequence_key] = (artifact, artifact_size)
                self.cache_bytes += artifact_size
        chunks = [chunk.to(device=device, dtype=torch.float32) for chunk in artifact["chunks"]]
        starts = [int(value) for value in artifact["starts"]]
        return chunks, starts, int(artifact["sequence_length"])

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe cache utilization statistics."""
        return {
            "max_cache_bytes": self.max_cache_bytes,
            "cache_bytes": self.cache_bytes,
            "cached_sequence_count": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
        }
