"""Product vision flags. CLIP fusion is off unless explicitly enabled."""

from __future__ import annotations

import os

DEFAULT_RERANK_K = 50


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def vision_fusion_enabled() -> bool:
    """RRF CLIP into the full ranking. Default off — it hurt caption BM25 on PMC OA."""
    return _flag("ABCITE_VISION_FUSION", "0")


def vision_rerank_enabled() -> bool:
    """Rerank BM25/text top-K with CLIP T→I. Default off; opt-in via ABCITE_VISION_RERANK=1."""
    return _flag("ABCITE_VISION_RERANK", "0")


def vision_rerank_k(default: int = DEFAULT_RERANK_K) -> int:
    raw = os.environ.get("ABCITE_VISION_RERANK_K", "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default
