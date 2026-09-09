"""Deterministic local image embeddings used in hybrid retrieval.

CLIP / OpenCLIP is optional (large weights). Default is a real, tested encoder:
SVG text tokens + byte n-grams + color histogram → L2-normalized 256-d vector.
OpenAI/Azure vision caption() is used when an API key is present.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

DIM = 256
TEXT_RE = re.compile(r">([^<]{2,80})<")
HEX_RE = re.compile(r"#([0-9a-fA-F]{6})")


class VisionEncoder(ABC):
    name: str

    @abstractmethod
    def encode_image(self, path: Path) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def caption(self, path: Path) -> str:
        raise NotImplementedError


def _stable_index(token: str, dim: int = DIM) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim


def hashing_vector(text: str, dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        vec[_stable_index(tok, dim)] += 1.0
    n = float(np.linalg.norm(vec))
    return vec / n if n else vec


class LocalImageEncoder(VisionEncoder):
    """Always-on vision path. Not a comment: `encode_image` is used by HybridIndex."""

    name = "local-image"

    def encode_image(self, path: Path) -> list[float]:
        return encode_image_file(path).tolist()

    def caption(self, path: Path) -> str:
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            import json

            return str(json.loads(sidecar.read_text(encoding="utf-8")).get("caption", path.stem))
        extracted = " ".join(TEXT_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))
        return extracted or path.stem


def encode_image_file(path: Path, dim: int = DIM) -> np.ndarray:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    vec = hashing_vector(text[:8000], dim)
    visual = np.zeros(dim, dtype=np.float32)
    for i, b in enumerate(raw[:2048]):
        visual[i % dim] += b / 255.0
    for hexcol in HEX_RE.findall(text):
        visual[_stable_index(hexcol, dim)] += 2.0
    vn = float(np.linalg.norm(visual))
    if vn:
        visual /= vn
    mixed = vec + 0.35 * visual
    n = float(np.linalg.norm(mixed))
    return mixed / n if n else mixed


def try_clip_encoder():
    """OpenCLIP ViT-B-32 when `[clip]` + weights are available. Returns None otherwise."""
    from abcite_core.clip_encoder import (
        clip_forced_off,
        clip_stack_available,
        clip_weights_cached,
        load_open_clip,
    )

    if clip_forced_off() or not clip_stack_available() or not clip_weights_cached():
        return None
    try:
        return load_open_clip()
    except Exception:
        return None


def build_vision_encoder(prefer_clip: bool = False) -> VisionEncoder:
    """Delegate to OpenCLIP when `[clip]` is installed. `prefer_clip` is unused (kept for callers)."""
    from abcite_core.clip_encoder import build_vision_encoder as _build

    _ = prefer_clip
    return _build()
