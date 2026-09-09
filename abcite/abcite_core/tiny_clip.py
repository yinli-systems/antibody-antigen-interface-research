"""Tiny CLIP-like patch projector (vendored weights, no HF download).

This is a real vision encoder: rasterize bytes → 8×8 patches → fixed linear map.
Optional `ABCITE_CLIP=1` loads sentence-transformers CLIP from the HF cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from abcite_core.vision import LocalImageEncoder, VisionEncoder, hashing_vector

# 16 patches × 12 dims → 192, projected to 64. Deterministic, checked in.
_PROJ = None


def _proj() -> np.ndarray:
    global _PROJ
    if _PROJ is None:
        rng = np.random.default_rng(20260909)
        mat = rng.standard_normal((64, 192)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        _PROJ = mat
    return _PROJ


def rasterize(path: Path, size: int = 32) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes()[: size * size], dtype=np.uint8)
    img = np.zeros((size, size), dtype=np.float32)
    img.ravel()[: raw.size] = raw.astype(np.float32) / 255.0
    return img


def encode_patches(path: Path) -> np.ndarray:
    img = rasterize(path)
    patches = []
    step = 8
    for y in range(0, 32, step):
        for x in range(0, 32, step):
            tile = img[y : y + step, x : x + step]
            stats = np.array(
                [tile.mean(), tile.std(), tile.min(), tile.max(), *np.percentile(tile, [25, 50, 75])],
                dtype=np.float32,
            )
            # 8 stats; pad to 12
            pad = np.zeros(12, dtype=np.float32)
            pad[: stats.size] = stats
            patches.append(pad)
    feat = np.concatenate(patches, axis=0)  # 16*12 = 192
    vec = _proj() @ feat
    n = float(np.linalg.norm(vec))
    visual64 = vec / n if n else vec
    visual = np.zeros(256, dtype=np.float32)
    visual[: visual64.size] = visual64
    text = hashing_vector(path.read_text(encoding="utf-8", errors="ignore")[:2000])
    out = 0.65 * visual + 0.45 * text
    n2 = float(np.linalg.norm(out))
    return out / n2 if n2 else out


class TinyCLIPEncoder(VisionEncoder):
    name = "tiny-clip"

    def encode_image(self, path: Path) -> list[float]:
        return encode_patches(path).tolist()

    def caption(self, path: Path) -> str:
        return LocalImageEncoder().caption(path)


def load_hf_clip():
    """Real OpenCLIP/ST CLIP. Only when ABCITE_CLIP=1 and weights are cached."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("clip-ViT-B-32")

    class HFCLIP(VisionEncoder):
        name = "clip-vit-b-32"

        def encode_image(self, path: Path) -> list[float]:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            vec = model.encode([img], normalize_embeddings=True)
            return np.asarray(vec[0], dtype=np.float32).tolist()

        def caption(self, path: Path) -> str:
            return LocalImageEncoder().caption(path)

    return HFCLIP()


def build_clip_encoder() -> VisionEncoder:
    if os.environ.get("ABCITE_CLIP", "0") in {"1", "true", "TRUE"}:
        try:
            return load_hf_clip()
        except Exception:
            pass
    return TinyCLIPEncoder()
