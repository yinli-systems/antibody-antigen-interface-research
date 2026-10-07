"""Download OpenCLIP ViT-B-32 (openai) weights into ~/.cache/clip."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    print("Downloading OpenCLIP ViT-B-32 (openai) via CDN → ~/.cache/clip …", flush=True)
    from abcite_core.clip_encoder import (
        CLIP_NAME,
        download_openclip_weights,
        load_open_clip,
    )
    from abcite_core.raster import load_rgb

    path = download_openclip_weights()
    print(f"checkpoint={path} bytes={path.stat().st_size}", flush=True)
    enc = load_open_clip()
    sample = Path(__file__).resolve().parents[1] / "corpus" / "figures" / "fig-1n8z.svg"
    vec: list[float] = []
    if sample.exists():
        rgb = load_rgb(sample)
        print(f"rasterized {sample.name} -> {rgb.size} {rgb.mode}", flush=True)
        vec = enc.encode_image(sample)
    print(f"CLIP encoder={enc.name or CLIP_NAME} dim={len(vec) or enc.dim}", flush=True)
    print("Weights cached. Retrieval auto-uses CLIP after `pip install -e '.[clip]'.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise
