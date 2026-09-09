"""Figure ingest: sidecar captions + optional vision captioner."""

from __future__ import annotations

from pathlib import Path

from providers.base import VisionEncoder


def caption_figure(path: Path, encoder: VisionEncoder) -> str:
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        import json

        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return str(data.get("caption", path.stem))
    return encoder.caption(path)
