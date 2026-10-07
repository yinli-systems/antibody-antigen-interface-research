"""Rasterize corpus figures to RGB for CLIP (PIL cannot open SVG natively)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_HEX = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    """Return an RGB PIL image. SVG is drawn; raster formats go through PIL."""
    suffix = path.suffix.lower()
    if suffix == ".svg":
        img = _svg_to_image(path)
    else:
        img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.Resampling.BICUBIC)
    return img.convert("RGB")


def _px(raw: str | None, default: float, axis: int) -> float:
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw.endswith("%"):
        return float(raw[:-1]) / 100.0 * axis
    if raw.endswith("px") or raw.endswith("pt"):
        raw = raw[:-2]
    return float(raw)


def _color(fill: str | None, opacity: str | None, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if not fill or fill == "none":
        return default
    fill = fill.strip()
    a = int(float(opacity) * 255) if opacity else default[3]
    if _HEX.match(fill):
        h = fill[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (r, g, b, a)
    return default


def _svg_to_image(path: Path) -> Image.Image:
    tree = ET.parse(path)
    root = tree.getroot()
    width = int(_px(root.get("width"), 224, 224))
    height = int(_px(root.get("height"), 224, 224))
    width = max(width, 1)
    height = max(height, 1)
    img = Image.new("RGBA", (width, height), (7, 16, 24, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        fill = _color(elem.get("fill"), elem.get("opacity"), (31, 58, 77, 255))
        if tag == "rect":
            x = _px(elem.get("x"), 0, width)
            y = _px(elem.get("y"), 0, height)
            w = _px(elem.get("width"), width, width)
            h = _px(elem.get("height"), height, height)
            if elem.get("width") == "100%":
                w = width
            if elem.get("height") == "100%":
                h = height
            draw.rectangle([x, y, x + w, y + h], fill=fill)
        elif tag == "circle":
            cx = _px(elem.get("cx"), 0, width)
            cy = _px(elem.get("cy"), 0, height)
            r = _px(elem.get("r"), 1, min(width, height))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif tag == "text":
            x = _px(elem.get("x"), 0, width)
            y = _px(elem.get("y"), 16, height)
            text = (elem.text or "").strip()
            if text:
                draw.text((x, y - 12), text, fill=fill, font=font)
    return img.convert("RGB")
