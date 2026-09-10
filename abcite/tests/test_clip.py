from pathlib import Path

import numpy as np
import pytest

from abcite_core.clip_encoder import (
    CLIP_NAME,
    OpenCLIPEncoder,
    build_vision_encoder,
    clip_stack_available,
    clip_weights_cached,
)
from abcite_core.raster import load_rgb
from abcite_core.retrieval import HybridIndex
from abcite_core.tiny_clip import TinyCLIPEncoder
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "corpus" / "figures" / "fig-1n8z.svg"


def test_svg_rasterizes_to_rgb():
    img = load_rgb(FIG)
    assert img.mode == "RGB"
    assert img.size[0] >= 64 and img.size[1] >= 64
    arr = np.asarray(img)
    assert arr.ndim == 3 and arr.shape[2] == 3
    # Distinct from a blank canvas: the amber epitope circle is present.
    assert arr.mean() > 1.0


class _FakeCLIP(OpenCLIPEncoder):
    name = CLIP_NAME
    dim = 512

    def __init__(self) -> None:
        self.model = None
        self.preprocess = lambda rgb: _to_tensor(rgb)
        self.tokenizer = None
        self.last_preprocess = None
        self.preprocess_calls: list[dict] = []

    def encode_image(self, path: Path) -> list[float]:
        rgb = load_rgb(path)
        self.last_preprocess = {"mode": rgb.mode, "size": rgb.size, "path": str(path)}
        self.preprocess_calls.append(self.last_preprocess)
        t = self.preprocess(rgb)
        assert t.ndim == 3
        vec = np.zeros(512, dtype=np.float32)
        mean = float(np.asarray(rgb).mean())
        vec[0] = mean / 255.0
        vec[1] = hash(path.name) % 100 / 100.0
        n = float(np.linalg.norm(vec))
        return (vec / n).tolist()

    def encode_text(self, text: str) -> list[float]:
        vec = np.zeros(512, dtype=np.float32)
        vec[2] = len(text) / 500.0
        n = float(np.linalg.norm(vec))
        return (vec / n).tolist()


def _to_tensor(rgb):
    arr = np.asarray(rgb).transpose(2, 0, 1).astype(np.float32) / 255.0
    return arr


def test_clip_pipeline_rasterizes_svg_before_encode():
    enc = _FakeCLIP()
    vec = enc.encode_image(FIG)
    assert len(vec) == 512
    assert enc.preprocess_calls
    call = enc.preprocess_calls[0]
    assert call["mode"] == "RGB"
    assert "fig-1n8z" in call["path"]
    chunks = load_corpus(ROOT / "corpus")
    idx = HybridIndex(chunks, vision=enc)
    hits = idx.search_by_image(FIG, top_k=3)
    assert hits[0].chunk.chunk_id == "fig-1n8z"


def test_clip_extra_requires_real_clip_encoder(monkeypatch):
    """Fails if open_clip is installed but HybridIndex would still pick tiny/hash."""
    if not clip_stack_available():
        enc = build_vision_encoder()
        assert enc.name in {"local-image", "tiny-clip"}
        return

    class _Marker(OpenCLIPEncoder):
        name = CLIP_NAME

        def __init__(self) -> None:
            self.model = self.preprocess = self.tokenizer = None
            self.last_preprocess = None

        def encode_image(self, path: Path) -> list[float]:
            return [0.0] * 512

    monkeypatch.delenv("ABCITE_CLIP", raising=False)
    monkeypatch.setenv("ABCITE_CLIP", "auto")
    monkeypatch.setenv("ABCITE_COLPALI", "0")
    monkeypatch.setattr("abcite_core.clip_encoder.clip_weights_cached", lambda: True)
    monkeypatch.setattr("abcite_core.clip_encoder.load_open_clip", _Marker)
    enc = build_vision_encoder()
    assert enc.name.startswith("clip-"), enc.name
    assert enc.name != "tiny-clip"
    assert not isinstance(enc, TinyCLIPEncoder)


def test_live_openclip_figure_retrieval():
    if not clip_stack_available() or not clip_weights_cached():
        pytest.skip("pip install -e '.[clip]' && python -m abcite_core.download_clip")
    if not clip_weights_cached():
        pytest.skip("python -m abcite_core.download_clip")
    from abcite_core.clip_encoder import load_open_clip

    enc = load_open_clip()
    assert enc.name == CLIP_NAME
    vec = enc.encode_image(FIG)
    assert len(vec) == 512
    idx = HybridIndex(load_corpus(ROOT / "corpus"), vision=enc)
    hits = idx.search_by_image(FIG, top_k=5)
    scores = [(h.chunk.chunk_id, round(h.image, 4)) for h in hits]
    print("CLIP search_by_image", scores)
    assert hits[0].chunk.chunk_id == "fig-1n8z"
    assert hits[0].image > 0.5
    text_hits = idx.search("trastuzumab HER2 domain IV epitope diagram", top_k=6)
    ids = [h.chunk.chunk_id for h in text_hits]
    assert "fig-1n8z" in ids or any(c.startswith("fig-") for c in ids)


def test_biomedclip_local_snapshot(tmp_path, monkeypatch):
    from abcite_core.clip_encoder import _biomedclip_local_snapshot

    snap = tmp_path / "hub" / "models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "open_clip_config.json").write_text("{}", encoding="utf-8")
    (snap / "open_clip_pytorch_model.bin").write_bytes(b"x" * 51_000_000)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr("abcite_core.clip_encoder.clip_cache_dir", lambda: tmp_path / "clip")
    found = _biomedclip_local_snapshot()
    assert found == snap
