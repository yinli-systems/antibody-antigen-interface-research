from pathlib import Path

from abcite_core.retrieval import HybridIndex
from abcite_core.tiny_clip import TinyCLIPEncoder, encode_patches
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_clip_encoder_dimension():
    path = ROOT / "corpus" / "figures" / "fig-1n8z.svg"
    vec = encode_patches(path)
    assert vec.shape == (256,)
    enc = TinyCLIPEncoder()
    assert len(enc.encode_image(path)) == 256
    assert enc.name == "tiny-clip"


def test_tiny_clip_image_nn_fig_1n8z():
    index = HybridIndex(load_corpus(ROOT / "corpus"), vision=TinyCLIPEncoder())
    hits = index.search_by_image(ROOT / "corpus" / "figures" / "fig-1n8z.svg", top_k=3)
    assert hits[0].chunk.chunk_id == "fig-1n8z"
