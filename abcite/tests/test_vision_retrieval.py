from pathlib import Path

from abcite_core.retrieval import HybridIndex
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_figure_only_query_retrieves_fig_1n8z():
    index = HybridIndex(load_corpus(ROOT / "corpus"))
    hits = index.search(
        "schematic diagram of trastuzumab Fab contacting HER2 domain IV membrane-proximal epitope",
        top_k=5,
    )
    ids = [h.chunk.chunk_id for h in hits]
    assert "fig-1n8z" in ids


def test_search_by_image_returns_same_figure():
    index = HybridIndex(load_corpus(ROOT / "corpus"))
    path = ROOT / "corpus" / "figures" / "fig-1n8z.svg"
    hits = index.search_by_image(path, top_k=3)
    assert hits
    assert hits[0].chunk.chunk_id == "fig-1n8z"
    assert hits[0].image > 0.5
