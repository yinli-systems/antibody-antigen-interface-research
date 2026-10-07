from pathlib import Path

from abcite_core.retrieval import HybridIndex
from agents.graph import EvidenceGraph
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_multimodal_figure_id_appears():
    g = EvidenceGraph(HybridIndex(load_corpus(ROOT / "corpus")), 0.12, True, "local")
    resp = g.invoke(
        "Using the trastuzumab epitope diagram, is the Fab contacting HER2 domain IV rather than domain II?"
    )
    ids = {c.chunk_id for c in resp.citations}
    assert "fig-1n8z" in ids
    fig = next(c for c in resp.citations if c.chunk_id == "fig-1n8z")
    assert fig.thumbnail_url
    assert fig.modality.value == "figure"
