from pathlib import Path

from abcite_core.retrieval import HybridIndex
from agents.graph import EvidenceGraph, should_abstain
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def graph() -> EvidenceGraph:
    chunks = load_corpus(ROOT / "corpus")
    return EvidenceGraph(HybridIndex(chunks), 0.12, True, "local")


def test_cite_or_abstain_supported():
    resp = graph().invoke("Does trastuzumab on HER2 (PDB 1N8Z) share the pertuzumab domain II epitope?")
    assert resp.status.value in {"answered", "hitl_pending"}
    ids = {c.chunk_id for c in resp.citations}
    assert ids, "must cite chunks"
    assert "1N8Z" in resp.answer or any("1N8Z" in c.quote or "1n8z" in c.chunk_id.lower() for c in resp.citations)


def test_cite_or_abstain_empty_hits():
    assert should_abstain("zzzz", [], 0.12) is True
