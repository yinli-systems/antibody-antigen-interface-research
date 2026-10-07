from pathlib import Path

from abcite_core.retrieval import HybridIndex
from agents.graph import EvidenceGraph, compose_with_provider
from ingest.chunking import load_corpus
from providers.openai import OpenAILLM, OpenAIVision

ROOT = Path(__file__).resolve().parents[1]


class _Resp:
    status_code = 200

    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def test_openai_phrasing_keeps_retrieved_citations():
    calls = {"n": 0}

    def post(url, **kwargs):
        calls["n"] += 1
        return _Resp(
            "Trastuzumab contacts HER2 domain IV in PDB 1N8Z, distinct from pertuzumab domain II."
        )

    llm = OpenAILLM("sk-test", post=post)
    index = HybridIndex(load_corpus(ROOT / "corpus"))
    graph = EvidenceGraph(index, 0.12, True, llm.name, llm=llm)
    resp = graph.invoke("Does trastuzumab on HER2 (PDB 1N8Z) share the pertuzumab domain II epitope?")
    assert calls["n"] == 1
    assert resp.citations, "citations must come from retrieval, not the LLM"
    assert resp.provider == "openai"
    assert "1N8Z" in resp.answer or "domain" in resp.answer.lower()


def test_openai_vision_caption_called_with_key():
    calls = {"n": 0}

    def post(url, **kwargs):
        calls["n"] += 1
        body = kwargs.get("json") or {}
        assert "image_url" in str(body)
        return _Resp("Local schematic of trastuzumab on HER2 domain IV.")

    vis = OpenAIVision("sk-test", post=post)
    cap = vis.caption(ROOT / "corpus" / "figures" / "fig-1n8z.svg")
    assert calls["n"] == 1
    assert "trastuzumab" in cap.lower()


def test_compose_rejects_uncited_hallucination():
    llm = OpenAILLM("sk-test", post=lambda *a, **k: _Resp("The moon is made of cheese PDB 9ZZZ."))
    chunks = load_corpus(ROOT / "corpus")[:2]
    text = compose_with_provider(llm, "overlap?", chunks, [])
    assert "9ZZZ" not in text or "Grounded summary" in text
