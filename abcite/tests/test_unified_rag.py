from pathlib import Path

from abcite_core.models import AnswerStatus
from abcite_core.retrieval import HybridIndex
from agents.graph import EvidenceGraph
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_unified_task_cites_or_abstains():
    g = EvidenceGraph(HybridIndex(load_corpus(ROOT / "corpus")), 0.12, False, "local")
    resp = g.invoke(
        "Does trastuzumab on HER2 (PDB 1N8Z) share the pertuzumab domain II epitope?",
        task="unified",
    )
    assert resp.status in {AnswerStatus.ANSWERED, AnswerStatus.ABSTAIN, AnswerStatus.HITL_PENDING}
    if resp.status == AnswerStatus.ANSWERED:
        assert resp.citations
        assert all(c.chunk_id for c in resp.citations)
