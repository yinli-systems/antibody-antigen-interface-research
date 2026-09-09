from abcite_core.models import AnswerStatus as AS
from agents.graph import needs_hitl


def test_hitl_interrupt_path_unit():
    assert needs_hitl("clinical therapeutic decision for a patient", False, True) is True
    assert needs_hitl("what is PDB 1N8Z?", False, True) is False
    assert needs_hitl("what is PDB 1N8Z?", True, True) is True
    assert needs_hitl("clinical", False, False) is False


def test_hitl_graph_status(monkeypatch):
    from pathlib import Path

    from abcite_core.retrieval import HybridIndex
    from agents.graph import EvidenceGraph
    from ingest.chunking import load_corpus

    root = Path(__file__).resolve().parents[1]
    g = EvidenceGraph(HybridIndex(load_corpus(root / "corpus")), 0.12, True, "local")
    resp = g.invoke(
        "Clinical therapeutic decision: should a patient switch from trastuzumab to pertuzumab because epitopes overlap 100%?",
        high_stakes=True,
    )
    assert resp.status == AS.HITL_PENDING
    assert resp.hitl_reason
