import json
from pathlib import Path

from abcite_core.retrieval import HybridIndex
from agents.graph import EvidenceGraph
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_all_unanswerable_gold_abstain():
    gold = []
    for line in (ROOT / "evals/gold/gold.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["expected_status"] == "abstain":
            gold.append(row)
    assert len(gold) >= 8
    g = EvidenceGraph(HybridIndex(load_corpus(ROOT / "corpus")), 0.12, True, "local")
    for item in gold:
        resp = g.invoke(item["question"])
        assert resp.status.value == "abstain", item["id"]
        assert resp.citations == []
