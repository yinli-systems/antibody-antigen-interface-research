"""CI eval gate + Hit@5 / faithfulness / judge / multimodal citation-id match."""

from __future__ import annotations

import json
from pathlib import Path

from abcite_core.judge import HeuristicJudge
from abcite_core.metrics import citation_id_match, faithfulness, hit_at_k, recall_at_k
from abcite_core.retrieval import HybridIndex
from agents.graph import EvidenceGraph
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def load_gold() -> list[dict]:
    gold_dir = ROOT / "evals" / "gold"
    rows: list[dict] = []
    for path in sorted(gold_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _table(rows: list[tuple[str, str]]) -> str:
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k.ljust(width)}  {v}" for k, v in rows)


def run_gate() -> dict:
    chunks = load_corpus(ROOT / "corpus")
    index = HybridIndex(chunks)
    graph = EvidenceGraph(index, abstain_threshold=0.12, hitl_enabled=True, provider_name="local")
    judge = HeuristicJudge()
    gold = load_gold()
    unans = [g for g in gold if g.get("expected_status") == "abstain"]
    answerable = [g for g in gold if g.get("expected_status") in {"answered", "hitl_pending"}]
    multimodal = [g for g in gold if g.get("must_cite_image_id")]

    abstain_ok = 0
    abstain_fail: list[str] = []
    for item in unans:
        resp = graph.invoke(item["question"], high_stakes=item.get("high_stakes", False))
        if resp.status.value == "abstain":
            abstain_ok += 1
        else:
            abstain_fail.append(item["id"])
    abstain_rate = abstain_ok / max(1, len(unans))

    hit5_ok = 0
    rec_sum = 0.0
    faith_sum = 0.0
    judge_sum = 0.0
    faith_n = 0
    for item in answerable:
        required = item.get("must_cite_chunk_ids") or []
        hits = index.search(item["question"], top_k=5)
        ids = [h.chunk.chunk_id for h in hits]
        if hit_at_k(ids, required, 5):
            hit5_ok += 1
        rec_sum += recall_at_k(ids, required, 5)
        resp = graph.invoke(item["question"], high_stakes=item.get("high_stakes", False))
        cited_texts = [c.quote for c in resp.citations]
        faith_sum += faithfulness(resp.answer, cited_texts, resp.status.value)
        judge_sum += judge.score(
            item["question"], resp.answer, cited_texts, resp.status.value
        ).score
        faith_n += 1

    mm_ok = 0
    for item in multimodal:
        resp = graph.invoke(item["question"], high_stakes=item.get("high_stakes", False))
        cited_ids = [c.chunk_id for c in resp.citations]
        if citation_id_match(cited_ids, item.get("must_cite_image_id")):
            mm_ok += 1

    hit5 = hit5_ok / max(1, len(answerable))
    recall5 = rec_sum / max(1, len(answerable))
    faith = faith_sum / max(1, faith_n)
    judge_score = judge_sum / max(1, faith_n)
    mm_rate = mm_ok / max(1, len(multimodal))

    summary = {
        "n_gold": len(gold),
        "n_unanswerable": len(unans),
        "n_answerable": len(answerable),
        "n_multimodal": len(multimodal),
        "abstain_on_unanswerable": abstain_rate,
        "failures": abstain_fail,
        "hit_at_5": hit5,
        "recall_at_5": recall5,
        "faithfulness": faith,
        "judge_score": judge_score,
        "judge_backend": judge.backend,
        "multimodal_citation_id_match": mm_rate,
    }

    table = _table(
        [
            ("gold items", str(len(gold))),
            ("unanswerable", str(len(unans))),
            ("answerable (incl HITL)", str(len(answerable))),
            ("multimodal (image id required)", str(len(multimodal))),
            ("abstain-on-unanswerable", f"{abstain_rate:.1%}"),
            ("Hit@5 (any required chunk)", f"{hit5:.1%}"),
            ("Recall@5 (required chunks)", f"{recall5:.1%}"),
            ("faithfulness (lexical vs cites)", f"{faith:.1%}"),
            ("judge score (heuristic)", f"{judge_score:.1%}"),
            ("multimodal citation-id match", f"{mm_rate:.1%}"),
            ("abstain failures", str(abstain_fail or "none")),
        ]
    )
    (ROOT / "evals" / "latest.md").write_text(
        "# AbCite eval snapshot (local extractive graph)\n\n"
        "```\n"
        f"{table}\n"
        "```\n\n"
        "CI gate: abstain-on-unanswerable must be 100% on this fixture set.\n"
        "Judge: HeuristicJudge in CI; LLMJudge when OPENAI_API_KEY/Azure is set "
        "(cache under evals/.cache). See evals/judge_alignment.json.\n",
        encoding="utf-8",
    )
    print(table)
    return summary


def main() -> None:
    summary = run_gate()
    if summary["abstain_on_unanswerable"] < 1.0:
        raise SystemExit(f"GATE FAIL: {summary}")
    if summary["n_gold"] < 150:
        raise SystemExit(f"GATE FAIL gold < 150: {summary['n_gold']}")
    print("GATE PASS")


if __name__ == "__main__":
    main()
