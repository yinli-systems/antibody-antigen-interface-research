"""CI eval gate: fixture vs hard-dev, claim faithfulness, CLIP gap, failure taxonomy."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from abcite_core.judge import HeuristicJudge
from abcite_core.metrics import (
    citation_id_match,
    claim_faithfulness,
    claim_nli_heuristic,
    cohens_kappa,
    faithfulness,
    hit_at_k,
    label_failure,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from abcite_core.rcsb import fetch_entry
from abcite_core.retrieval import HybridIndex
from agents.graph import EvidenceGraph
from evals.clip_margin import clip_contrast
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "evals" / "gold"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_split(name: str) -> list[dict]:
    return _read_jsonl(GOLD_DIR / f"{name}.jsonl")


def load_gold() -> list[dict]:
    """All labeled items (fixture + hard). CI abstain gate uses fixture only."""
    return load_split("fixture") + load_split("hard")


def load_fixture() -> list[dict]:
    return load_split("fixture")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _table(rows: list[tuple[str, str]]) -> str:
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k.ljust(width)}  {v}" for k, v in rows)


def _metrics_for(items: list[dict], index: HybridIndex, graph: EvidenceGraph, judge: HeuristicJudge) -> dict:
    unans = [g for g in items if g.get("expected_status") == "abstain"]
    answerable = [g for g in items if g.get("expected_status") in {"answered", "hitl_pending"}]
    multimodal = [g for g in items if g.get("must_cite_image_id")]

    abstain_ok = 0
    abstain_fail: list[str] = []
    failures: list[str] = []
    taxonomy_rows: list[dict] = []
    by_source: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for item in unans:
        hits = index.search(item["question"], top_k=10)
        ids = [h.chunk.chunk_id for h in hits]
        resp = graph.invoke(item["question"], high_stakes=item.get("high_stakes", False))
        if resp.status.value == "abstain":
            abstain_ok += 1
        else:
            abstain_fail.append(item["id"])
        cited = [c.chunk_id for c in resp.citations]
        lab = label_failure(
            expected_status="abstain",
            actual_status=resp.status.value,
            cited_ids=cited,
            retrieved_ids=ids,
            required_ids=item.get("must_cite_chunk_ids") or [],
            query_class=item.get("query_class"),
            required_image_id=item.get("must_cite_image_id"),
        )
        failures.append(lab)
        taxonomy_rows.append(
            {
                "id": item["id"],
                "query_class": item.get("query_class"),
                "class": lab,
                "expected_status": "abstain",
                "actual_status": resp.status.value,
                "retrieved_ids": ids[:8],
                "cited_ids": cited,
            }
        )

    hit5_flags: list[float] = []
    recs: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []
    faiths: list[float] = []
    claims: list[float] = []
    nlis: list[float] = []
    judges: list[float] = []

    for item in answerable:
        required = item.get("must_cite_chunk_ids") or []
        hits = index.search(item["question"], top_k=10)
        ids = [h.chunk.chunk_id for h in hits]
        h5 = 1.0 if hit_at_k(ids, required, 5) else 0.0
        r5 = recall_at_k(ids, required, 5)
        nd = ndcg_at_k(ids, required, 10)
        rr = mrr(ids, required)
        hit5_flags.append(h5)
        recs.append(r5)
        ndcgs.append(nd)
        mrrs.append(rr)
        src = item.get("source") or "unknown"
        by_source[src]["hit5"].append(h5)
        by_source[src]["ndcg10"].append(nd)
        by_source[src]["mrr"].append(rr)
        resp = graph.invoke(item["question"], high_stakes=item.get("high_stakes", False))
        cited_texts = [c.quote for c in resp.citations]
        cited_ids = [c.chunk_id for c in resp.citations]
        faiths.append(faithfulness(resp.answer, cited_texts, resp.status.value))
        claims.append(claim_faithfulness(resp.answer, cited_texts, resp.status.value))
        nlis.append(claim_nli_heuristic(resp.answer, cited_texts, resp.status.value))
        judges.append(judge.score(item["question"], resp.answer, cited_texts, resp.status.value).score)
        lab = label_failure(
            expected_status=item["expected_status"],
            actual_status=resp.status.value,
            cited_ids=cited_ids,
            retrieved_ids=ids,
            required_ids=required,
            query_class=item.get("query_class"),
            required_image_id=item.get("must_cite_image_id"),
        )
        failures.append(lab)
        taxonomy_rows.append(
            {
                "id": item["id"],
                "query_class": item.get("query_class"),
                "class": lab,
                "expected_status": item["expected_status"],
                "actual_status": resp.status.value,
                "retrieved_ids": ids[:8],
                "cited_ids": cited_ids,
            }
        )

    mm_ok = 0
    for item in multimodal:
        resp = graph.invoke(item["question"], high_stakes=item.get("high_stakes", False))
        cited_ids = [c.chunk_id for c in resp.citations]
        if citation_id_match(cited_ids, item.get("must_cite_image_id")):
            mm_ok += 1

    source_rows = {}
    for src, buckets in by_source.items():
        source_rows[src] = {
            "n": len(buckets["hit5"]),
            "hit_at_5": _mean(buckets["hit5"]),
            "ndcg_at_10": _mean(buckets["ndcg10"]),
            "mrr": _mean(buckets["mrr"]),
        }

    return {
        "n_gold": len(items),
        "n_unanswerable": len(unans),
        "n_answerable": len(answerable),
        "n_multimodal": len(multimodal),
        "abstain_on_unanswerable": abstain_ok / max(1, len(unans)),
        "failures_ids": abstain_fail,
        "hit_at_5": _mean(hit5_flags),
        "recall_at_5": _mean(recs),
        "ndcg_at_10": _mean(ndcgs),
        "mrr": _mean(mrrs),
        "faithfulness_lexical": _mean(faiths),
        "faithfulness_claim": _mean(claims),
        "faithfulness_claim_nli": _mean(nlis),
        "judge_score": _mean(judges),
        "multimodal_citation_id_match": mm_ok / max(1, len(multimodal)),
        "by_source": source_rows,
        "failure_counts": dict(Counter(failures)),
        "taxonomy": taxonomy_rows,
    }


def _fmt_split(name: str, m: dict) -> str:
    rows = [
        (f"{name} items", str(m["n_gold"])),
        ("unanswerable", str(m["n_unanswerable"])),
        ("answerable (incl HITL)", str(m["n_answerable"])),
        ("multimodal (image id required)", str(m["n_multimodal"])),
        ("abstain-on-unanswerable", f"{m['abstain_on_unanswerable']:.1%}"),
        ("Hit@5", f"{m['hit_at_5']:.1%}"),
        ("Recall@5", f"{m['recall_at_5']:.1%}"),
        ("nDCG@10", f"{m['ndcg_at_10']:.3f}"),
        ("MRR", f"{m['mrr']:.3f}"),
        ("faithfulness (lexical only)", f"{m['faithfulness_lexical']:.1%}"),
        ("faithfulness (claim-span)", f"{m['faithfulness_claim']:.1%}"),
        ("claim-NLI heuristic", f"{m['faithfulness_claim_nli']:.1%}"),
        ("judge score (lexical heuristic)", f"{m['judge_score']:.1%}"),
        ("multimodal citation-id match", f"{m['multimodal_citation_id_match']:.1%}"),
        ("auto failure counts", json.dumps(m["failure_counts"])),
        ("taxonomy n", str(len(m.get("taxonomy") or []))),
        ("abstain failures", str(m["failures_ids"] or "none")),
    ]
    for src, sm in sorted(m.get("by_source", {}).items()):
        rows.append(
            (
                f"  {src} n={sm['n']}",
                f"Hit@5 {sm['hit_at_5']:.1%}  nDCG@10 {sm['ndcg_at_10']:.3f}  MRR {sm['mrr']:.3f}",
            )
        )
    return _table(rows)


def load_human_labels() -> dict:
    path = ROOT / "evals" / "human_labels.json"
    if not path.exists():
        return {"items": [], "n_annotators": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def judge_alignment_report(graph: EvidenceGraph, judge: HeuristicJudge) -> dict:
    human = load_human_labels()
    items = human.get("items") or []
    gold_by_id = {g["id"]: g for g in load_gold()}
    h_bin: list[int] = []
    lex_bin: list[int] = []
    nli_bin: list[int] = []
    details = []
    for row in items:
        gold = gold_by_id.get(row["id"])
        if not gold:
            continue
        resp = graph.invoke(gold["question"], high_stakes=gold.get("high_stakes", False))
        cites = [c.quote for c in resp.citations]
        lex = faithfulness(resp.answer, cites, resp.status.value)
        nli = claim_nli_heuristic(resp.answer, cites, resp.status.value)
        hf = float(row.get("human_faithfulness", 0.0))
        h_bin.append(1 if hf >= 0.5 else 0)
        lex_bin.append(1 if lex >= 0.5 else 0)
        nli_bin.append(1 if nli >= 0.5 else 0)
        details.append(
            {
                "id": row["id"],
                "human_faithfulness": hf,
                "lexical": round(lex, 4),
                "claim_nli": round(nli, 4),
                "claim_span": round(claim_faithfulness(resp.answer, cites, resp.status.value), 4),
            }
        )
    n_annotators = int(human.get("n_annotators") or 1)
    kappa_lex = cohens_kappa(h_bin, lex_bin)
    kappa_nli = cohens_kappa(h_bin, nli_bin)
    report = {
        "n_human_labeled": len(details),
        "n_annotators": n_annotators,
        "second_annotator": n_annotators >= 2,
        "cohens_kappa_human_vs_human": None,
        "cohens_kappa_human_vs_lexical": kappa_lex,
        "cohens_kappa_human_vs_claim_nli": kappa_nli,
        "note": human.get("note"),
        "kappa_note": (
            "Cohen's κ is human vs machine (lexical / claim-NLI), not inter-human κ. "
            "Second annotator is not a duplicate of the first; n_annotators=1."
            if n_annotators < 2
            else "Inter-human κ is available."
        ),
        "items": details,
    }
    (ROOT / "evals" / "judge_alignment.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def write_demo_1n8z(graph: EvidenceGraph, index: HybridIndex) -> dict:
    q = (
        "Does trastuzumab on HER2 (PDB 1N8Z) share the pertuzumab domain II epitope, "
        "and which public paper/PDB path supports that?"
    )
    hits = index.search(q, top_k=8)
    resp = graph.invoke(q, high_stakes=False)
    try:
        rcsb = fetch_entry("1N8Z")
    except Exception as exc:  # noqa: BLE001
        rcsb = {"error": str(exc)}
    receipt = {
        "id": "demo_1n8z",
        "query": q,
        "pdb_id": "1N8Z",
        "antibody": "trastuzumab",
        "antigen": "HER2",
        "paper": {
            "citation": "Cho et al., Nature 421:756-760 (2003)",
            "pmid": "12610673",
            "doi": "https://doi.org/10.1038/nature01392",
            "rcsb": "https://www.rcsb.org/structure/1N8Z",
        },
        "retrieved_ids": [h.chunk.chunk_id for h in hits],
        "retrieved_scores": {h.chunk.chunk_id: round(h.score, 4) for h in hits},
        "status": resp.status.value,
        "abstain": resp.status.value == "abstain",
        "cites": [
            {"chunk_id": c.chunk_id, "quote": c.quote[:280], "source_url": c.source_url}
            for c in resp.citations
        ],
        "answer_preview": resp.answer[:600],
        "rcsb_cache": {
            "pdb_id": rcsb.get("pdb_id"),
            "title": rcsb.get("title"),
            "cache_hit": rcsb.get("cache_hit"),
            "rcsb": rcsb.get("rcsb"),
        },
        "note": "Closed-index demo: Cho 2003 + RCSB 1N8Z cache, not a live PaperQA crawl.",
    }
    out = ROOT / "evals" / "traces" / "demo_1n8z.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def run_gate() -> dict:
    chunks = load_corpus(ROOT / "corpus")
    index = HybridIndex(chunks)
    graph = EvidenceGraph(index, abstain_threshold=0.12, hitl_enabled=True, provider_name="local")
    judge = HeuristicJudge()
    fixture = load_fixture()
    hard = load_split("hard")
    fixture_m = _metrics_for(fixture, index, graph, judge)
    hard_m = _metrics_for(hard, index, graph, judge)
    clip = clip_contrast(index)
    clip_public = {k: v for k, v in clip.items() if k != "per_query"}
    clip_hub_n = len(clip.get("hub_figures_rank1_for_others") or {})
    hard_fail = dict(hard_m["failure_counts"])
    if clip_hub_n:
        hard_fail["clip_hub"] = clip_hub_n

    tax_path = ROOT / "evals" / "traces" / "hard_taxonomy.json"
    tax_path.parent.mkdir(parents=True, exist_ok=True)
    tax_payload = {
        "n": len(hard_m.get("taxonomy") or []),
        "counts": hard_m["failure_counts"],
        "items": hard_m.get("taxonomy") or [],
        "note": "Auto labels on all hard-dev items. Classes: ok / hallucinated_cite / missed_abstain / over_abstain / miss_figure / miss_multi_hop.",
    }
    tax_path.write_text(json.dumps(tax_payload, indent=2) + "\n", encoding="utf-8")

    alignment = judge_alignment_report(graph, judge)
    demo = write_demo_1n8z(graph, index)

    summary = {
        "n_gold": len(fixture) + len(hard),
        "n_fixture": len(fixture),
        "n_hard": len(hard),
        "n_unanswerable": fixture_m["n_unanswerable"],
        "n_answerable": fixture_m["n_answerable"],
        "n_multimodal": fixture_m["n_multimodal"],
        "abstain_on_unanswerable": fixture_m["abstain_on_unanswerable"],
        "failures": fixture_m["failures_ids"],
        "hit_at_5": fixture_m["hit_at_5"],
        "recall_at_5": fixture_m["recall_at_5"],
        "faithfulness": fixture_m["faithfulness_lexical"],
        "faithfulness_claim": fixture_m["faithfulness_claim"],
        "judge_score": fixture_m["judge_score"],
        "judge_backend": judge.backend,
        "multimodal_citation_id_match": fixture_m["multimodal_citation_id_match"],
        "fixture": fixture_m,
        "hard": hard_m,
        "clip_contrast": clip_public,
        "failure_counts_hard": hard_fail,
        "hard_taxonomy_path": "evals/traces/hard_taxonomy.json",
        "judge_alignment": {
            "n_human_labeled": alignment["n_human_labeled"],
            "n_annotators": alignment["n_annotators"],
            "kappa_human_vs_human": alignment["cohens_kappa_human_vs_human"],
            "kappa_human_vs_lexical": alignment["cohens_kappa_human_vs_lexical"],
            "kappa_human_vs_claim_nli": alignment["cohens_kappa_human_vs_claim_nli"],
            "kappa_kind": "human_vs_machine",
            "second_annotator": alignment["second_annotator"],
        },
        "demo_1n8z": {
            "path": "evals/traces/demo_1n8z.json",
            "status": demo["status"],
            "retrieved_ids": demo["retrieved_ids"][:8],
            "cite_ids": [c["chunk_id"] for c in demo["cites"]],
            "abstain": demo["abstain"],
        },
    }

    t_fix = _fmt_split("FIXTURE", fixture_m)
    t_hard = _fmt_split("HARD-DEV", hard_m)
    clip_lines = _table(
        [
            ("CLIP encoder", str(clip_public.get("encoder"))),
            ("figure queries", str(clip_public.get("n_figure_queries"))),
            ("mean fused RRF gap", str(clip_public.get("mean_gap_rank1_minus_rank2"))),
            ("mean image-only cosine gap", str(clip_public.get("mean_gap_image_only"))),
            ("mean caption BM25 gap", str(clip_public.get("mean_caption_bm25_gap"))),
            ("min gap fused", str(clip_public.get("min_gap"))),
            ("self-hit rate", f"{clip_public.get('self_hit_rate', 0):.1%}"),
            ("hub figs (rank1 for >1 query)", str(clip_public.get("hub_figures_rank1_for_others"))),
            ("BiomedCLIP", str(clip_public.get("biomedclip_skip") or "not requested")),
            ("CLIP SOTA claim", "false"),
        ]
    )
    kappa_note = (
        f"{alignment.get('kappa_note')} "
        f"κ human-vs-lexical={alignment['cohens_kappa_human_vs_lexical']!r}; "
        f"κ human-vs-claim-NLI={alignment['cohens_kappa_human_vs_claim_nli']!r}; "
        f"κ human-vs-human={alignment['cohens_kappa_human_vs_human']!r}."
    )
    tax_counts = json.dumps(hard_m["failure_counts"])
    md = (
        "# AbCite eval snapshot (fixture vs hard-dev)\n\n"
        "Fixture is the in-corpus 150 used by CI (abstain-on-unanswerable must be 100%). "
        "Hard-dev is paraphrases / multi-hop / out-of-index / residue questions; "
        "Hit@5 is expected to drop. Do not tune retrieval to fake 99% on hard.\n\n"
        "## Fixture\n\n```\n"
        f"{t_fix}\n"
        "```\n\n## Hard-dev\n\n```\n"
        f"{t_hard}\n"
        "```\n\n## CLIP contrast (not SOTA)\n\n```\n"
        f"{clip_lines}\n"
        "```\n\n"
        f"Judge alignment (**human vs machine**, not inter-human κ): {kappa_note}\n\n"
        f"Hard taxonomy counts (all {hard_m['n_gold']} items): `{tax_counts}` "
        "— see `evals/traces/hard_taxonomy.json`.\n\n"
        "Demo receipt: `evals/traces/demo_1n8z.json` (Cho 2003 / RCSB 1N8Z).\n"
        "Lexical faithfulness is a smoke test only; claim-span / claim-NLI are still heuristics, not RAGAS.\n"
    )
    (ROOT / "evals" / "latest.md").write_text(md, encoding="utf-8")
    print(t_fix)
    print()
    print(t_hard)
    print()
    print(clip_lines)
    return summary


def main() -> None:
    summary = run_gate()
    if summary["abstain_on_unanswerable"] < 1.0:
        raise SystemExit(f"GATE FAIL fixture abstain: {summary['failures']}")
    if summary["n_fixture"] < 150:
        raise SystemExit(f"GATE FAIL fixture < 150: {summary['n_fixture']}")
    if summary["n_hard"] < 35:
        raise SystemExit(f"GATE FAIL hard < 35: {summary['n_hard']}")
    print("GATE PASS")


if __name__ == "__main__":
    main()
