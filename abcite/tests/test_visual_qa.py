from pathlib import Path

import pytest

from abcite_core.figure_gallery import FigureGallery, FigureRec, load_figure_records
from abcite_core.holdout import query_contains_pdb, visual_query_ok
from abcite_core.models import Chunk, Modality
from abcite_core.retrieval import HybridIndex
from abcite_core.vision import LocalImageEncoder
from abcite_core.visual_qa import (
    caption_keyword_overlap,
    evaluate_visual_qa,
    heuristic_vqa,
    load_visual_qa_gold,
)
from agents.graph import EvidenceGraph
from evals.run import _multimodal_snapshot_md

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold" / "visual_qa.jsonl"
FIX = ROOT / "evals" / "figures" / "fixture"


def _fixture_recs() -> list[FigureRec]:
    if not (FIX / "metadata.jsonl").exists():
        pytest.skip("fixture not downloaded yet")
    recs = load_figure_records(prefer_cache=False)
    if len(recs) < 8:
        pytest.skip("need 8+ fixture images")
    return recs


def test_visual_qa_gold_constraints():
    import json

    rows = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 50
    golds: list[str] = []
    fail_n = 0
    for row in rows:
        assert visual_query_ok(row["question"])
        assert not query_contains_pdb(row["question"])
        assert "1n8z" not in row["question"].lower()
        assert "caption_bm25_should_fail" in row
        if row.get("caption_bm25_should_fail"):
            fail_n += 1
        if row.get("expected_status") != "abstain":
            assert row.get("gold_figure_ids")
            golds.append(row["gold_figure_ids"][0])
            assert row.get("must_look_at_image") is True
    assert len(golds) == len(set(golds))
    assert fail_n >= 20
    abstain = [r for r in rows if r.get("expected_status") == "abstain"]
    assert len(abstain) >= 4


def test_caption_bm25_should_fail_on_fixture_captions():
    recs = _fixture_recs()
    by_id = {r.figure_id: r for r in recs}
    items = load_visual_qa_gold(set(by_id), recs)
    flagged = [it for it in items if it.get("caption_bm25_should_fail") and it.get("gold_figure_ids")]
    assert len(flagged) >= 8
    for it in flagged:
        cap = by_id[it["gold_figure_ids"][0]].caption
        ov = caption_keyword_overlap(it["question"], cap)
        assert ov["should_fail"], (it["id"], ov["overlap_tokens"])


def test_visual_qa_eval_on_fixture():
    recs = _fixture_recs()
    gal = FigureGallery(recs, LocalImageEncoder())
    items = load_visual_qa_gold(set(gal.ids), recs)
    assert len([it for it in items if it.get("gold_figure_ids")]) >= 8
    report = evaluate_visual_qa(gal, items, rerank_k=5)
    assert report["protocol"] == "visual_qa"
    assert report["n_gallery"] == len(recs)
    for col in ("caption_bm25", "clip_image_only", "vision_rerank"):
        assert "recall@1" in report[col]
        assert "ndcg@5" in report[col]
    assert report["vqa"]["n"] >= 8
    assert "not the figure caption" in report["note"] or "not" in report["note"]


def test_vqa_abstains_when_figure_lacks_cues():
    recs = _fixture_recs()
    photo = next(r for r in recs if r.figure_id in {"fix-01", "pmc-00001"})
    miss_q = "rainbow circular dichroism polar plot with a magenta star on the origin"
    pred = heuristic_vqa(miss_q, photo, photo.path)
    assert pred.status == "abstain"
    assert pred.figure_id is None


def test_search_visual_and_graph_task():
    chunks = [
        Chunk(
            chunk_id="fig-alpha",
            parent_id="txt-a",
            modality=Modality.FIGURE,
            title="Alpha epitope cartoon",
            text="Figure caption: alpha-unique-caption membrane-proximal.",
            caption="alpha-unique-caption membrane-proximal domain IV",
            source_url="https://example.test/a",
            source_label="a",
            image_path="corpus/figures/fig-1n8z.svg",
        ),
        Chunk(
            chunk_id="fig-beta",
            parent_id="txt-b",
            modality=Modality.FIGURE,
            title="Beta epitope cartoon",
            text="Figure caption: beta-unique-caption cryptic face.",
            caption="beta-unique-caption cryptic rbd ace2-distal",
            source_url="https://example.test/b",
            source_label="b",
            image_path="corpus/figures/fig-6w41.svg",
        ),
    ]
    idx = HybridIndex(chunks, vision=LocalImageEncoder())
    hits = idx.search_visual("emerald nuclear speckles against a dark field", top_k=2)
    assert hits
    assert hasattr(hits[0], "image")
    g = EvidenceGraph(idx, 0.12, False, "local")
    resp = g.invoke(
        "emerald nuclear speckles against a dark field",
        task="visual_qa",
    )
    assert resp.status.value in {"answered", "abstain"}
    evidence = g.invoke(
        "Does trastuzumab on HER2 share the pertuzumab domain II epitope?",
        task="evidence",
    )
    assert evidence.trace_id != resp.trace_id


def test_eval_snapshot_keeps_visual_qa_primary():
    md = _multimodal_snapshot_md()
    if not md:
        pytest.skip("no multimodal snapshot json")
    assert "Protocol A" in md and "Protocol B" in md
    assert "Appendix" in md
    assert "N=15949" in md
    assert "0.835" in md
    assert "Rerank helps BM25? **no**" in md
