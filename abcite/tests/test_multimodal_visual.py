from pathlib import Path

import numpy as np
import pytest

from abcite_core.figure_gallery import (
    FigureGallery,
    FigureRec,
    caption_as_query_items,
    load_figure_records,
    recall_at_k,
    vision_rerank_scores,
)
from abcite_core.holdout import (
    query_contains_pdb,
    split_by_paper,
    strip_accession_tokens,
    strip_pdb_ids,
    visual_query_ok,
)
from abcite_core.models import Chunk, Modality
from abcite_core.retrieval import HybridIndex
from abcite_core.vision import LocalImageEncoder
from abcite_core.vision_flags import vision_fusion_enabled, vision_rerank_enabled
from evals.run_multimodal import evaluate, load_visual_gold

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "evals" / "figures" / "fixture"


def _fixture_recs() -> list[FigureRec]:
    if not (FIX / "metadata.jsonl").exists():
        pytest.skip("fixture not downloaded yet")
    recs = load_figure_records(prefer_cache=False)
    if len(recs) < 8:
        pytest.skip("need 8+ fixture images")
    return recs


def test_pdb_holdout_and_visual_queries():
    assert query_contains_pdb("compare 1N8Z trastuzumab")
    assert not query_contains_pdb("western blot of IgG")
    assert "1n8z" not in strip_pdb_ids("the 1N8Z complex ribbon").lower()
    stripped = strip_accession_tokens("PDB 1N8Z UniProt P04626 NP_004439.2 western blot")
    low = stripped.lower()
    assert "1n8z" not in low
    assert "p04626" not in low
    assert "np_004439" not in low
    assert "pdb" not in low
    assert "uniprot" not in low
    gold = load_visual_gold({f"fix-{i:02d}" for i in range(1, 13)})
    assert gold
    for row in gold:
        assert visual_query_ok(row["question"])
        assert not query_contains_pdb(row["question"])
        assert "1n8z" not in row["question"].lower()


def test_caption_as_query_fixture_four_columns():
    recs = _fixture_recs()
    gal = FigureGallery(recs, LocalImageEncoder())
    items = caption_as_query_items(recs)
    assert len(items) >= 8
    for it in items:
        assert it["gold_figure_ids"] == [it["id"].removeprefix("cap-")]
        assert not query_contains_pdb(it["question"])
        assert "1n8z" not in it["question"].lower()
        assert it["question"]
    report = evaluate(gal, items, protocol="caption_as_query", rerank_k=5)
    assert report["protocol"] == "caption_as_query"
    assert report["n_gallery"] == len(recs)
    assert report["n_queries"] == len(items)
    assert report["fusion_default"] == "off"
    assert report["rerank_k"] == 5
    for col in ("caption_bm25", "clip_image_only", "vision_rerank", "fusion"):
        assert "recall@1" in report[col]
        assert "recall@5" in report[col]
        assert "recall@10" in report[col]
    # Lexical caption→caption BM25 is the SciMMIR-like text baseline.
    assert report["caption_bm25"]["recall@5"] == 1.0
    assert report["caption_bm25"]["recall@1"] >= 0.7
    assert "caption-as-query" in report["note"]
    assert "rerank_helps_bm25" in report
    if len(recs) >= 2:
        assert not np.allclose(gal.image_vecs[0], gal.image_vecs[1])


def test_fixture_gallery_four_columns():
    recs = _fixture_recs()
    gal = FigureGallery(recs, LocalImageEncoder())
    items = load_visual_gold(set(gal.ids), recs)
    if len(items) < 3:
        items = [
            {
                "id": "t",
                "question": "scientific plot panel from a paper figure",
                "gold_figure_ids": [gal.ids[0]],
            }
        ]
    report = evaluate(gal, items, protocol="unique_gold_visual", rerank_k=5)
    for col in ("caption_bm25", "clip_image_only", "vision_rerank", "fusion"):
        assert col in report
        assert "recall@1" in report[col]
    assert "CLIP multimodal" not in report["note"]
    assert report["n_gallery"] >= 8
    if len(recs) >= 2:
        assert not np.allclose(gal.image_vecs[0], gal.image_vecs[1])


def test_paper_holdout_query_papers_disjoint():
    recs = _fixture_recs()
    split = split_by_paper(recs, query_frac=0.2, seed=7)
    assert split.query_papers.isdisjoint(split.index_papers)
    assert split.query_recs
    assert split.n_papers >= 2
    query_ids = {r.figure_id for r in split.query_recs}
    index_ids = {r.figure_id for r in split.index_recs}
    assert query_ids.isdisjoint(index_ids)
    items = caption_as_query_items(list(split.query_recs))
    gal = FigureGallery(recs, LocalImageEncoder())
    report = evaluate(gal, items, protocol="caption_as_query", rerank_k=5)
    assert report["n_queries"] == len(items)
    for it in items:
        pmc = it.get("pmcid")
        if pmc:
            assert pmc in split.query_papers


def test_vision_rerank_reorders_topk_only():
    recs = _fixture_recs()
    gal = FigureGallery(recs, LocalImageEncoder())
    n = gal.n
    text = np.arange(n, 0, -1, dtype=np.float32)
    image = np.zeros(n, dtype=np.float32)
    image[0] = 0.1
    image[1] = 0.9
    ranked = gal.ranked_ids(vision_rerank_scores(text, image, k=2))
    assert ranked[0] == gal.ids[1]
    assert ranked[1] == gal.ids[0]
    assert ranked[2:] == gal.ids[2:]
    ranked_k1 = gal.ranked_ids(vision_rerank_scores(text, image, k=1))
    assert ranked_k1[0] == gal.ids[0]
    assert ranked_k1[1] == gal.ids[1]


def test_vision_fusion_off_rerank_off_by_default(monkeypatch):
    monkeypatch.delenv("ABCITE_VISION_FUSION", raising=False)
    monkeypatch.delenv("ABCITE_VISION_RERANK", raising=False)
    assert vision_fusion_enabled() is False
    assert vision_rerank_enabled() is False

    class InvertVision(LocalImageEncoder):
        name = "invert-local"

        def encode_text(self, text: str) -> list[float]:
            vec = [0.0] * 256
            if "alpha-unique-caption" in text:
                vec[0] = 1.0
            else:
                vec[1] = 1.0
            return vec

        def encode_image(self, path: Path) -> list[float]:
            vec = [0.0] * 256
            # Invert: 6w41 image looks like the alpha query.
            if "6w41" in path.name:
                vec[0] = 1.0
            else:
                vec[1] = 1.0
            return vec

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
    idx = HybridIndex(chunks, vision=InvertVision())
    hits = idx.search("alpha-unique-caption membrane-proximal", top_k=2)
    assert hits[0].chunk.chunk_id == "fig-alpha"
    monkeypatch.setenv("ABCITE_VISION_RERANK", "1")
    monkeypatch.setenv("ABCITE_VISION_RERANK_K", "2")
    idx_r = HybridIndex(chunks, vision=InvertVision())
    reranked = idx_r.search("alpha-unique-caption membrane-proximal", top_k=2)
    assert reranked[0].chunk.chunk_id == "fig-beta"


def test_visual_gold_remaps_fix_to_pmc():
    pmc_ids = {f"pmc-{i:05d}" for i in range(1, 14)}
    gold = load_visual_gold(pmc_ids)
    assert gold
    used = {g for row in gold for g in row["gold_figure_ids"]}
    assert any(x.startswith("pmc-") for x in used)
    assert not any(x.startswith("fix-") for x in used)


def test_visual_gold_grows_without_pdb():
    rows = load_visual_gold({f"pmc-{i:05d}" for i in range(1, 1200)})
    assert len(rows) > 19
    for row in rows:
        assert visual_query_ok(row["question"])
        assert not query_contains_pdb(row["question"])


def test_eval_snapshot_keeps_protocol_tables():
    from evals.run import _multimodal_snapshot_md

    md = _multimodal_snapshot_md()
    if not md:
        pytest.skip("no multimodal_protocols_latest.json")
    assert "Protocol A" in md and "Protocol B" in md
    assert "N=15949" in md
    assert "0.835" in md
    assert "Rerank helps BM25? **no**" in md


def test_recall_helper():
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_at_k(["a", "b"], ["c"], 2) == 0.0
