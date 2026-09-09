from pathlib import Path

import numpy as np
import pytest

from abcite_core.figure_gallery import (
    FigureGallery,
    FigureRec,
    caption_as_query_items,
    load_figure_records,
    recall_at_k,
)
from abcite_core.holdout import (
    query_contains_pdb,
    strip_accession_tokens,
    strip_pdb_ids,
    visual_query_ok,
)
from abcite_core.vision import LocalImageEncoder
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


def test_caption_as_query_fixture_three_columns():
    recs = _fixture_recs()
    gal = FigureGallery(recs, LocalImageEncoder())
    items = caption_as_query_items(recs)
    assert len(items) >= 8
    for it in items:
        assert it["gold_figure_ids"] == [it["id"].removeprefix("cap-")]
        assert not query_contains_pdb(it["question"])
        assert "1n8z" not in it["question"].lower()
        assert it["question"]
    report = evaluate(gal, items, protocol="caption_as_query")
    assert report["protocol"] == "caption_as_query"
    assert report["n_gallery"] == len(recs)
    assert report["n_queries"] == len(items)
    for col in ("caption_bm25", "clip_image_only", "fusion"):
        assert "recall@1" in report[col]
        assert "recall@5" in report[col]
        assert "recall@10" in report[col]
    # Lexical caption→caption BM25 is the SciMMIR-like text baseline.
    assert report["caption_bm25"]["recall@5"] == 1.0
    assert report["caption_bm25"]["recall@1"] >= 0.7
    assert "caption-as-query" in report["note"]
    if len(recs) >= 2:
        assert not np.allclose(gal.image_vecs[0], gal.image_vecs[1])


def test_fixture_gallery_three_columns():
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
    report = evaluate(gal, items, protocol="unique_gold_visual")
    for col in ("caption_bm25", "clip_image_only", "fusion"):
        assert col in report
        assert "recall@1" in report[col]
    assert "CLIP multimodal" not in report["note"]
    assert report["n_gallery"] >= 8
    if len(recs) >= 2:
        assert not np.allclose(gal.image_vecs[0], gal.image_vecs[1])


def test_visual_gold_remaps_fix_to_pmc():
    pmc_ids = {f"pmc-{i:05d}" for i in range(1, 14)}
    gold = load_visual_gold(pmc_ids)
    assert gold
    used = {g for row in gold for g in row["gold_figure_ids"]}
    assert any(x.startswith("pmc-") for x in used)
    assert not any(x.startswith("fix-") for x in used)


def test_recall_helper():
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_at_k(["a", "b"], ["c"], 2) == 0.0
