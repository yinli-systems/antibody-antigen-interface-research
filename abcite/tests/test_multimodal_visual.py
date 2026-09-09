from pathlib import Path

import numpy as np
import pytest

from abcite_core.figure_gallery import FigureGallery, FigureRec, recall_at_k
from abcite_core.holdout import query_contains_pdb, strip_pdb_ids, visual_query_ok
from abcite_core.vision import LocalImageEncoder
from evals.run_multimodal import evaluate, load_visual_gold

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "evals" / "figures" / "fixture"


def test_pdb_holdout_and_visual_queries():
    assert query_contains_pdb("compare 1N8Z trastuzumab")
    assert not query_contains_pdb("western blot of IgG")
    assert "1n8z" not in strip_pdb_ids("the 1N8Z complex ribbon").lower()
    gold = load_visual_gold({f"fix-{i:02d}" for i in range(1, 13)})
    assert gold
    for row in gold:
        assert visual_query_ok(row["question"])
        assert not query_contains_pdb(row["question"])
        assert "1n8z" not in row["question"].lower()


def test_fixture_gallery_three_columns():
    if not (FIX / "metadata.jsonl").exists():
        pytest.skip("fixture not downloaded yet")
    recs = []
    import json

    for line in (FIX / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        path = ROOT / "evals" / "figures" / raw["path"]
        if path.is_file():
            recs.append(
                FigureRec(
                    raw["figure_id"],
                    path,
                    raw.get("caption") or "",
                    raw.get("pmcid"),
                    "fixture",
                    raw.get("visual_class") or "scientific plot",
                    "fixture",
                )
            )
    if len(recs) < 8:
        pytest.skip("need 8+ fixture images")
    gal = FigureGallery(recs, LocalImageEncoder())
    items = load_visual_gold(set(gal.ids))
    if len(items) < 3:
        items = [
            {
                "id": "t",
                "question": "scientific plot panel from a paper figure",
                "gold_figure_ids": [gal.ids[0]],
            }
        ]
    report = evaluate(gal, items)
    for col in ("caption_bm25", "clip_image_only", "fusion"):
        assert col in report
        assert "recall@1" in report[col]
    assert "CLIP multimodal" not in report["note"]
    assert report["n_gallery"] >= 8
    # Image vectors must not be caption hashing only: two files differ.
    if len(recs) >= 2:
        assert not np.allclose(gal.image_vecs[0], gal.image_vecs[1])


def test_recall_helper():
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_at_k(["a", "b"], ["c"], 2) == 0.0
