from pathlib import Path

import pytest

from abcite_core.models import AnswerStatus
from abcite_core.video_rag import (
    VideoGallery,
    evaluate_video_qa,
    load_video_qa_gold,
    load_video_records,
    video_qa_response,
)
from abcite_core.visual_qa import caption_keyword_overlap
from abcite_core.vision import LocalImageEncoder

ROOT = Path(__file__).resolve().parents[1]


def test_video_gold_and_bm25_should_fail():
    recs = load_video_records(prefer_cache=False)
    by = {r.video_id: r for r in recs}
    assert len(recs) >= 12
    items = load_video_qa_gold(set(by))
    flagged = [it for it in items if it.get("transcript_bm25_should_fail") and it.get("gold_video_ids")]
    assert len(flagged) >= 8
    for it in flagged:
        cap = by[it["gold_video_ids"][0]].transcript
        ov = caption_keyword_overlap(it["question"], cap)
        assert ov["should_fail"], (it["id"], ov["overlap_tokens"])


def test_video_eval_and_cite_or_abstain():
    recs = load_video_records(prefer_cache=False)
    gal = VideoGallery(recs, vision=LocalImageEncoder())
    items = load_video_qa_gold(set(gal.ids))
    report = evaluate_video_qa(gal, items)
    assert report["protocol"] == "video_qa"
    assert report["n_gallery"] == len(recs)
    assert report["n_answerable"] >= 8
    assert "transcript_bm25" in report and "frame_clip_t2i" in report
    assert report["frames_beat_transcript_bm25_r@1"]
    resp = video_qa_response(
        items[0]["question"],
        gal,
        provider="local",
        trace_id="t-video",
        latency_ms=1.0,
    )
    assert resp.status in {AnswerStatus.ANSWERED, AnswerStatus.ABSTAIN}
    abs_q = next(it for it in items if it.get("expected_status") == "abstain")
    abs_resp = video_qa_response(abs_q["question"], gal, provider="local", trace_id="t-vabs", latency_ms=1.0)
    assert abs_resp.status == AnswerStatus.ABSTAIN


def test_oa_video_candidates_should_fail_filler_bm25():
    from evals.download_av import FILLER, VIDEO_CANDIDATES

    blob = " ".join(FILLER)
    ids = [c["id"] for c in VIDEO_CANDIDATES]
    assert len(ids) == len(set(ids))
    assert len(VIDEO_CANDIDATES) >= 8
    for cand in VIDEO_CANDIDATES:
        ov = caption_keyword_overlap(cand["question"], blob)
        assert ov["should_fail"], (cand["id"], ov["overlap_tokens"])


def test_oa_video_gold_unique_ids():
    gold = ROOT / "evals" / "gold" / "video_qa_oa.jsonl"
    if not gold.exists():
        pytest.skip("no OA video gold")
    rows = [
        __import__("json").loads(line)
        for line in gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    ans = [r for r in rows if r.get("gold_video_ids")]
    assert len(ans) >= 8
    golds = [r["gold_video_ids"][0] for r in ans]
    assert len(golds) == len(set(golds))
    for r in rows:
        assert r.get("transcript_bm25_should_fail") is True
