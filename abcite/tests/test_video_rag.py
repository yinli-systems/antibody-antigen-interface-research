from pathlib import Path

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
    assert len(recs) >= 4
    items = load_video_qa_gold(set(by))
    flagged = [it for it in items if it.get("transcript_bm25_should_fail") and it.get("gold_video_ids")]
    assert len(flagged) >= 3
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
    assert report["n_answerable"] >= 3
    assert "transcript_bm25" in report and "frame_clip_t2i" in report
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
