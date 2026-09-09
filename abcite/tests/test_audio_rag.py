from pathlib import Path

from abcite_core.audio_rag import (
    AudioGallery,
    audio_qa_response,
    evaluate_audio_qa,
    load_audio_qa_gold,
    load_audio_records,
)
from abcite_core.models import AnswerStatus
from abcite_core.visual_qa import caption_keyword_overlap as vis_overlap

ROOT = Path(__file__).resolve().parents[1]


def test_audio_gold_and_bm25_should_fail():
    recs = load_audio_records(prefer_cache=False)
    by = {r.audio_id: r for r in recs}
    assert len(recs) >= 6
    items = load_audio_qa_gold(set(by))
    flagged = [it for it in items if it.get("transcript_bm25_should_fail") and it.get("gold_audio_ids")]
    assert len(flagged) >= 5
    for it in flagged:
        cap = by[it["gold_audio_ids"][0]].transcript
        ov = vis_overlap(it["question"], cap)
        assert ov["should_fail"], (it["id"], ov["overlap_tokens"])


def test_audio_eval_and_cite_or_abstain():
    recs = load_audio_records(prefer_cache=False)
    gal = AudioGallery(recs)
    items = load_audio_qa_gold(set(gal.ids))
    report = evaluate_audio_qa(gal, items)
    assert report["protocol"] == "audio_qa"
    assert report["n_gallery"] == len(recs)
    assert report["n_answerable"] >= 5
    assert "transcript_bm25" in report and "clap_t2a" in report
    resp = audio_qa_response(
        items[0]["question"],
        gal,
        provider="local",
        trace_id="t-audio",
        latency_ms=1.0,
    )
    assert resp.status in {AnswerStatus.ANSWERED, AnswerStatus.ABSTAIN}
    abs_q = next(it for it in items if it.get("expected_status") == "abstain")
    abs_resp = audio_qa_response(abs_q["question"], gal, provider="local", trace_id="t-abs", latency_ms=1.0)
    assert abs_resp.status == AnswerStatus.ABSTAIN
