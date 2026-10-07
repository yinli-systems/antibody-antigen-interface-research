"""Unified multimodal snapshot: text fixture + visual + audio + video tables."""

from __future__ import annotations

import json
from pathlib import Path

from abcite_core.audio_rag import AudioGallery, evaluate_audio_qa, load_audio_qa_gold, load_audio_records
from abcite_core.video_rag import VideoGallery, evaluate_video_qa, load_video_qa_gold, load_video_records

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    visual = {}
    vp = ROOT / "evals" / "visual_qa_protocols_latest.json"
    if vp.exists():
        visual = json.loads(vp.read_text(encoding="utf-8"))
    audio = evaluate_audio_qa(
        AudioGallery(load_audio_records(prefer_cache=True)),
        load_audio_qa_gold(set(r.audio_id for r in load_audio_records(prefer_cache=True))),
    )
    vrecs = load_video_records(prefer_cache=True)
    video = evaluate_video_qa(VideoGallery(vrecs), load_video_qa_gold({r.video_id for r in vrecs}))
    dest_a = ROOT / "evals" / "audio_qa_latest.json"
    dest_v = ROOT / "evals" / "video_qa_latest.json"
    dest_a.write_text(json.dumps(audio, indent=2), encoding="utf-8")
    dest_v.write_text(json.dumps(video, indent=2), encoding="utf-8")
    combined = {
        "protocol": "unified_multimodal",
        "visual_qa": visual.get("four_column") or visual,
        "audio_qa": audio,
        "video_qa": video,
        "fusion_policy": (
            "within-modality RRF only; never RRF vision into caption BM25 "
            "for text-only or visual_qa protocols"
        ),
    }
    out = ROOT / "evals" / "unified_latest.json"
    out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
