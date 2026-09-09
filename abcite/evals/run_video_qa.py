"""Video QA eval: transcript BM25 vs frame CLIP vs audio vs within-video RRF."""

from __future__ import annotations

import json
from pathlib import Path

from abcite_core.video_rag import VideoGallery, evaluate_video_qa, load_video_qa_gold, load_video_records

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from abcite_core.clip_encoder import build_vision_encoder

    recs = load_video_records(prefer_cache=True)
    gal = VideoGallery(recs, vision=build_vision_encoder())
    items = load_video_qa_gold(set(gal.ids))
    report = evaluate_video_qa(gal, items)
    dest = ROOT / "evals" / "video_qa_latest.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
