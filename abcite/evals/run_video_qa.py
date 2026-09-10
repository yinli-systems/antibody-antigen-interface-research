"""Video QA eval: transcript BM25 vs frame CLIP vs audio vs within-video RRF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abcite_core.video_rag import VideoGallery, evaluate_video_qa, load_video_qa_gold, load_video_records

ROOT = Path(__file__).resolve().parents[1]
OA_META = ROOT / "evals" / "video" / "oa" / "metadata.jsonl"
OA_GOLD = ROOT / "evals" / "gold" / "video_qa_oa.jsonl"


def main() -> None:
    from abcite_core.clip_encoder import build_vision_encoder

    ap = argparse.ArgumentParser()
    ap.add_argument("--oa", action="store_true", help="Wikimedia Commons gallery (not synthetic fixtures)")
    args = ap.parse_args()
    if args.oa:
        if not OA_META.exists():
            print("OA video missing. Run: python evals/download_av.py")
            return
        recs = load_video_records(meta_path=OA_META)
        gold_path = OA_GOLD
        dest = ROOT / "evals" / "video_qa_oa_latest.json"
    else:
        recs = load_video_records(prefer_cache=True)
        gold_path = None
        dest = ROOT / "evals" / "video_qa_latest.json"
    gal = VideoGallery(recs, vision=build_vision_encoder())
    items = load_video_qa_gold(set(gal.ids), path=gold_path)
    report = evaluate_video_qa(gal, items)
    report["oa"] = bool(args.oa)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
