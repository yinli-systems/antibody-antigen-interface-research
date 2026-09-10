"""Audio QA eval: transcript BM25 vs CLAP/hash vs within-audio RRF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abcite_core.audio_rag import AudioGallery, evaluate_audio_qa, load_audio_qa_gold, load_audio_records

ROOT = Path(__file__).resolve().parents[1]
OA_META = ROOT / "evals" / "audio" / "oa" / "metadata.jsonl"
OA_GOLD = ROOT / "evals" / "gold" / "audio_qa_oa.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oa", action="store_true", help="Wikimedia Commons gallery (not synthetic fixtures)")
    args = ap.parse_args()
    if args.oa:
        if not OA_META.exists():
            raise SystemExit("OA audio missing. Run: python evals/download_av.py")
        recs = load_audio_records(meta_path=OA_META)
        gold_path = OA_GOLD
        dest = ROOT / "evals" / "audio_qa_oa_latest.json"
    else:
        recs = load_audio_records(prefer_cache=True)
        gold_path = None
        dest = ROOT / "evals" / "audio_qa_latest.json"
    gal = AudioGallery(recs)
    items = load_audio_qa_gold(set(gal.ids), path=gold_path)
    report = evaluate_audio_qa(gal, items)
    report["oa"] = bool(args.oa)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
