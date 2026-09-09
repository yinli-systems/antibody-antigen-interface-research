"""Audio QA eval: transcript BM25 vs CLAP/hash vs within-audio RRF."""

from __future__ import annotations

import json
from pathlib import Path

from abcite_core.audio_rag import AudioGallery, evaluate_audio_qa, load_audio_qa_gold, load_audio_records

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    recs = load_audio_records(prefer_cache=True)
    gal = AudioGallery(recs)
    items = load_audio_qa_gold(set(gal.ids))
    report = evaluate_audio_qa(gal, items)
    dest = ROOT / "evals" / "audio_qa_latest.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
