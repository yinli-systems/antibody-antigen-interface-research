"""Download short CC science clips (optional). Fixtures stay synthetic.

Tries Wikimedia Commons File Paths for short open clips. Never required for CI.
Usage:
  python evals/download_av.py --max 8
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "evals" / "av_cache"
UA = "abcite-av-downloader/0.1 (research eval; mailto:abcite@local)"
CTX = ssl.create_default_context()

# Short, openly licensed science clips (Wikimedia Commons). May 404; script is best-effort.
CANDIDATES = [
    (
        "wm-electrophoresis",
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/3/3a/Agarose_gel_electrophoresis.ogv/Agarose_gel_electrophoresis.ogv.480p.vp9.webm",
        "Gel electrophoresis demonstration (Wikimedia Commons).",
    ),
    (
        "wm-microscope",
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/4/47/Optical_microscope.ogv/Optical_microscope.ogv.480p.vp9.webm",
        "Optical microscope demonstration (Wikimedia Commons).",
    ),
]


def _get(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=30) as resp:
            dest.write_bytes(resp.read())
        return dest.stat().st_size > 1024
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=8)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (vid, url, title) in enumerate(CANDIDATES[: args.max]):
        dest = CACHE / f"{vid}.webm"
        ok = dest.exists() or _get(url, dest)
        print(f"{vid} {'ok' if ok else 'fail'} {url}")
        if not ok:
            continue
        from abcite_core.video_rag import extract_keyframes

        frames = extract_keyframes(dest, CACHE / vid, max_frames=4)
        rows.append(
            {
                "video_id": vid,
                "frames": [str(p.relative_to(ROOT)) for p in frames],
                "audio_path": None,
                "transcript": title,
                "title": title,
                "source": "wikimedia-commons",
                "href": url,
                "synthetic": False,
            }
        )
    meta = ROOT / "evals" / "video" / "cache" / "metadata.jsonl"
    if rows:
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"downloaded={len(rows)} (synthetic fixtures remain the CI default)")


if __name__ == "__main__":
    main()
