"""Download short CC science clips for OA audio/video RAG.

CI fixtures stay synthetic (`evals/build_av_fixtures.py`). This script is
best-effort Wikimedia Commons. Never required for pytest.

Usage:
  python evals/download_av.py --max 12
  ABCITE_WHISPER=1 python evals/download_av.py   # ASR transcripts when ffmpeg+whisper exist
  python evals/run_audio_qa.py --oa
  python evals/run_video_qa.py --oa

Queries describe **sound or on-screen appearance**, not Wikipedia titles.
Transcripts default to generic lecture filler so BM25 should fail. Whisper
ASR replaces filler when ABCITE_WHISPER=1. Do not mix OA receipts with the
synthetic wav-hash table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "evals" / "av_cache"
AUD_OA = ROOT / "evals" / "audio" / "oa"
VID_OA = ROOT / "evals" / "video" / "oa"
AUD_GOLD = ROOT / "evals" / "gold" / "audio_qa_oa.jsonl"
VID_GOLD = ROOT / "evals" / "gold" / "video_qa_oa.jsonl"
UA = "abcite-av-downloader/0.2 (research eval; https://github.com/abcite)"
CTX = ssl.create_default_context()

# Generic lecture filler — no acoustic/scene cue tokens (same idea as fixtures).
FILLER = [
    "The seminar continues without a slide title.",
    "Please hold while the host adjusts the microphone.",
    "The chair asks the audience to wait a moment.",
    "Someone coughs then the feed resumes.",
    "A door closes in the hallway behind the speaker.",
    "Papers shuffle on the lectern.",
    "The technician says the capture light is on.",
    "Water is poured into a glass off microphone.",
    "A seat scrapes then the pause returns.",
    "The projector fan is mentioned only in passing.",
    "Someone whispers that the talk will start late.",
    "The ventilation is audible under the pause.",
]

# Acoustic queries must not overlap FILLER content tokens (≤1).
# File titles verified via Commons imageinfo (2026-09-10).
AUDIO_CANDIDATES = [
    {
        "id": "oa-aud-heartbeat",
        "file": "File:Heartbeat.ogg",
        "title": "Heartbeat recording",
        "question": "which clip is a repeating low thud pulse like a cardiac cycle",
        "answer": "A heartbeat / lub-dub pulse train.",
    },
    {
        "id": "oa-aud-geiger",
        "file": "File:Geiger_counter.ogg",
        "title": "Geiger counter clicks",
        "question": "sparse sharp click train, irregular, not a sine drone",
        "answer": "Geiger-like irregular clicks.",
    },
    {
        "id": "oa-aud-noise",
        "file": "File:White_noise.ogg",
        "title": "White noise",
        "question": "broadband hiss, not tonal",
        "answer": "White-ish hiss.",
    },
    {
        "id": "oa-aud-thunder",
        "file": "File:Thunder.ogg",
        "title": "Thunder",
        "question": "distant rumble burst, not a hiss",
        "answer": "Thunder rumble.",
    },
    {
        "id": "oa-aud-doppler",
        "file": "File:Fetal doppler recording heartbeat and movement.ogg",
        "title": "Fetal Doppler",
        "question": "whoosh pulse from a handheld fetal doppler, not a geiger click",
        "answer": "Fetal Doppler whoosh.",
    },
    {
        "id": "oa-aud-korotkoff",
        "file": "File:Nl-Korotkoff-tonen.ogg",
        "title": "Korotkoff sounds",
        "question": "soft cuff thumps used to read blood pressure",
        "answer": "Korotkoff thumps.",
    },
    {
        "id": "oa-aud-mosquito",
        "file": "File:Mosquito.ogg",
        "title": "Mosquito",
        "question": "thin high whine, not a hiss and not a thud pulse",
        "answer": "Mosquito-like high whine.",
    },
    {
        "id": "oa-aud-cricket",
        "file": "File:Cricket.ogg",
        "title": "Cricket chirp",
        "question": "rhythmic insect chirp train, not broadband hiss",
        "answer": "Cricket chirps.",
    },
    {
        "id": "oa-aud-siren",
        "file": "File:Siren.ogg",
        "title": "Police siren",
        "question": "wailing two-tone alarm glide, not a click train",
        "answer": "Two-tone siren wail.",
    },
    {
        "id": "oa-aud-a440",
        "file": "File:440 Hz.ogg",
        "title": "A440 tone",
        "question": "steady pure tone, not a pulse and not a hiss",
        "answer": "Steady A440-like tone.",
    },
    {
        "id": "oa-aud-modem",
        "file": "File:Dial-up_modem_noises.ogg",
        "title": "Dial-up modem",
        "question": "harsh handshake screech that is not a sine drone",
        "answer": "Dial-up modem handshake.",
    },
    {
        "id": "oa-aud-tuningfork",
        "file": "File:Tuning_fork.ogg",
        "title": "Tuning fork",
        "question": "decaying pure ring of a struck metal bar",
        "answer": "Tuning-fork ring.",
    },
]

VIDEO_CANDIDATES = [
    {
        "id": "oa-vid-gelbox",
        "file": "File:Conducting-Polymer-Electrodes-for-Gel-Electrophoresis-pone.0089416.s001.ogv",
        "title": "Conducting polymer gel electrophoresis",
        "question": "which clip shows a small gel box with running sample lanes on a bench",
        "answer": "Benchtop gel box with lanes.",
    },
    {
        "id": "oa-vid-fige",
        "file": "File:Increase-in-local-protein-concentration-by-field-inversion-gel-electrophoresis-1477-5956-5-18-S2.ogv",
        "title": "Field-inversion gel bands",
        "question": "which clip shows glowing nucleic-acid bands drifting in a dark slab",
        "answer": "Glowing bands in a dark gel slab.",
    },
    {
        "id": "oa-vid-anim",
        "file": "File:Electrophoresis explained - biology animation.webm",
        "title": "Electrophoresis animation",
        "question": "which clip is a labeled cartoon of molecules migrating through a gel matrix",
        "answer": "Cartoon of migrating molecules.",
    },
    {
        "id": "oa-vid-confocal",
        "file": "File:Fluorescent and confocal microscopes.ogv",
        "title": "Fluorescent and confocal microscopes",
        "question": "which clip shows a fluorescence microscope body with a laser scan head",
        "answer": "Confocal / fluorescence scope body.",
    },
    {
        "id": "oa-vid-milk",
        "file": "File:Milk under the microscope.webm",
        "title": "Milk under the microscope",
        "question": "which clip shows swirling droplets under a bright-field microscope",
        "answer": "Bright-field droplets (milk).",
    },
    {
        "id": "oa-vid-tem",
        "file": "File:Microscope Electronique à Transmission.ogv",
        "title": "Transmission electron microscope",
        "question": "which clip shows a tall transmission-electron column in a lab",
        "answer": "TEM column.",
    },
    {
        "id": "oa-vid-neutrophil",
        "file": "File:Neutrophil_chasing_bacteria.ogv",
        "title": "Neutrophil chasing bacteria",
        "question": "which clip shows a crawling cell chasing smaller rods across a field",
        "answer": "Crawling leukocyte chasing rods.",
    },
    {
        "id": "oa-vid-mitosis",
        "file": "File:Mitosis.ogv",
        "title": "Mitosis",
        "question": "which clip shows a nucleus splitting into two daughter nuclei",
        "answer": "Mitosis / nuclear split.",
    },
    {
        "id": "oa-vid-sem",
        "file": "File:Scanning_electron_microscope.ogv",
        "title": "Scanning electron microscope",
        "question": "which clip shows a scanning-electron chamber with a sample stub",
        "answer": "SEM chamber / stub.",
    },
    {
        "id": "oa-vid-bacteria",
        "file": "File:Bacteria_swimming.ogv",
        "title": "Bacteria swimming",
        "question": "which clip shows tiny rods darting in a wet-mount field",
        "answer": "Swimming rods in wet mount.",
    },
]

ABSTAIN_AUDIO = [
    {
        "id": "oa-aud-abs-01",
        "question": "speaker names a magenta Maltese-cross lipid droplet with a 5 μm scale bar",
    },
    {
        "id": "oa-aud-abs-02",
        "question": "talk that uniquely mentions a ring chromosome marked by a crimson chevron",
    },
]
ABSTAIN_VIDEO = [
    {
        "id": "oa-vid-abs-01",
        "question": "spinning-disk movie still with a tracked vesicle and a 2 μm scale bar",
    },
    {
        "id": "oa-vid-abs-02",
        "question": "anaglyph ribosome cartoon with unique offset missing from this gallery",
    },
]


def _url_suffix(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    ext = Path(path).suffix.lower()
    if ext in {".ogg", ".ogv", ".oga", ".mp3", ".wav", ".webm", ".mp4", ".mkv", ".flac"}:
        return ext
    return ".bin"


def _get(url: str, dest: Path, max_bytes: int = 80_000_000) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=90) as resp:
            n = 0
            with dest.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > max_bytes:
                        dest.unlink(missing_ok=True)
                        print(f"  skip oversize {url}", flush=True)
                        return False
                    fh.write(chunk)
        return dest.stat().st_size > 1024
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  fetch fail {type(exc).__name__}: {exc}", flush=True)
        dest.unlink(missing_ok=True)
        return False


def commons_file_url(title: str) -> str | None:
    qs = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "format": "json",
        }
    )
    api = "https://commons.wikimedia.org/w/api.php?" + qs
    req = urllib.request.Request(api, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    pages = (payload.get("query") or {}).get("pages") or {}
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if infos and infos[0].get("url"):
            return str(infos[0]["url"])
    return None


def _ffmpeg(*args: str) -> bool:
    bin_ = shutil.which("ffmpeg")
    if not bin_:
        return False
    r = subprocess.run([bin_, "-y", "-hide_banner", "-loglevel", "error", *args], check=False)
    return r.returncode == 0


def extract_wav(src: Path, dest: Path, seconds: float = 20.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return True
    return _ffmpeg("-i", str(src), "-t", str(seconds), "-ac", "1", "-ar", "16000", str(dest))


def extract_frames_ffmpeg(src: Path, dest: Path, max_frames: int = 4) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    pattern = dest / "k%02d.jpg"
    if not _ffmpeg("-i", str(src), "-vf", f"fps=1/{max(1, max_frames)}", "-frames:v", str(max_frames), str(pattern)):
        return []
    return sorted(dest.glob("k*.jpg"))[:max_frames]


def _maybe_whisper(path: Path, fallback: str) -> str:
    if os.environ.get("ABCITE_WHISPER", "0").lower() not in {"1", "true", "yes", "on"}:
        return fallback
    try:
        from abcite_core.audio_rag import AudioEncoder

        text = AudioEncoder().transcribe(path, fallback=fallback)
        return text or fallback
    except Exception:
        return fallback


def _gold_audio_row(cid: str, question: str, answer: str) -> dict:
    return {
        "id": cid,
        "question": question,
        "gold_audio_ids": [cid],
        "answer": answer,
        "transcript_bm25_should_fail": True,
        "expected_status": "answered",
        "must_listen": True,
        "oa": True,
    }


def _gold_video_row(cid: str, question: str, answer: str) -> dict:
    return {
        "id": cid,
        "question": question,
        "gold_video_ids": [cid],
        "answer": answer,
        "transcript_bm25_should_fail": True,
        "expected_status": "answered",
        "must_watch": True,
        "oa": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=16)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    AUD_OA.mkdir(parents=True, exist_ok=True)
    VID_OA.mkdir(parents=True, exist_ok=True)

    audio_rows: list[dict] = []
    audio_gold: list[dict] = []
    for i, cand in enumerate(AUDIO_CANDIDATES[: args.max]):
        url = commons_file_url(cand["file"])
        print(f"{cand['id']} url={url or 'missing'}", flush=True)
        if not url:
            continue
        raw = CACHE / f"{cand['id']}{_url_suffix(url)}"
        if not (raw.exists() or _get(url, raw)):
            continue
        wav = AUD_OA / f"{cand['id']}.wav"
        if not extract_wav(raw, wav):
            print(f"  wav extract fail {cand['id']}", flush=True)
            continue
        filler = FILLER[i % len(FILLER)]
        transcript = _maybe_whisper(wav, filler)
        rel = str(wav.relative_to(ROOT))
        audio_rows.append(
            {
                "audio_id": cand["id"],
                "path": rel,
                "transcript": transcript,
                "title": cand["title"],
                "source": "wikimedia-commons",
                "href": url,
                "synthetic": False,
            }
        )
        audio_gold.append(_gold_audio_row(cand["id"], cand["question"], cand["answer"]))

    video_rows: list[dict] = []
    video_gold: list[dict] = []
    n_vid = max(0, args.max - 0)
    for i, cand in enumerate(VIDEO_CANDIDATES[:n_vid]):
        url = commons_file_url(cand["file"])
        print(f"{cand['id']} url={url or 'missing'}", flush=True)
        if not url:
            continue
        raw = CACHE / f"{cand['id']}{_url_suffix(url)}"
        if not (raw.exists() or _get(url, raw)):
            continue
        from abcite_core.video_rag import extract_keyframes

        frames = extract_keyframes(raw, CACHE / cand["id"], max_frames=4)
        if not frames:
            frames = extract_frames_ffmpeg(raw, CACHE / cand["id"], max_frames=4)
        if not frames:
            print(f"  no keyframes {cand['id']}", flush=True)
            continue
        wav = VID_OA / f"{cand['id']}.wav"
        extract_wav(raw, wav)
        filler = FILLER[(i + 3) % len(FILLER)]
        transcript = _maybe_whisper(wav, filler) if wav.exists() else filler
        video_rows.append(
            {
                "video_id": cand["id"],
                "frames": [str(p.relative_to(ROOT)) for p in frames],
                "audio_path": str(wav.relative_to(ROOT)) if wav.exists() else None,
                "transcript": transcript,
                "title": cand["title"],
                "source": "wikimedia-commons",
                "href": url,
                "synthetic": False,
                "captions": [],
            }
        )
        video_gold.append(_gold_video_row(cand["id"], cand["question"], cand["answer"]))

    for abs_row in ABSTAIN_AUDIO:
        audio_gold.append(
            {
                "id": abs_row["id"],
                "question": abs_row["question"],
                "gold_audio_ids": [],
                "answer": "",
                "transcript_bm25_should_fail": True,
                "expected_status": "abstain",
                "must_listen": True,
                "oa": True,
            }
        )
    for abs_row in ABSTAIN_VIDEO:
        video_gold.append(
            {
                "id": abs_row["id"],
                "question": abs_row["question"],
                "gold_video_ids": [],
                "answer": "",
                "transcript_bm25_should_fail": True,
                "expected_status": "abstain",
                "must_watch": True,
                "oa": True,
            }
        )

    if audio_rows:
        (AUD_OA / "metadata.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in audio_rows), encoding="utf-8"
        )
        AUD_GOLD.write_text("".join(json.dumps(r) + "\n" for r in audio_gold), encoding="utf-8")
    if video_rows:
        (VID_OA / "metadata.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in video_rows), encoding="utf-8"
        )
        VID_GOLD.write_text("".join(json.dumps(r) + "\n" for r in video_gold), encoding="utf-8")
    print(
        f"oa-audio={len(audio_rows)} oa-video={len(video_rows)} "
        "(synthetic fixtures remain the CI default; eval with --oa)",
        flush=True,
    )


if __name__ == "__main__":
    main()
