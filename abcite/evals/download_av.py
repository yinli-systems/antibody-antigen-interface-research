"""Download short CC science clips for OA audio/video RAG.

CI fixtures stay synthetic (`evals/build_av_fixtures.py`). This script is
best-effort Wikimedia Commons + NASA images (public domain). Never required
for pytest.

Usage:
  python evals/download_av.py --max 64
  ABCITE_WHISPER=1 python evals/download_av.py   # ASR transcripts when ffmpeg+whisper exist
  python evals/run_audio_qa.py --oa
  python evals/run_video_qa.py --oa

Queries describe **sound or on-screen appearance**, not Wikipedia titles.
Transcripts default to generic lecture filler so BM25 should fail. Keep
ABCITE_WHISPER=0 for the OA BM25-should-fail table. Do not mix OA receipts
with the synthetic wav-hash table. Licenses: evals/oa_av_LICENSE.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import subprocess
import time
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
# Commons titles verified via imageinfo (2026-09-10). Missing 404s dropped.
AUDIO_CANDIDATES = [
    {"id": "oa-aud-heartbeat", "file": "File:Heartbeat.ogg", "license": "CC BY 3.0", "title": "Heartbeat recording", "question": "which clip is a repeating low thud pulse like a cardiac cycle", "answer": "A heartbeat / lub-dub pulse train."},
    {"id": "oa-aud-geiger", "file": "File:Geiger_counter.ogg", "license": "CC BY 3.0", "title": "Geiger counter clicks", "question": "sparse sharp click train, irregular, not a sine drone", "answer": "Geiger-like irregular clicks."},
    {"id": "oa-aud-noise", "file": "File:White_noise.ogg", "license": "Public domain", "title": "White noise", "question": "broadband hiss, not tonal", "answer": "White-ish hiss."},
    {"id": "oa-aud-thunder", "file": "File:Thunder.ogg", "license": "CC BY 3.0", "title": "Thunder", "question": "distant rumble burst, not a hiss", "answer": "Thunder rumble."},
    {"id": "oa-aud-doppler", "file": "File:Fetal doppler recording heartbeat and movement.ogg", "license": "CC BY-SA 3.0", "title": "Fetal Doppler", "question": "whoosh pulse from a handheld fetal doppler, not a geiger click", "answer": "Fetal Doppler whoosh."},
    {"id": "oa-aud-korotkoff", "file": "File:Nl-Korotkoff-tonen.ogg", "license": "CC BY-SA 3.0", "title": "Korotkoff sounds", "question": "soft cuff thumps used to read blood pressure", "answer": "Korotkoff thumps."},
    {"id": "oa-aud-cricket", "file": "File:Cricket.ogg", "license": "CC BY-SA 3.0", "title": "Cricket chirp", "question": "rhythmic insect chirp train, not broadband hiss", "answer": "Cricket chirps."},
    {"id": "oa-aud-siren", "file": "File:Siren.ogg", "license": "Public domain", "title": "Police siren", "question": "wailing two-tone alarm glide, not a click train", "answer": "Two-tone siren wail."},
    {"id": "oa-aud-a440", "file": "File:440 Hz.ogg", "license": "Public domain", "title": "A440 tone", "question": "steady pure tone, not a pulse and not a hiss", "answer": "Steady A440-like tone."},
    {"id": "oa-aud-cat", "file": "File:Cat.ogg", "license": "CC BY 3.0", "title": "Cat", "question": "short feline meow, not a bark and not a chirp", "answer": "A cat meow."},
    {"id": "oa-aud-bark", "file": "File:Barking.ogg", "license": "CC BY-SA 4.0", "title": "Barking", "question": "sharp dog bark bursts, not a meow", "answer": "Dog barks."},
    {"id": "oa-aud-frog", "file": "File:Frog.ogg", "license": "CC BY-SA 3.0", "title": "Frog", "question": "guttural amphibian croak, not an insect chirp", "answer": "Frog croak."},
    {"id": "oa-aud-applause", "file": "File:Applause.ogg", "license": "CC BY-SA 3.0", "title": "Applause", "question": "many hands clapping in overlapping bursts", "answer": "Applause."},
    {"id": "oa-aud-bells", "file": "File:Church bells.ogg", "license": "Public domain", "title": "Church bells", "question": "pealing metal bells in a tower, not a siren", "answer": "Church bells."},
    {"id": "oa-aud-fire", "file": "File:Fire.ogg", "license": "Public domain", "title": "Fire", "question": "wood-fire crackle and pop, not rain", "answer": "Fire crackle."},
    {"id": "oa-aud-laugh", "file": "File:Laughter.ogg", "license": "Public domain", "title": "Laughter", "question": "human laugh bursts, not a sneeze", "answer": "Laughter."},
    {"id": "oa-aud-rain", "file": "File:Rain.ogg", "license": "Public domain", "title": "Rain", "question": "steady droplet patter on a surface, not a cascade roar", "answer": "Rain patter."},
    {"id": "oa-aud-sneeze", "file": "File:Sneeze.ogg", "license": "CC0", "title": "Sneeze", "question": "one abrupt human sneeze, not laughter", "answer": "A sneeze."},
    {"id": "oa-aud-banjo", "file": "File:Banjo.ogg", "license": "CC BY-SA 3.0", "title": "Banjo", "question": "plucked twangy strings, not a piano", "answer": "Banjo twang."},
    {"id": "oa-aud-clock", "file": "File:Clock ticking.ogg", "license": "Public domain", "title": "Clock ticking", "question": "regular mechanical tick-tock, not a drum hit", "answer": "Clock ticks."},
    {"id": "oa-aud-flute", "file": "File:Flute.ogg", "license": "Public domain", "title": "Flute", "question": "airy woodwind melody, not a harmonica reed chord", "answer": "Flute."},
    {"id": "oa-aud-harmonica", "file": "File:Harmonica.ogg", "license": "Public domain", "title": "Harmonica", "question": "blown reed chords, not a flute", "answer": "Harmonica."},
    {"id": "oa-aud-piano", "file": "File:Piano.ogg", "license": "CC BY-SA 3.0", "title": "Piano", "question": "struck piano keys, not plucked strings", "answer": "Piano."},
    {"id": "oa-aud-brown", "file": "File:Brown noise.ogg", "license": "CC BY-SA 3.0", "title": "Brown noise", "question": "low dull rumble noise that stays steady, not a thunder burst", "answer": "Brown noise."},
    {"id": "oa-aud-pink", "file": "File:Pink noise.ogg", "license": "CC BY-SA 3.0", "title": "Pink noise", "question": "mid-heavy rustle distinct from white hiss and brown rumble", "answer": "Pink noise."},
    {"id": "oa-aud-morse", "file": "File:Morse code.ogg", "license": "Public domain", "title": "Morse code", "question": "short-long beep code, not a steady sine", "answer": "Morse beeps."},
    {"id": "oa-aud-waterfall", "file": "File:Waterfall.ogg", "license": "CC BY-SA 4.0", "title": "Waterfall", "question": "continuous cascade roar, not droplet patter", "answer": "Waterfall roar."},
    {"id": "oa-aud-kick", "file": "File:Bass drum.ogg", "license": "CC BY-SA 3.0", "title": "Bass drum", "question": "single deep drum hit, not a clock tick", "answer": "Bass-drum hit."},
    {"id": "oa-aud-cicada", "file": "File:Cicada.ogg", "license": "CC BY-SA 3.0", "title": "Cicada", "question": "high buzzing insect drone, not a croak", "answer": "Cicada buzz."},
    {"id": "oa-aud-register", "file": "File:Cash register.ogg", "license": "Public domain", "title": "Cash register", "question": "bell ding plus a drawer scrape, not church bells", "answer": "Cash-register ding."},
]

def _nasa_small(nasa_id: str) -> str:
    enc = urllib.parse.quote(nasa_id)
    return f"https://images-assets.nasa.gov/video/{enc}/{enc}~small.mp4"


VIDEO_CANDIDATES = [
    {"id": "oa-vid-gelbox", "file": "File:Conducting-Polymer-Electrodes-for-Gel-Electrophoresis-pone.0089416.s001.ogv", "license": "CC BY 4.0", "title": "Conducting polymer gel electrophoresis", "question": "which clip shows a small gel box with running sample lanes on a bench", "answer": "Benchtop gel box with lanes."},
    {"id": "oa-vid-fige", "file": "File:Increase-in-local-protein-concentration-by-field-inversion-gel-electrophoresis-1477-5956-5-18-S2.ogv", "license": "CC BY 2.0", "title": "Field-inversion gel bands", "question": "which clip shows glowing nucleic-acid bands drifting in a dark slab", "answer": "Glowing bands in a dark gel slab."},
    {"id": "oa-vid-anim", "file": "File:Electrophoresis explained - biology animation.webm", "license": "CC BY 3.0", "title": "Electrophoresis animation", "question": "which clip is a labeled cartoon of molecules migrating through a gel matrix", "answer": "Cartoon of migrating molecules."},
    {"id": "oa-vid-confocal", "file": "File:Fluorescent and confocal microscopes.ogv", "license": "CC BY-SA 4.0", "title": "Fluorescent and confocal microscopes", "question": "which clip shows a fluorescence microscope body with a laser scan head", "answer": "Confocal / fluorescence scope body."},
    {"id": "oa-vid-milk", "file": "File:Milk under the microscope.webm", "license": "CC BY 4.0", "title": "Milk under the microscope", "question": "which clip shows swirling droplets under a bright-field microscope", "answer": "Bright-field droplets (milk)."},
    {"id": "oa-vid-tem", "file": "File:Microscope Electronique à Transmission.ogv", "license": "CC BY-SA 4.0", "title": "Transmission electron microscope", "question": "which clip shows a tall transmission-electron column in a lab", "answer": "TEM column."},
    {"id": "oa-vid-neutrophil", "file": "File:Monomeric-and-Dimeric-CXCL8-Are-Both-Essential-for-In-Vivo-Neutrophil-Recruitment-pone.0011754.s004.ogv", "license": "CC BY 2.5", "title": "In-vivo neutrophil recruitment", "question": "which clip shows round white cells tumbling along a vessel wall in living tissue", "answer": "Neutrophils rolling in a vessel."},
    {"id": "oa-vid-mitosis", "file": "File:Global-Analysis-of-CPEBs-Reveals-Sequential-and-Non-Redundant-Functions-in-Mitotic-Cell-Cycle-pone.0138794.s004.ogv", "license": "CC BY 4.0", "title": "Mitotic cell-cycle movie", "question": "which clip shows a round cell with chromosomes lining up then pulling into two masses", "answer": "Mitotic chromosome split."},
    {"id": "oa-vid-polarize", "file": "File:A-Mathematical-Model-for-Neutrophil-Gradient-Sensing-and-Polarization-pcbi.0030036.sv002.ogv", "license": "CC BY 2.5", "title": "Neutrophil polarization", "question": "which clip shows one cell flattening and sending a single leading edge toward one pole", "answer": "A cell polarizing to one pole."},
    {"id": "oa-vid-mitosis2", "file": "File:Global-Analysis-of-CPEBs-Reveals-Sequential-and-Non-Redundant-Functions-in-Mitotic-Cell-Cycle-pone.0138794.s003.ogv", "license": "CC BY 4.0", "title": "Second mitotic time-lapse", "question": "which clip is a time-lapse of a cell rounding up before the chromosomes segregate", "answer": "Cell rounding before anaphase."},
    {"id": "oa-vid-amoeba", "file": "File:Amoeba proteus.ogv", "license": "CC BY-SA 3.0", "title": "Amoeba proteus", "question": "which clip shows a shapeless blob crawling by pushing out pseudopods", "answer": "Amoeba crawling."},
    {"id": "oa-vid-cyclosis", "file": "File:Cyclosis.ogv", "license": "CC BY-SA 3.0", "title": "Cyclosis", "question": "which clip shows green chloroplasts streaming in a loop inside a plant cell", "answer": "Chloroplast streaming (cyclosis)."},
    {"id": "oa-vid-vessels", "file": "File:3-D-Visualization-and-Quantitation-of-Microvessels-in-Transparent-Human-Colorectal-Carcinoma-pone.0081857.s003.ogv", "license": "CC BY 2.5", "title": "3-D microvessels", "question": "which clip is a 3-D fly-through of a dense stained vessel network in tissue", "answer": "3-D stained vessels."},
    {"id": "oa-vid-microfluidic", "file": "File:A-Microfluidic-Platform-for-Correlative-Live-Cell-and-Super-Resolution-Microscopy-pone.0115512.s002.ogv", "license": "CC0", "title": "Microfluidic live-cell", "question": "which clip shows cells drifting through a narrow microfluidic channel", "answer": "Cells in a microfluidic channel."},
    {"id": "oa-vid-spheroid", "file": "File:A-High-Content-Small-Molecule-Screen-Identifies-Sensitivity-of-Glioblastoma-Stem-Cells-to-pone.0077053.s011.ogv", "license": "CC BY 2.5", "title": "Glioblastoma spheroids", "question": "which clip is a multi-well time-lapse of round tumor spheroids", "answer": "Tumor spheroids in wells."},
    {"id": "oa-vid-filter", "file": "File:A-Microfiltration-Device-for-Urogenital-Schistosomiasis-Diagnostics-pone.0154640.s003.ogv", "license": "CC BY 4.0", "title": "Schistosomiasis filter", "question": "which clip shows parasite eggs trapped on a diagnostic filter mesh", "answer": "Eggs on a filter mesh."},
    {"id": "oa-vid-ozone", "file": "File:2023 Ozone Hole Update.webm", "license": "Public domain", "title": "Ozone hole update", "question": "which clip is a false-color polar map of a seasonal ozone-layer gap", "answer": "Polar ozone-hole map."},
    {"id": "oa-vid-cfd", "file": "File:Advanced Simulation Library - Multicomponent flow video.ogv", "license": "GFDL", "title": "Multicomponent flow CFD", "question": "which clip is a colored two-fluid mixing simulation inside a box", "answer": "Two-fluid CFD mix."},
    {"id": "oa-vid-chestnut", "file": "File:Chestnut with maggot CT.ogv", "license": "CC BY-SA 3.0", "title": "Chestnut CT", "question": "which clip is grayscale CT slices of a nut with a curved larva tunnel", "answer": "Chestnut CT with larva."},
    {
        "id": "oa-vid-earth",
        "file": "GSFC_20140421_EarthOrbit_m11525",
        "url": _nasa_small("GSFC_20140421_EarthOrbit_m11525"),
        "source": "nasa",
        "license": "NASA public domain",
        "title": "Earth from orbit 2013",
        "question": "which clip shows continents and swirling clouds on a blue planet from space",
        "answer": "Earth from orbit.",
    },
    {
        "id": "oa-vid-launch",
        "file": "GSFC_20121101_GPM_m11129_Launch",
        "url": _nasa_small("GSFC_20121101_GPM_m11129_Launch"),
        "source": "nasa",
        "license": "NASA public domain",
        "title": "GPM launch animation",
        "question": "which clip is a CGI rocket leaving a pad then a satellite unfolding in orbit",
        "answer": "Launch-and-deploy animation.",
    },
    {
        "id": "oa-vid-solar",
        "file": "GSFC_20160426_SDO_m12224_SolarFlare",
        "url": _nasa_small("GSFC_20160426_SDO_m12224_SolarFlare"),
        "source": "nasa",
        "license": "NASA public domain",
        "title": "SDO solar flare",
        "question": "which clip shows a bright looping eruption on the solar disk",
        "answer": "Solar-flare loop.",
    },
    {
        "id": "oa-vid-jupiter",
        "file": "GSFC_20180313_Jupiter_m12878_GreatRedSpot",
        "url": _nasa_small("GSFC_20180313_Jupiter_m12878_GreatRedSpot"),
        "source": "nasa",
        "license": "NASA public domain",
        "title": "Jupiter Great Red Spot",
        "question": "which clip shows a banded gas giant with an oval rust-colored storm",
        "answer": "Jupiter red-spot bands.",
    },
    {
        "id": "oa-vid-aurora",
        "file": "GSFC_20180314_Aurora_m12865_Steve",
        "url": _nasa_small("GSFC_20180314_Aurora_m12865_Steve"),
        "source": "nasa",
        "license": "NASA public domain",
        "title": "Aurora STEVE",
        "question": "which clip shows a purple-green night-sky curtain over a dark horizon",
        "answer": "Aurora curtain.",
    },
    {
        "id": "oa-vid-rocket",
        "file": "KSC-20240525-MH-RKL01-0001-Rocket_Lab_PREFIRE_1_Launch_1080p-M6988",
        "url": _nasa_small("KSC-20240525-MH-RKL01-0001-Rocket_Lab_PREFIRE_1_Launch_1080p-M6988"),
        "source": "nasa",
        "license": "NASA public domain",
        "title": "Rocket Lab PREFIRE launch",
        "question": "which clip shows a real slender rocket lifting off a coastal pad, not a globe",
        "answer": "Coastal rocket liftoff.",
    },
    {
        "id": "oa-vid-volcano",
        "file": "GSFC_20141113_Volcanoes_m10183_Degassing",
        "url": _nasa_small("GSFC_20141113_Volcanoes_m10183_Degassing"),
        "source": "nasa",
        "license": "NASA public domain",
        "title": "Volcano degassing schematic",
        "question": "which clip is a schematic of volcanic plumes injecting aerosols into clouds",
        "answer": "Volcano-aerosol schematic.",
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
        with urllib.request.urlopen(req, context=CTX, timeout=180) as resp:
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
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                time.sleep(2.0)
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None
    else:
        return None
    pages = (payload.get("query") or {}).get("pages") or {}
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if infos and infos[0].get("url"):
            return str(infos[0]["url"])
    return None


def filepath_url(title: str) -> str:
    name = title.removeprefix("File:")
    return "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(name)


def resolve_url(cand: dict) -> str | None:
    if cand.get("url"):
        return str(cand["url"])
    title = cand["file"]
    url = commons_file_url(title)
    if url:
        return url
    print(f"  imageinfo miss; Special:FilePath {title}", flush=True)
    return filepath_url(title)


def source_of(cand: dict) -> str:
    if cand.get("source"):
        return str(cand["source"])
    if "nasa.gov" in str(cand.get("url") or ""):
        return "nasa"
    return "wikimedia-commons"


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
    ap.add_argument("--max", type=int, default=64)
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--video-only", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    AUD_OA.mkdir(parents=True, exist_ok=True)
    VID_OA.mkdir(parents=True, exist_ok=True)

    audio_rows: list[dict] = []
    audio_gold: list[dict] = []
    if not args.video_only:
        for i, cand in enumerate(AUDIO_CANDIDATES[: args.max]):
            url = resolve_url(cand)
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
                    "source": source_of(cand),
                    "license": cand.get("license", ""),
                    "href": url,
                    "synthetic": False,
                }
            )
            audio_gold.append(_gold_audio_row(cand["id"], cand["question"], cand["answer"]))
            time.sleep(0.25)

    video_rows: list[dict] = []
    video_gold: list[dict] = []
    if not args.audio_only:
        for i, cand in enumerate(VIDEO_CANDIDATES[: args.max]):
            url = resolve_url(cand)
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
                    "source": source_of(cand),
                    "license": cand.get("license", ""),
                    "href": url,
                    "synthetic": False,
                    "captions": [],
                }
            )
            video_gold.append(_gold_video_row(cand["id"], cand["question"], cand["answer"]))
            time.sleep(0.25)

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
