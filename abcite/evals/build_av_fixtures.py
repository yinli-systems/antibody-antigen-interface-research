"""Write labeled synthetic wav + PNG keyframe fixtures (honest, larger than toy).

Transcripts are generic lecture filler on purpose: queries ask about sound or
scene, so transcript BM25 should fail and the embedding column can win.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from abcite_core.audio_rag import write_fixture_wav

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "evals" / "audio" / "fixture"
VID = ROOT / "evals" / "video" / "fixture"

# kind / freqs chosen so spectra land in different wav_embed bands.
AUDIO = [
    {"audio_id": "aud-01", "kind": "sine", "freq": 180.0, "title": "low drone"},
    {"audio_id": "aud-02", "kind": "sine", "freq": 850.0, "title": "mid hum"},
    {"audio_id": "aud-03", "kind": "sine", "freq": 1900.0, "title": "midhigh tone"},
    {"audio_id": "aud-04", "kind": "sine", "freq": 3200.0, "title": "piercing beep"},
    {"audio_id": "aud-05", "kind": "sine", "freq": 5100.0, "title": "bright whistle"},
    {"audio_id": "aud-06", "kind": "sine", "freq": 7200.0, "title": "shrill beep"},
    {"audio_id": "aud-07", "kind": "chirp_up", "freq": 200.0, "title": "rising chirp"},
    {"audio_id": "aud-08", "kind": "chirp_down", "freq": 4000.0, "title": "falling chirp"},
    {"audio_id": "aud-09", "kind": "pulse", "freq": 8.0, "title": "slow click train"},
    {"audio_id": "aud-10", "kind": "pulse", "freq": 22.0, "title": "rapid pulse"},
    {"audio_id": "aud-11", "kind": "hiss", "freq": 0.0, "title": "broadband hiss"},
    {"audio_id": "aud-12", "kind": "roar", "freq": 0.0, "title": "dull roar"},
    {"audio_id": "aud-13", "kind": "two_tone", "freq": 280.0, "freq_b": 2600.0, "title": "two-tone interval"},
    {"audio_id": "aud-14", "kind": "two_tone", "freq": 900.0, "freq_b": 6200.0, "title": "split interval"},
    {"audio_id": "aud-15", "kind": "hush", "freq": 0.0, "title": "room hush"},
    {"audio_id": "aud-16", "kind": "tremolo", "freq": 440.0, "title": "wobble tone"},
    {"audio_id": "aud-17", "kind": "sine", "freq": 420.0, "title": "low-mid drone"},
    {"audio_id": "aud-18", "kind": "sine", "freq": 1400.0, "title": "narrow hum"},
    {"audio_id": "aud-19", "kind": "pulse", "freq": 5.0, "title": "sparse clicks"},
    {"audio_id": "aud-20", "kind": "hiss", "freq": 0.0, "title": "soft hiss decoy"},
    {"audio_id": "aud-21", "kind": "roar", "freq": 0.0, "title": "roar decoy"},
    {"audio_id": "aud-22", "kind": "sine", "freq": 6000.0, "title": "whistle decoy"},
    {"audio_id": "aud-23", "kind": "chirp_up", "freq": 300.0, "title": "sweep decoy"},
    {"audio_id": "aud-24", "kind": "hush", "freq": 0.0, "title": "hush decoy"},
]

# Shared filler — no acoustic or scene cue tokens.
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
    "A phone is silenced in the front row.",
    "The host thanks the previous presenter.",
    "No one speaks for several seconds.",
    "The wireless pack warning is not described.",
    "Attendees take notes without reading aloud.",
    "The moderator repeats that questions come later.",
    "A laptop lid shuts near the aisle.",
    "The camera operator zooms without comment.",
    "Someone requests a short break after this item.",
    "The clock on the wall is not mentioned.",
    "A bottle cap twists at the back table.",
    "The session chair nods and the feed continues.",
]

VIDEO = [
    {
        "video_id": "vid-01",
        "freq": 240.0,
        "title": "scarlet grid to blue scatter",
        "frames": [("k0", (180, 30, 30), "grid"), ("k1", (30, 40, 170), "scatter")],
    },
    {
        "video_id": "vid-02",
        "freq": 360.0,
        "title": "dark gel then bands",
        "frames": [("k0", (20, 20, 20), "lanes"), ("k1", (40, 40, 40), "bands")],
    },
    {
        "video_id": "vid-03",
        "freq": 480.0,
        "title": "emerald foci then indigo oval",
        "frames": [("k0", (20, 140, 40), "dots"), ("k1", (30, 30, 120), "ovals")],
    },
    {
        "video_id": "vid-04",
        "freq": 600.0,
        "title": "grainy slice then gold pointer",
        "frames": [("k0", (90, 90, 90), "slice"), ("k1", (120, 100, 40), "pointer")],
    },
    {
        "video_id": "vid-05",
        "freq": 200.0,
        "title": "orange heatmap then teal wave",
        "frames": [("k0", (200, 90, 20), "heatmap"), ("k1", (20, 140, 130), "wave")],
    },
    {
        "video_id": "vid-06",
        "freq": 700.0,
        "title": "magenta blob then cyan rings",
        "frames": [("k0", (180, 30, 150), "blob"), ("k1", (20, 180, 190), "rings")],
    },
    {
        "video_id": "vid-07",
        "freq": 310.0,
        "title": "yellow bars then navy scatter",
        "frames": [("k0", (210, 200, 30), "bars"), ("k1", (20, 30, 90), "scatter")],
    },
    {
        "video_id": "vid-08",
        "freq": 520.0,
        "title": "crimson grid then gold bars",
        "frames": [("k0", (160, 20, 20), "grid"), ("k1", (190, 160, 30), "bars")],
    },
    {
        "video_id": "vid-09",
        "freq": 180.0,
        "title": "teal wave decoy",
        "frames": [("k0", (15, 130, 120), "wave"), ("k1", (20, 120, 110), "wave")],
    },
    {
        "video_id": "vid-10",
        "freq": 880.0,
        "title": "dark lanes decoy",
        "frames": [("k0", (18, 18, 18), "lanes"), ("k1", (22, 22, 22), "lanes")],
    },
    {
        "video_id": "vid-11",
        "freq": 150.0,
        "title": "cyan rings decoy",
        "frames": [("k0", (10, 170, 180), "rings"), ("k1", (15, 160, 170), "rings")],
    },
    {
        "video_id": "vid-12",
        "freq": 990.0,
        "title": "orange heatmap decoy",
        "frames": [("k0", (190, 80, 15), "heatmap"), ("k1", (200, 95, 25), "heatmap")],
    },
    {
        "video_id": "vid-13",
        "freq": 410.0,
        "title": "emerald dots decoy",
        "frames": [("k0", (25, 150, 45), "dots"), ("k1", (30, 145, 40), "dots")],
    },
    {
        "video_id": "vid-14",
        "freq": 275.0,
        "title": "gold pointer decoy",
        "frames": [("k0", (130, 110, 35), "pointer"), ("k1", (140, 115, 40), "pointer")],
    },
    {
        "video_id": "vid-15",
        "freq": 630.0,
        "title": "indigo oval decoy",
        "frames": [("k0", (35, 35, 130), "ovals"), ("k1", (40, 40, 140), "ovals")],
    },
    {
        "video_id": "vid-16",
        "freq": 70.0,
        "title": "yellow bars decoy",
        "frames": [("k0", (200, 190, 25), "bars"), ("k1", (205, 195, 30), "bars")],
    },
]


def _frame(path: Path, rgb: tuple[int, int, int], kind: str) -> None:
    im = Image.new("RGB", (96, 72), rgb)
    dr = ImageDraw.Draw(im)
    if kind == "grid":
        for x in range(8, 90, 10):
            for y in range(8, 66, 10):
                dr.rectangle([x, y, x + 6, y + 6], fill=(220, 80, 40))
    elif kind == "scatter":
        for x, y in ((12, 50), (28, 30), (50, 18), (70, 40), (82, 22)):
            dr.ellipse([x, y, x + 5, y + 5], fill=(180, 200, 255))
    elif kind == "lanes":
        for x in (10, 22, 34, 46, 58, 70, 82):
            dr.rectangle([x, 8, x + 6, 64], fill=(10, 10, 10))
        dr.rectangle([4, 8, 8, 64], fill=(200, 40, 180))
    elif kind == "bands":
        dr.rectangle([10, 28, 86, 34], fill=(240, 240, 240))
        dr.rectangle([10, 44, 86, 52], fill=(230, 230, 230))
    elif kind == "dots":
        for x, y in ((20, 20), (40, 18), (55, 30), (70, 22)):
            dr.ellipse([x, y, x + 7, y + 7], fill=(40, 230, 80))
    elif kind == "ovals":
        dr.ellipse([28, 16, 68, 56], outline=(80, 80, 255), width=3)
        dr.ellipse([36, 24, 60, 48], fill=(20, 200, 60))
    elif kind == "slice":
        dr.ellipse([18, 8, 78, 64], fill=(70, 70, 70))
        dr.ellipse([40, 24, 56, 40], fill=(140, 40, 40))
    elif kind == "pointer":
        dr.ellipse([18, 8, 78, 64], fill=(80, 80, 70))
        dr.polygon([(70, 18), (86, 10), (78, 28)], fill=(220, 180, 40))
    elif kind == "heatmap":
        for i, col in enumerate(((200, 40, 10), (220, 90, 20), (240, 140, 40))):
            dr.rectangle([8 + i * 28, 10, 32 + i * 28, 62], fill=col)
    elif kind == "wave":
        pts = [(x, int(36 + 18 * __import__("math").sin(x / 7.0))) for x in range(4, 92, 4)]
        dr.line(pts, fill=(180, 255, 230), width=3)
    elif kind == "blob":
        dr.ellipse([22, 16, 74, 58], fill=(230, 60, 200))
    elif kind == "rings":
        dr.ellipse([16, 8, 80, 64], outline=(20, 230, 230), width=3)
        dr.ellipse([28, 18, 68, 54], outline=(40, 200, 220), width=2)
    elif kind == "bars":
        for x, h in ((12, 40), (28, 55), (44, 28), (60, 48), (76, 36)):
            dr.rectangle([x, 64 - h, x + 10, 64], fill=(240, 220, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main() -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    audio_rows = []
    for i, row in enumerate(AUDIO):
        rel = f"evals/audio/fixture/{row['audio_id']}.wav"
        write_fixture_wav(
            ROOT / rel,
            row["kind"],
            freq_hz=float(row.get("freq") or 0.0),
            freq_b=float(row.get("freq_b") or 0.0),
        )
        audio_rows.append(
            {
                "audio_id": row["audio_id"],
                "path": rel,
                "transcript": FILLER[i % len(FILLER)],
                "title": row["title"],
                "source": "synthetic-fixture",
                "synthetic": True,
            }
        )
    (AUD / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in audio_rows), encoding="utf-8"
    )
    video_rows = []
    for i, row in enumerate(VIDEO):
        frames = []
        for name, rgb, kind in row["frames"]:
            rel = f"evals/video/fixture/{row['video_id']}_{name}.png"
            _frame(ROOT / rel, rgb, kind)
            frames.append(rel)
        arel = f"evals/video/fixture/{row['video_id']}.wav"
        write_fixture_wav(ROOT / arel, "sine", freq_hz=row["freq"])
        video_rows.append(
            {
                "video_id": row["video_id"],
                "frames": frames,
                "audio_path": arel,
                "transcript": FILLER[(i + 3) % len(FILLER)],
                "title": row["title"],
                "source": "synthetic-fixture",
                "synthetic": True,
                "captions": [],
            }
        )
    (VID / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in video_rows), encoding="utf-8"
    )
    print(f"audio={len(audio_rows)} video={len(video_rows)}")


if __name__ == "__main__":
    main()
