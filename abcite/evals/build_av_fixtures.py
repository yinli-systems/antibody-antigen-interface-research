"""Write tiny committed wav + PNG keyframe fixtures (synthetic, labeled)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from abcite_core.audio_rag import write_sine_wav

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "evals" / "audio" / "fixture"
VID = ROOT / "evals" / "video" / "fixture"

AUDIO = [
    {
        "audio_id": "aud-01",
        "freq": 220.0,
        "title": "spike crown remark",
        "transcript": "The prefusion spike displays a three-fold crown at its distal tip.",
    },
    {
        "audio_id": "aud-02",
        "freq": 330.0,
        "title": "blot induction remark",
        "transcript": "Western blot lanes show a fifty kilodalton band only after induction.",
    },
    {
        "audio_id": "aud-03",
        "freq": 440.0,
        "title": "speckle remark",
        "transcript": "Confocal stacks reveal nuclear speckles after the emerald dye.",
    },
    {
        "audio_id": "aud-04",
        "freq": 550.0,
        "title": "orbit remark",
        "transcript": "The lecturer notes a gadolinium-enhancing orbit lesion on the left.",
    },
    {
        "audio_id": "aud-05",
        "freq": 660.0,
        "title": "forest remark",
        "transcript": "A forest plot diamond sits below the dashed null line.",
    },
    {
        "audio_id": "aud-06",
        "freq": 90.0,
        "title": "room tone",
        "transcript": "Room tone only. No scientific claim is spoken.",
    },
]

VIDEO = [
    {
        "video_id": "vid-01",
        "freq": 240.0,
        "title": "heatmap to volcano",
        "transcript": "The lecture now jumps to a second slide without naming colors.",
        "frames": [
            ("k0", (180, 30, 30), "grid"),
            ("k1", (30, 40, 170), "scatter"),
        ],
    },
    {
        "video_id": "vid-02",
        "freq": 360.0,
        "title": "gel then bands",
        "transcript": "Next the speaker zooms the lower half of the same membrane.",
        "frames": [
            ("k0", (20, 20, 20), "lanes"),
            ("k1", (40, 40, 40), "bands"),
        ],
    },
    {
        "video_id": "vid-03",
        "freq": 480.0,
        "title": "foci then merge",
        "transcript": "The second still is the merged channel of the same field.",
        "transcript_extra": "",
        "frames": [
            ("k0", (20, 140, 40), "dots"),
            ("k1", (30, 30, 120), "ovals"),
        ],
    },
    {
        "video_id": "vid-04",
        "freq": 600.0,
        "title": "orbit then pointer",
        "transcript": "Follow-up imaging is shown without repeating the anatomy name.",
        "frames": [
            ("k0", (90, 90, 90), "slice"),
            ("k1", (120, 100, 40), "pointer"),
        ],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main() -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    audio_rows = []
    for row in AUDIO:
        rel = f"evals/audio/fixture/{row['audio_id']}.wav"
        write_sine_wav(ROOT / rel, row["freq"])
        audio_rows.append(
            {
                "audio_id": row["audio_id"],
                "path": rel,
                "transcript": row["transcript"],
                "title": row["title"],
                "source": "synthetic-fixture",
                "synthetic": True,
            }
        )
    (AUD / "metadata.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in audio_rows), encoding="utf-8"
    )
    video_rows = []
    for row in VIDEO:
        frames = []
        for name, rgb, kind in row["frames"]:
            rel = f"evals/video/fixture/{row['video_id']}_{name}.png"
            _frame(ROOT / rel, rgb, kind)
            frames.append(rel)
        arel = f"evals/video/fixture/{row['video_id']}.wav"
        write_sine_wav(ROOT / arel, row["freq"])
        video_rows.append(
            {
                "video_id": row["video_id"],
                "frames": frames,
                "audio_path": arel,
                "transcript": row["transcript"],
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
