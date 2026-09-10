"""Video RAG: keyframes (CLIP/ColPali) + Whisper audio track + transcript BM25.

A video record is a small directory: keyframe images + optional wav.
CI fixtures are synthetic (labeled). Download script can add CC science clips.
Fusion is **within-video only**. Do not RRF frames into caption BM25 for
text-only or visual_qa protocols.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from abcite_core.audio_rag import AudioEncoder
from abcite_core.metrics import ndcg_at_k
from abcite_core.models import AnswerStatus, Citation, Modality, QueryResponse
from abcite_core.visual_qa import caption_keyword_overlap
from abcite_core.vision import DIM, hashing_vector

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold" / "video_qa.jsonl"
FIX_META = ROOT / "evals" / "video" / "fixture" / "metadata.jsonl"
CACHE_META = ROOT / "evals" / "video" / "cache" / "metadata.jsonl"


@dataclass
class VideoRec:
    video_id: str
    frames: list[Path]
    audio_path: Path | None
    transcript: str
    title: str
    source: str
    href: str = ""
    synthetic: bool = True
    captions: list[str] = field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9_\-]+", text.lower())


# Color / shape words that describe the frame, not the lecture transcript.
SCENE_COLOR = {
    "scarlet": (0.95, 0.15, 0.1),
    "crimson": (0.9, 0.1, 0.1),
    "red": (0.9, 0.12, 0.1),
    "heatmap": (0.85, 0.2, 0.12),
    "blue": (0.12, 0.18, 0.9),
    "indigo": (0.15, 0.15, 0.75),
    "navy": (0.08, 0.12, 0.7),
    "volcano": (0.12, 0.2, 0.85),
    "emerald": (0.12, 0.85, 0.2),
    "green": (0.15, 0.8, 0.18),
    "teal": (0.1, 0.65, 0.55),
    "gold": (0.85, 0.7, 0.12),
    "yellow": (0.9, 0.85, 0.15),
    "magenta": (0.85, 0.15, 0.75),
    "cyan": (0.1, 0.8, 0.85),
    "orange": (0.9, 0.45, 0.08),
    "grainy": (0.4, 0.4, 0.4),
    "dark": (0.12, 0.12, 0.12),
}

SCENE_SHAPE = {
    "grid": 0,
    "heatmap": 0,
    "scatter": 1,
    "volcano": 1,
    "lanes": 2,
    "rungs": 2,
    "gel": 2,
    "bands": 3,
    "dots": 4,
    "foci": 4,
    "speckles": 4,
    "oval": 5,
    "overlay": 5,
    "pointer": 6,
    "arrow": 6,
    "slice": 7,
    "orbit": 7,
    "rings": 5,
    "wave": 1,
    "bars": 3,
    "blob": 4,
}


def query_scene_cues(text: str) -> list[str]:
    blob = text.lower()
    keys = list(SCENE_COLOR) + list(SCENE_SHAPE)
    return [k for k in keys if k in blob]


def _hue_bin(rgb: np.ndarray) -> int:
    r, g, b = (float(x) for x in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 0.08:
        return 8  # gray
    if mx == r:
        h = (g - b) / (mx - mn)
    elif mx == g:
        h = 2.0 + (b - r) / (mx - mn)
    else:
        h = 4.0 + (r - g) / (mx - mn)
    h = (h / 6.0) % 1.0
    return min(7, int(h * 8))


def scene_image_embed(path: Path, dim: int = DIM) -> np.ndarray:
    """Hue histogram + coarse shape slots. Shared with scene_text_embed for CI."""
    from PIL import Image

    im = Image.open(path).convert("RGB").resize((64, 48))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    pixels = arr.reshape(-1, 3)
    hue = np.zeros(9, dtype=np.float32)
    for pix in pixels[::4]:
        hue[_hue_bin(pix)] += 1.0
    hue = hue / max(1.0, float(hue.sum()))
    gray = arr.mean(axis=2)
    edges = float(np.mean(np.abs(np.diff(gray, axis=1)))) + float(np.mean(np.abs(np.diff(gray, axis=0))))
    vec = np.zeros(dim, dtype=np.float32)
    vec[0:9] = hue
    vec[9] = float(np.mean(gray))
    vec[10] = min(edges * 4.0, 1.0)
    vec[11] = float(np.mean(gray > 0.7))
    vec[12] = float(np.mean(gray < 0.2))
    nrm = float(np.linalg.norm(vec))
    return (vec / nrm) if nrm else vec


def scene_text_embed(text: str, dim: int = DIM) -> np.ndarray:
    cues = query_scene_cues(text)
    vec = np.zeros(dim, dtype=np.float32)
    if not cues:
        return vec
    hue = np.zeros(9, dtype=np.float32)
    n_col = 0
    for cue in cues:
        if cue in SCENE_COLOR:
            hue[_hue_bin(np.asarray(SCENE_COLOR[cue], dtype=np.float32))] += 1.0
            n_col += 1
        if cue in SCENE_SHAPE:
            vec[18 + SCENE_SHAPE[cue]] = 1.0
    if n_col:
        hue = hue / float(n_col)
        vec[0:9] = hue
        vec[12] = float(hue[8] > 0.4)
    nrm = float(np.linalg.norm(vec))
    return (vec / nrm) if nrm else vec


def load_video_records(prefer_cache: bool = True, *, meta_path: Path | None = None) -> list[VideoRec]:
    meta = meta_path or (CACHE_META if prefer_cache and CACHE_META.exists() else FIX_META)
    if not meta.exists():
        raise FileNotFoundError(f"no video metadata at {meta}")
    recs: list[VideoRec] = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        frames = [ROOT / p for p in (raw.get("frames") or []) if (ROOT / p).is_file()]
        if not frames:
            continue
        ap = raw.get("audio_path")
        audio = ROOT / ap if ap and (ROOT / ap).is_file() else None
        recs.append(
            VideoRec(
                video_id=raw["video_id"],
                frames=frames,
                audio_path=audio,
                transcript=str(raw.get("transcript") or ""),
                title=str(raw.get("title") or raw["video_id"]),
                source=str(raw.get("source") or "fixture"),
                href=str(raw.get("href") or ""),
                synthetic=bool(raw.get("synthetic", True)),
                captions=[str(c) for c in (raw.get("captions") or [])],
            )
        )
    if not recs:
        raise FileNotFoundError(f"metadata at {meta} has no readable frames")
    return recs


def load_video_qa_gold(gallery_ids: set[str], *, path: Path | None = None) -> list[dict]:
    gold = path or GOLD
    rows = [
        json.loads(line)
        for line in gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ok: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        gids = [g for g in (row.get("gold_video_ids") or []) if g in gallery_ids]
        if row.get("expected_status") != "abstain" and not gids:
            continue
        if gids:
            if gids[0] in seen:
                raise SystemExit(f"duplicate video gold {gids[0]} in {row.get('id')}")
            seen.add(gids[0])
        item = dict(row)
        item["gold_video_ids"] = gids
        ok.append(item)
    return ok


class VideoGallery:
    def __init__(self, recs: list[VideoRec], vision=None, audio: AudioEncoder | None = None) -> None:
        from abcite_core.clip_encoder import build_vision_encoder

        self.recs = recs
        self.vision = vision or build_vision_encoder()
        self.audio = audio or AudioEncoder()
        self.ids = [r.video_id for r in recs]
        self.transcripts = [r.transcript for r in recs]
        self.docs = [_tokenize(t) for t in self.transcripts]
        self.df: dict[str, int] = defaultdict(int)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            for t, f in tf.items():
                self.df[t] += 1
                self.postings[t].append((i, f))
        self.n = len(recs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.n)
        self.dl = [len(d) or 1 for d in self.docs]
        frame_vecs: list[np.ndarray] = []
        local = getattr(self.vision, "name", "") == "local-image"
        for rec in recs:
            if local:
                vecs = [scene_image_embed(p) for p in rec.frames]
            else:
                vecs = [np.asarray(self.vision.encode_image(p), dtype=np.float32) for p in rec.frames]
            mean = np.mean(np.stack(vecs, axis=0), axis=0) if vecs else hashing_vector(rec.video_id)
            nrm = float(np.linalg.norm(mean))
            frame_vecs.append(mean / nrm if nrm else mean)
        self.frame_vecs = np.stack(frame_vecs, axis=0).astype(np.float32)
        audio_vecs: list[np.ndarray] = []
        for rec in recs:
            if rec.audio_path:
                audio_vecs.append(self.audio.encode_audio(rec.audio_path))
            else:
                audio_vecs.append(hashing_vector(rec.transcript or rec.video_id, self.audio.dim))
        self.audio_vecs = np.stack(audio_vecs, axis=0).astype(np.float32)

    def transcript_bm25(self, query: str) -> np.ndarray:
        q = _tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)
        qn = Counter(q)
        k1, b = 1.5, 0.75
        avgdl = self.avgdl or 1.0
        for term, qtf in qn.items():
            posts = self.postings.get(term)
            if not posts:
                continue
            n_q = self.df.get(term, 0)
            idf = math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))
            for i, freq in posts:
                denom = freq + k1 * (1 - b + b * self.dl[i] / avgdl)
                scores[i] += idf * (freq * (k1 + 1) / denom) * qtf
        return scores

    def frame_scores(self, query: str) -> np.ndarray:
        if getattr(self.vision, "name", "") == "local-image":
            qv = scene_text_embed(query)
        else:
            encode_text = getattr(self.vision, "encode_text", None)
            if callable(encode_text):
                qv = np.asarray(encode_text(query), dtype=np.float32)
            else:
                qv = hashing_vector(query)
        if qv.shape[0] != self.frame_vecs.shape[1]:
            return np.zeros(self.n, dtype=np.float32)
        return self.frame_vecs @ qv

    def audio_scores(self, query: str) -> np.ndarray:
        qv = np.asarray(self.audio.encode_text(query), dtype=np.float32)
        if qv.shape[0] != self.audio_vecs.shape[1]:
            return np.zeros(self.n, dtype=np.float32)
        return self.audio_vecs @ qv

    def ranked_ids(self, scores: np.ndarray) -> list[str]:
        return [self.ids[int(i)] for i in np.argsort(-scores)]

    def search(self, query: str, top_k: int = 6, *, fuse: bool = True) -> list[tuple[VideoRec, float]]:
        frames = self.frame_scores(query)
        audio = self.audio_scores(query)
        bm25 = self.transcript_bm25(query)
        if fuse:
            from abcite_core.figure_gallery import _rrf

            # Within-video only: frames ⟂ audio ⟂ transcript. Not caption BM25 of papers.
            scores = _rrf(frames) + _rrf(audio)
        else:
            scores = frames
        order = np.argsort(-scores)[:top_k]
        return [(self.recs[int(i)], float(scores[i]), float(bm25[i]), float(frames[i])) for i in order]


def _recall_block(scores_list: list[np.ndarray], items: list[dict], gal: VideoGallery) -> dict:
    r1 = r5 = r10 = ndcg5 = 0.0
    n = 0
    for scores, it in zip(scores_list, items, strict=True):
        gold = it.get("gold_video_ids") or []
        if not gold:
            continue
        ranked = gal.ranked_ids(scores)
        r1 += 1.0 if gold[0] in ranked[:1] else 0.0
        r5 += 1.0 if gold[0] in ranked[:5] else 0.0
        r10 += 1.0 if gold[0] in ranked[:10] else 0.0
        ndcg5 += ndcg_at_k(ranked, gold, 5)
        n += 1
    n = max(1, n)
    return {
        "recall@1": round(r1 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "ndcg@5": round(ndcg5 / n, 4),
        "n": n,
    }


def evaluate_video_qa(gallery: VideoGallery, items: list[dict]) -> dict:
    from abcite_core.figure_gallery import _rrf

    answerable = [it for it in items if it.get("gold_video_ids")]
    fail = [it for it in answerable if it.get("transcript_bm25_should_fail")]
    empty = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "ndcg@5": 0.0, "n": 0}
    if not answerable:
        return {"protocol": "video_qa", "n_gallery": gallery.n, "n_queries": len(items)}
    questions = [it["question"] for it in answerable]
    bm25_rows = [gallery.transcript_bm25(q) for q in questions]
    frame_rows = [gallery.frame_scores(q) for q in questions]
    audio_rows = [gallery.audio_scores(q) for q in questions]
    fuse_rows = [
        _rrf(f) + _rrf(a) for f, a in zip(frame_rows, audio_rows, strict=True)
    ]
    idx = [i for i, it in enumerate(answerable) if it.get("transcript_bm25_should_fail")]
    report = {
        "protocol": "video_qa",
        "n_gallery": gallery.n,
        "n_queries": len(items),
        "n_answerable": len(answerable),
        "encoder": getattr(gallery.vision, "name", "unknown"),
        "synthetic_fixture": all(r.synthetic for r in gallery.recs),
        "transcript_bm25": _recall_block(bm25_rows, answerable, gallery),
        "frame_clip_t2i": _recall_block(frame_rows, answerable, gallery),
        "audio_clap_t2a": _recall_block(audio_rows, answerable, gallery),
        "within_video_rrf": _recall_block(fuse_rows, answerable, gallery),
        "transcript_bm25_should_fail_subset": (
            _recall_block([bm25_rows[i] for i in idx], fail, gallery) if fail else empty
        ),
        "frame_should_fail_subset": (
            _recall_block([frame_rows[i] for i in idx], fail, gallery) if fail else empty
        ),
        "note": (
            "Video keyframe + spoken-track retrieval. Query describes scene (color, layout), "
            "not the lecture transcript. RRF is within-video (frames ⟂ audio). "
            "CI local-image uses shared-space scene cues; CLIP when weights are present. "
            "Not mixed with paper caption BM25."
        ),
    }
    bm25_r1 = float(report["transcript_bm25"]["recall@1"])
    frame_r1 = float(report["frame_clip_t2i"]["recall@1"])
    report["frames_beat_transcript_bm25_r@1"] = frame_r1 > bm25_r1
    return report


def video_qa_response(
    question: str,
    gallery: VideoGallery,
    *,
    provider: str,
    trace_id: str,
    latency_ms: float,
    top_k: int = 6,
) -> QueryResponse:
    hits = gallery.search(question, top_k=top_k, fuse=True)
    scores = {h[0].video_id: h[1] for h in hits}
    if not hits or hits[0][1] <= 0:
        return QueryResponse(
            trace_id=trace_id,
            status=AnswerStatus.ABSTAIN,
            answer="Insufficient video evidence in the retrieved clips. Abstaining.",
            citations=[],
            provider=provider,
            retrieval_scores=scores,
            latency_ms=latency_ms,
        )
    rec, _score, _bm25, frames = hits[0]
    blob = " ".join([rec.transcript] + rec.captions)
    ov = caption_keyword_overlap(question, blob)
    cues = query_scene_cues(question)
    second = float(hits[1][3]) if len(hits) > 1 else 0.0
    scene_ok = bool(cues) and frames > 0.12 and frames >= second
    if not scene_ok and ov["n_overlap"] == 0:
        return QueryResponse(
            trace_id=trace_id,
            status=AnswerStatus.ABSTAIN,
            answer=f"Retrieved clip {rec.video_id} does not support the audiovisual question. Abstaining.",
            citations=[],
            provider=provider,
            retrieval_scores=scores,
            latency_ms=latency_ms,
        )
    quote = (rec.transcript or rec.title)[:280]
    thumb = None
    if rec.frames:
        thumb = f"/media/{rec.frames[0].name}"
    cite = Citation(
        chunk_id=rec.video_id,
        claim=question,
        quote=quote,
        char_start=0,
        char_end=len(quote),
        modality=Modality.VIDEO,
        source_url=rec.href or rec.frames[0].as_posix(),
        source_label=rec.source,
        thumbnail_url=thumb,
    )
    return QueryResponse(
        trace_id=trace_id,
        status=AnswerStatus.ANSWERED,
        answer=f"The retrieved clip {rec.video_id} matches the audiovisual question. Cite {rec.video_id}.",
        citations=[cite],
        provider=provider,
        retrieval_scores=scores,
        latency_ms=latency_ms,
    )


def extract_keyframes(video_path: Path, dest: Path, max_frames: int = 4) -> list[Path]:
    """Best-effort keyframe dump. OpenCV first, then ffmpeg."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 0:
            idxs = np.linspace(0, n - 1, num=min(max_frames, n), dtype=int)
            out: list[Path] = []
            for i, idx in enumerate(idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                path = dest / f"{video_path.stem}_k{i:02d}.jpg"
                cv2.imwrite(str(path), frame)
                out.append(path)
            cap.release()
            if out:
                return out
        cap.release()
    except Exception:
        pass
    import shutil
    import subprocess

    bin_ = shutil.which("ffmpeg")
    if not bin_:
        return []
    pattern = dest / "k%02d.jpg"
    r = subprocess.run(
        [
            bin_,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{max(1, max_frames)}",
            "-frames:v",
            str(max_frames),
            str(pattern),
        ],
        check=False,
    )
    if r.returncode != 0:
        return []
    return sorted(dest.glob("k*.jpg"))[:max_frames]
