"""Audio RAG: Whisper ASR + CLAP (or hashing) + transcript BM25.

CI uses a deterministic wav-hash encoder and committed fixture transcripts.
`faster-whisper` / LAION CLAP load when installed (GPU optional). Never required.
Fusion is **within-audio only** (transcript BM25 ⟂ CLAP). Do not RRF into
caption BM25 for the text-only or visual_qa protocols.
"""

from __future__ import annotations

import json
import math
import os
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from abcite_core.metrics import ndcg_at_k
from abcite_core.models import AnswerStatus, Citation, Modality, QueryResponse
from abcite_core.visual_qa import caption_keyword_overlap
from abcite_core.vision import hashing_vector

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold" / "audio_qa.jsonl"
FIX_META = ROOT / "evals" / "audio" / "fixture" / "metadata.jsonl"
CACHE_META = ROOT / "evals" / "audio" / "cache" / "metadata.jsonl"
DIM = 256


@dataclass
class AudioRec:
    audio_id: str
    path: Path
    transcript: str
    title: str
    source: str
    href: str = ""
    synthetic: bool = True


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9_\-]+", text.lower())


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        nch = fh.getnchannels()
        sw = fh.getsampwidth()
        rate = fh.getframerate()
        n = fh.getnframes()
        raw = fh.readframes(n)
    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    return samples, rate


def _write_pcm(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32000.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm)


def write_sine_wav(path: Path, freq_hz: float, seconds: float = 0.45, rate: int = 16000) -> None:
    write_fixture_wav(path, "sine", freq_hz=freq_hz, seconds=seconds, rate=rate)


def write_fixture_wav(
    path: Path,
    kind: str,
    *,
    freq_hz: float = 440.0,
    freq_b: float = 0.0,
    seconds: float = 0.8,
    rate: int = 16000,
) -> None:
    """Honest synthetic clips with distinct spectra (CI; labeled)."""
    n = max(160, int(rate * seconds))
    t = np.arange(n, dtype=np.float32) / float(rate)
    kind = kind.lower()
    if kind == "sine":
        sig = 0.35 * np.sin(2 * math.pi * freq_hz * t)
    elif kind == "chirp_up":
        sig = 0.35 * np.sin(2 * math.pi * (200.0 * t + 2400.0 * t * t))
    elif kind == "chirp_down":
        sig = 0.35 * np.sin(2 * math.pi * (4200.0 * t - 2400.0 * t * t))
    elif kind == "pulse":
        period = max(2, int(rate / max(1.0, freq_hz)))
        sig = np.zeros(n, dtype=np.float32)
        sig[::period] = 0.9
        sig[1::period] = -0.4
    elif kind == "hiss":
        rng = np.random.default_rng(7)
        sig = 0.22 * rng.standard_normal(n).astype(np.float32)
    elif kind == "roar":
        rng = np.random.default_rng(11)
        white = rng.standard_normal(n).astype(np.float32)
        sig = np.convolve(white, np.ones(48, dtype=np.float32) / 48.0, mode="same")
        sig *= 0.45
    elif kind == "two_tone":
        b = freq_b or 2400.0
        sig = 0.22 * np.sin(2 * math.pi * freq_hz * t) + 0.22 * np.sin(2 * math.pi * b * t)
    elif kind == "tremolo":
        sig = 0.35 * np.sin(2 * math.pi * freq_hz * t) * (0.45 + 0.55 * np.sin(2 * math.pi * 8.0 * t))
    elif kind == "hush":
        rng = np.random.default_rng(3)
        sig = 0.012 * rng.standard_normal(n).astype(np.float32)
    else:
        sig = 0.35 * np.sin(2 * math.pi * freq_hz * t)
    _write_pcm(path, sig.astype(np.float32), rate)


def _spectral_bands(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """8 log-mel-ish rfft bands + 8 temporal descriptors. Shared by wav/text."""
    del rate
    win = 256
    hop = 128
    n_w = max(1, (len(samples) - win) // hop + 1)
    bands = np.zeros(8, dtype=np.float32)
    for i in range(n_w):
        sl = samples[i * hop : i * hop + win]
        if sl.size < win:
            sl = np.pad(sl, (0, win - sl.size))
        spec = np.abs(np.fft.rfft(sl * np.hanning(win)))
        chunk = max(1, spec.size // 8)
        for b in range(8):
            bands[b] += float(spec[b * chunk : (b + 1) * chunk].mean())
    bands = np.log1p(bands)
    raw = bands / (float(np.linalg.norm(bands)) + 1e-8)
    peak_i = int(np.argmax(bands))
    peakedness = float(bands[peak_i] / (float(np.mean(bands)) + 1e-8))
    if peakedness >= 2.2:
        peaked = np.full(8, 0.04, dtype=np.float32)
        peaked[peak_i] = 1.0
        if peak_i > 0:
            peaked[peak_i - 1] = 0.28
        if peak_i < 7:
            peaked[peak_i + 1] = 0.28
        mix = 0.8 * peaked + 0.2 * raw
    else:
        mix = raw
    nrm = float(np.linalg.norm(mix))
    bands = mix / nrm if nrm else mix
    rms = float(np.sqrt(np.mean(np.square(samples))) + 1e-8)
    peak = float(np.max(np.abs(samples)) + 1e-8)
    zc = float(np.mean(np.abs(np.diff(np.sign(samples)))))
    crest = peak / rms
    silent = float(np.mean(np.abs(samples) < 0.03))
    # Spectral centroid slope across windows (rising vs falling chirp).
    cents: list[float] = []
    for i in range(n_w):
        sl = samples[i * hop : i * hop + win]
        if sl.size < win:
            sl = np.pad(sl, (0, win - sl.size))
        spec = np.abs(np.fft.rfft(sl * np.hanning(win)))
        freqs = np.arange(spec.size, dtype=np.float32)
        cents.append(float((spec * freqs).sum() / (spec.sum() + 1e-8)))
    slope = 0.0
    if len(cents) >= 2:
        slope = float(np.clip((cents[-1] - cents[0]) / (max(cents) + 1e-8), -1.0, 1.0))
    top2 = np.argsort(raw)[-2:]
    two_peak = 1.0 if abs(int(top2[0]) - int(top2[1])) >= 2 and raw[top2[0]] > 0.2 else 0.0
    temporal = np.array(
        [
            min(crest / 8.0, 1.0),
            min(zc, 1.0),
            min(rms * 3.0, 1.0),
            silent,
            1.0 if crest > 4.0 else 0.0,
            1.0 if silent > 0.7 else 0.0,
            1.0 if zc > 0.45 and peakedness < 2.2 else 0.0,
            two_peak,
        ],
        dtype=np.float32,
    )
    # Stash chirp slope in the last temporal-adjacent slot via a 9th feature copied into vec later.
    temporal = np.concatenate([temporal, np.array([max(slope, 0.0), max(-slope, 0.0)], dtype=np.float32)])
    return bands, temporal


def wav_embed(path: Path, dim: int = DIM) -> np.ndarray:
    """Spectral + temporal features in a shared 256-d space (CI stand-in for CLAP)."""
    samples, rate = load_wav_mono(path)
    if samples.size == 0:
        return hashing_vector(path.name, dim)
    bands, temporal = _spectral_bands(samples, rate)
    vec = np.zeros(dim, dtype=np.float32)
    vec[:8] = bands
    vec[8 : 8 + temporal.size] = temporal
    nrm = float(np.linalg.norm(vec))
    return (vec / nrm) if nrm else vec


# Query tokens that describe sound, not lecture wording. Must stay out of fixture transcripts.
ACOUSTIC_CUES: dict[str, np.ndarray] = {
    "drone": np.array([1, 0.15, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "rumble": np.array([1, 0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "bass": np.array([1, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "hum": np.array([0.25, 1, 0.25, 0, 0, 0, 0, 0], dtype=np.float32),
    "kilohertz": np.array([0.1, 1, 0.4, 0, 0, 0, 0, 0], dtype=np.float32),
    "midhigh": np.array([0, 0.2, 1, 0.4, 0, 0, 0, 0], dtype=np.float32),
    "piercing": np.array([0, 0, 0.2, 1, 0.5, 0, 0, 0], dtype=np.float32),
    "beep": np.array([0, 0, 0, 0.2, 0.5, 1, 0.4, 0], dtype=np.float32),
    "whistle": np.array([0, 0, 0, 0, 0.3, 0.6, 1, 0.4], dtype=np.float32),
    "shrill": np.array([0, 0, 0, 0, 0.1, 0.3, 0.6, 1], dtype=np.float32),
    "chirp": np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.6, 0.3, 0.1], dtype=np.float32),
    "sweep": np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.6, 0.3, 0.1], dtype=np.float32),
    "falling": np.array([0.2, 0.3, 0.5, 0.7, 0.9, 0.8, 0.4, 0.1], dtype=np.float32),
    "click": np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32),
    "pulse": np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32),
    "hiss": np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6], dtype=np.float32),
    "broadband": np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6], dtype=np.float32),
    "roar": np.array([0.9, 0.7, 0.3, 0.1, 0, 0, 0, 0], dtype=np.float32),
    "twotone": np.array([0.7, 0.1, 0, 0.2, 0.8, 0.3, 0, 0], dtype=np.float32),
    "interval": np.array([0.5, 0.2, 0, 0, 0.2, 0.4, 0.8, 0.2], dtype=np.float32),
    "wobble": np.array([0.4, 0.8, 0.3, 0, 0, 0, 0, 0], dtype=np.float32),
    "tremolo": np.array([0.4, 0.8, 0.3, 0, 0, 0, 0, 0], dtype=np.float32),
    "hush": np.array([0.05, 0.02, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "roomtone": np.array([0.05, 0.02, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "ambient": np.array([0.05, 0.02, 0, 0, 0, 0, 0, 0], dtype=np.float32),
}

_TEMPORAL_BY_CUE = {
    "click": np.array([0.9, 0.4, 0.3, 0.4, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    "pulse": np.array([0.9, 0.5, 0.3, 0.3, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    "hiss": np.array([0.2, 0.8, 0.5, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    "broadband": np.array([0.2, 0.8, 0.5, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    "hush": np.array([0.1, 0.1, 0.05, 0.95, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    "roomtone": np.array([0.1, 0.1, 0.05, 0.95, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    "ambient": np.array([0.1, 0.1, 0.05, 0.95, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    "twotone": np.array([0.3, 0.3, 0.4, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
    "interval": np.array([0.3, 0.3, 0.4, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
    "chirp": np.array([0.3, 0.3, 0.4, 0, 0, 0, 0, 0, 0.6, 0.2], dtype=np.float32),
    "sweep": np.array([0.3, 0.3, 0.4, 0, 0, 0, 0, 0, 0.6, 0.2], dtype=np.float32),
    "falling": np.array([0.3, 0.3, 0.4, 0, 0, 0, 0, 0, 0.0, 1.0], dtype=np.float32),
    "roar": np.array([0.3, 0.2, 0.5, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
}


def query_acoustic_cues(text: str) -> list[str]:
    blob = text.lower().replace("two-tone", "twotone").replace("two tone", "twotone")
    blob = blob.replace("room tone", "roomtone").replace("mid-high", "midhigh")
    return [k for k in ACOUSTIC_CUES if k in blob]


def acoustic_text_embed(text: str, dim: int = DIM) -> np.ndarray:
    """Map sound-description queries into the same 16 feature slots as wav_embed."""
    cues = query_acoustic_cues(text)
    vec = np.zeros(dim, dtype=np.float32)
    if not cues:
        return vec
    bands = np.zeros(8, dtype=np.float32)
    temporal = np.zeros(10, dtype=np.float32)
    for cue in cues:
        bands += ACOUSTIC_CUES[cue]
        temporal += _TEMPORAL_BY_CUE.get(cue, np.array([0.3, 0.2, 0.4, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    if "falling" in cues:
        temporal[8] = 0.0
        temporal[9] = 1.0
    elif "chirp" in cues or "sweep" in cues:
        temporal[8] = 1.0
        temporal[9] = 0.0
    bn = float(np.linalg.norm(bands))
    if bn:
        bands = bands / bn
    tn = float(np.linalg.norm(temporal))
    if tn:
        temporal = temporal / tn
    vec[:8] = bands
    vec[8 : 8 + temporal.size] = temporal
    nrm = float(np.linalg.norm(vec))
    return (vec / nrm) if nrm else vec


class AudioEncoder:
    """CLAP if available; otherwise shared-space wav spectral features."""

    name = "wav-hash"
    dim = DIM

    def encode_audio(self, path: Path) -> np.ndarray:
        clap = _try_clap()
        if clap is not None:
            return clap.encode_audio(path)
        return wav_embed(path, self.dim)

    def encode_text(self, text: str) -> np.ndarray:
        clap = _try_clap()
        if clap is not None:
            return clap.encode_text(text)
        return acoustic_text_embed(text, self.dim)

    def transcribe(self, path: Path, fallback: str = "") -> str:
        asr = os.environ.get("ABCITE_WHISPER", "auto").lower()
        if asr in {"0", "false", "off"}:
            return fallback
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(os.environ.get("ABCITE_WHISPER_MODEL", "tiny"), device="auto")
            segs, _ = model.transcribe(str(path), beam_size=1)
            text = " ".join(s.text.strip() for s in segs).strip()
            if text:
                return text
        except Exception:
            pass
        return fallback


class _ClapEnc:
    def __init__(self, model, processor, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.name = "clap"
        self.dim = 512

    def encode_audio(self, path: Path) -> np.ndarray:
        import torch

        samples, rate = load_wav_mono(path)
        inputs = self.processor(audios=samples, sampling_rate=rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.get_audio_features(**inputs)
        vec = out[0].float().cpu().numpy()
        n = float(np.linalg.norm(vec))
        return (vec / n) if n else vec.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        import torch

        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        vec = out[0].float().cpu().numpy()
        n = float(np.linalg.norm(vec))
        return (vec / n) if n else vec.astype(np.float32)


_CLAP: _ClapEnc | None | bool = False


def _try_clap() -> _ClapEnc | None:
    global _CLAP
    if _CLAP is False:
        # Opt-in: default off so CI never hits the Hub. ABCITE_CLAP=1 loads LAION CLAP.
        if os.environ.get("ABCITE_CLAP", "0").lower() not in {"1", "true", "yes", "on"}:
            _CLAP = None
            return None
        try:
            import torch
            from transformers import ClapModel, ClapProcessor

            name = os.environ.get("ABCITE_CLAP_MODEL", "laion/clap-htsat-unfused")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ClapModel.from_pretrained(name).to(device).eval()
            proc = ClapProcessor.from_pretrained(name)
            _CLAP = _ClapEnc(model, proc, device)
        except Exception:
            _CLAP = None
    return None if _CLAP is False else _CLAP


def load_audio_records(prefer_cache: bool = True) -> list[AudioRec]:
    meta = CACHE_META if prefer_cache and CACHE_META.exists() else FIX_META
    if not meta.exists():
        raise FileNotFoundError(f"no audio metadata at {meta}")
    recs: list[AudioRec] = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        path = ROOT / raw["path"]
        if not path.is_file():
            continue
        recs.append(
            AudioRec(
                audio_id=raw["audio_id"],
                path=path,
                transcript=str(raw.get("transcript") or ""),
                title=str(raw.get("title") or raw["audio_id"]),
                source=str(raw.get("source") or "fixture"),
                href=str(raw.get("href") or ""),
                synthetic=bool(raw.get("synthetic", True)),
            )
        )
    if not recs:
        raise FileNotFoundError(f"metadata at {meta} has no readable wav")
    return recs


def load_audio_qa_gold(gallery_ids: set[str], *, path: Path | None = None) -> list[dict]:
    gold = path or GOLD
    rows = [
        json.loads(line)
        for line in gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ok: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        gids = [g for g in (row.get("gold_audio_ids") or []) if g in gallery_ids]
        if row.get("expected_status") != "abstain" and not gids:
            continue
        if gids:
            if gids[0] in seen:
                raise SystemExit(f"duplicate audio gold {gids[0]} in {row.get('id')}")
            seen.add(gids[0])
        item = dict(row)
        item["gold_audio_ids"] = gids
        ok.append(item)
    return ok


class AudioGallery:
    def __init__(self, recs: list[AudioRec], encoder: AudioEncoder | None = None) -> None:
        self.recs = recs
        self.encoder = encoder or AudioEncoder()
        self.ids = [r.audio_id for r in recs]
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
        self.audio_vecs = np.stack(
            [self.encoder.encode_audio(r.path) for r in recs], axis=0
        ).astype(np.float32)

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

    def clap_scores(self, query: str) -> np.ndarray:
        qv = np.asarray(self.encoder.encode_text(query), dtype=np.float32)
        if qv.shape[0] != self.audio_vecs.shape[1]:
            return np.zeros(self.n, dtype=np.float32)
        return self.audio_vecs @ qv

    def ranked_ids(self, scores: np.ndarray) -> list[str]:
        return [self.ids[int(i)] for i in np.argsort(-scores)]

    def search(self, query: str, top_k: int = 6, *, fuse: bool = False) -> list[tuple[AudioRec, float, float, float]]:
        bm25 = self.transcript_bm25(query)
        clap = self.clap_scores(query)
        if fuse:
            from abcite_core.figure_gallery import _rrf

            scores = _rrf(bm25) + _rrf(clap)
        else:
            scores = clap
        order = np.argsort(-scores)[:top_k]
        return [
            (self.recs[int(i)], float(scores[i]), float(bm25[i]), float(clap[i]))
            for i in order
        ]


def _recall_block(scores_list: list[np.ndarray], items: list[dict], gal: AudioGallery) -> dict:
    r1 = r5 = r10 = ndcg5 = 0.0
    n = 0
    for scores, it in zip(scores_list, items, strict=True):
        gold = it.get("gold_audio_ids") or []
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


def evaluate_audio_qa(gallery: AudioGallery, items: list[dict]) -> dict:
    from abcite_core.figure_gallery import _rrf

    answerable = [it for it in items if it.get("gold_audio_ids")]
    fail = [it for it in answerable if it.get("transcript_bm25_should_fail")]
    empty = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "ndcg@5": 0.0, "n": 0}
    if not answerable:
        return {"protocol": "audio_qa", "n_gallery": gallery.n, "n_queries": len(items)}
    questions = [it["question"] for it in answerable]
    bm25_rows = [gallery.transcript_bm25(q) for q in questions]
    clap_rows = [gallery.clap_scores(q) for q in questions]
    fuse_rows = [_rrf(b) + _rrf(c) for b, c in zip(bm25_rows, clap_rows, strict=True)]
    idx = [i for i, it in enumerate(answerable) if it.get("transcript_bm25_should_fail")]
    report = {
        "protocol": "audio_qa",
        "n_gallery": gallery.n,
        "n_queries": len(items),
        "n_answerable": len(answerable),
        "encoder": gallery.encoder.name,
        "synthetic_fixture": all(r.synthetic for r in gallery.recs),
        "transcript_bm25": _recall_block(bm25_rows, answerable, gallery),
        "clap_t2a": _recall_block(clap_rows, answerable, gallery),
        "within_audio_rrf": _recall_block(fuse_rows, answerable, gallery),
        "transcript_bm25_should_fail_subset": (
            _recall_block([bm25_rows[i] for i in idx], fail, gallery) if fail else empty
        ),
        "clap_should_fail_subset": (
            _recall_block([clap_rows[i] for i in idx], fail, gallery) if fail else empty
        ),
        "note": (
            "Audio retrieval. Query describes sound (pitch, chirp, pulse, hush), not spoken words. "
            "transcript_bm25_should_fail has ≤1 content-token overlap. "
            "CI encoder is shared-space spectral cues (wav-hash); CLAP when ABCITE_CLAP=1. "
            "RRF is within-audio only."
        ),
    }
    bm25_r1 = float(report["transcript_bm25"]["recall@1"])
    clap_r1 = float(report["clap_t2a"]["recall@1"])
    report["clap_beats_transcript_bm25_r@1"] = clap_r1 > bm25_r1
    return report


def audio_qa_response(
    question: str,
    gallery: AudioGallery,
    *,
    provider: str,
    trace_id: str,
    latency_ms: float,
    top_k: int = 6,
) -> QueryResponse:
    hits = gallery.search(question, top_k=top_k, fuse=False)
    scores = {h[0].audio_id: h[1] for h in hits}
    if not hits or hits[0][3] < 0.08:
        return QueryResponse(
            trace_id=trace_id,
            status=AnswerStatus.ABSTAIN,
            answer="Insufficient audio evidence in the retrieved clips. Abstaining.",
            citations=[],
            provider=provider,
            retrieval_scores=scores,
            latency_ms=latency_ms,
        )
    rec, _score, _bm25, clap = hits[0]
    cues = query_acoustic_cues(question)
    ov = caption_keyword_overlap(question, rec.transcript)
    second = hits[1][3] if len(hits) > 1 else 0.0
    sound_ok = bool(cues) and clap >= 0.18 and clap >= second + 0.02
    if not sound_ok and ov["n_overlap"] == 0:
        return QueryResponse(
            trace_id=trace_id,
            status=AnswerStatus.ABSTAIN,
            answer=f"Retrieved clip {rec.audio_id} does not support the spoken question. Abstaining.",
            citations=[],
            provider=provider,
            retrieval_scores=scores,
            latency_ms=latency_ms,
        )
    quote = rec.transcript[:280]
    cite = Citation(
        chunk_id=rec.audio_id,
        claim=question,
        quote=quote,
        char_start=0,
        char_end=len(quote),
        modality=Modality.AUDIO,
        source_url=rec.href or rec.path.as_posix(),
        source_label=rec.source,
    )
    return QueryResponse(
        trace_id=trace_id,
        status=AnswerStatus.ANSWERED,
        answer=f"The retrieved clip {rec.audio_id} matches the spoken question. Cite {rec.audio_id}.",
        citations=[cite],
        provider=provider,
        retrieval_scores=scores,
        latency_ms=latency_ms,
    )
