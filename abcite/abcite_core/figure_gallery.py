"""Image gallery for the visual multimodal protocol (not the 14-SVG cite-id demo).

Image vectors are CLIP/hash pixels only — captions and PDB ids never enter encode_image.
Queries are visual-content text; PDB tokens are stripped before CLIP text encode.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from abcite_core.holdout import strip_caption_leak, strip_pdb_ids, visual_query_ok
from abcite_core.vision import hashing_vector

ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "evals" / "figures"


@dataclass
class FigureRec:
    figure_id: str
    path: Path
    caption: str
    pmcid: str | None
    split: str
    visual_class: str
    source: str


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9_\-]+", text.lower())


def load_figure_records(prefer_cache: bool = True) -> list[FigureRec]:
    cache_meta = FIG_ROOT / "cache" / "metadata.jsonl"
    fixture_meta = FIG_ROOT / "fixture" / "metadata.jsonl"
    meta = cache_meta if prefer_cache and cache_meta.exists() else fixture_meta
    if not meta.exists():
        raise FileNotFoundError(
            f"No figure metadata. Run: python evals/download_figures.py --max 1200\nlooked for {cache_meta} and {fixture_meta}"
        )
    recs: list[FigureRec] = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        path = FIG_ROOT / raw["path"]
        if not path.is_file():
            continue
        recs.append(
            FigureRec(
                figure_id=raw["figure_id"],
                path=path,
                caption=str(raw.get("caption") or ""),
                pmcid=raw.get("pmcid"),
                split=str(raw.get("split") or ""),
                visual_class=str(raw.get("visual_class") or "scientific figure panel"),
                source=str(raw.get("source") or ""),
            )
        )
    if not recs:
        raise FileNotFoundError(f"metadata at {meta} has no readable images")
    return recs


class FigureGallery:
    def __init__(self, recs: list[FigureRec], vision) -> None:
        self.recs = recs
        self.vision = vision
        self.ids = [r.figure_id for r in recs]
        self.captions = [r.caption for r in recs]
        self.cap_docs = [_tokenize(strip_pdb_ids(c)) for c in self.captions]
        self.df: dict[str, int] = defaultdict(int)
        for doc in self.cap_docs:
            for t in set(doc):
                self.df[t] += 1
        self.n = len(recs)
        self.avgdl = sum(len(d) for d in self.cap_docs) / max(1, self.n)
        self.image_vecs = self._load_or_encode()

    def _cache_key(self) -> Path:
        name = getattr(self.vision, "name", "enc")
        return FIG_ROOT / "cache" / f"embeddings-{name}-n{self.n}.npy"

    def _load_or_encode(self) -> np.ndarray:
        cache = self._cache_key()
        ids_path = cache.with_suffix(".ids.json")
        if cache.exists() and ids_path.exists():
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if ids == self.ids:
                arr = np.load(cache)
                if arr.shape[0] == self.n:
                    return arr.astype(np.float32)
        vecs = []
        encode_image = self.vision.encode_image
        for rec in self.recs:
            # Caption / PDB must not be concatenated into this vector.
            vecs.append(np.asarray(encode_image(rec.path), dtype=np.float32))
        arr = np.stack(vecs, axis=0)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if "fixture" not in str(self.recs[0].path):
            np.save(cache, arr)
            ids_path.write_text(json.dumps(self.ids), encoding="utf-8")
        return arr

    def _bm25(self, query: str) -> np.ndarray:
        q = _tokenize(strip_pdb_ids(query))
        scores = np.zeros(self.n, dtype=np.float32)
        qn = Counter(q)
        k1, b = 1.5, 0.75
        for i, doc in enumerate(self.cap_docs):
            tf = Counter(doc)
            dl = len(doc) or 1
            s = 0.0
            for term, qtf in qn.items():
                n_q = self.df.get(term, 0)
                if n_q == 0:
                    continue
                idf = math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * dl / self.avgdl)
                s += idf * (freq * (k1 + 1) / denom) * qtf
            scores[i] = s
        return scores

    def _query_vec(self, question: str, caption_for_holdout: str | None) -> np.ndarray:
        text = strip_caption_leak(question, caption_for_holdout)
        text = strip_pdb_ids(text)
        encode_text = getattr(self.vision, "encode_text", None)
        if callable(encode_text):
            return np.asarray(encode_text(text), dtype=np.float32)
        return hashing_vector(text)

    def clip_image_only_scores(self, question: str) -> np.ndarray:
        qv = self._query_vec(question, None)
        if qv.shape[0] != self.image_vecs.shape[1]:
            return np.zeros(self.n, dtype=np.float32)
        return self.image_vecs @ qv

    def caption_bm25_scores(self, question: str) -> np.ndarray:
        return self._bm25(question)

    def fusion_scores(self, question: str) -> np.ndarray:
        cap = self.caption_bm25_scores(question)
        img = self.clip_image_only_scores(question)
        return _rrf(cap) + _rrf(img)

    def ranked_ids(self, scores: np.ndarray) -> list[str]:
        order = np.argsort(-scores)
        return [self.ids[int(i)] for i in order]

    def mean_rank_gap_image(self) -> float:
        """Image-to-image self-minus-NN cosine on this gallery."""
        sims = self.image_vecs @ self.image_vecs.T
        gaps: list[float] = []
        for i in range(self.n):
            row = sims[i].copy()
            self_s = float(row[i])
            row[i] = -np.inf
            nn = float(np.max(row)) if self.n > 1 else 0.0
            gaps.append(self_s - nn)
        return float(sum(gaps) / len(gaps)) if gaps else 0.0


def _rrf(scores: np.ndarray, k: int = 60) -> np.ndarray:
    order = np.argsort(-scores)
    out = np.zeros_like(scores, dtype=np.float32)
    for rank, idx in enumerate(order, start=1):
        out[idx] = 1.0 / (k + rank)
    return out


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    g = set(gold)
    return 1.0 if any(x in g for x in ranked[:k]) else 0.0


def mean_text_gap(gallery: FigureGallery, questions: list[str]) -> float:
    gaps: list[float] = []
    for q in questions:
        scores = gallery.clip_image_only_scores(q)
        order = np.argsort(-scores)
        if len(order) < 2:
            continue
        gaps.append(float(scores[order[0]] - scores[order[1]]))
    return float(sum(gaps) / len(gaps)) if gaps else 0.0


def synthesize_visual_queries(recs: list[FigureRec], limit: int | None = None) -> list[dict]:
    """Queries describe figure type; captions and PDB ids are not copied in."""
    items = recs if limit is None else recs[:limit]
    gold: list[dict] = []
    for rec in items:
        q = f"{rec.visual_class} panel from a paper figure"
        if not visual_query_ok(q):
            q = "scientific plot panel from a paper figure"
        gold.append(
            {
                "id": f"vis-{rec.figure_id}",
                "question": q,
                "gold_figure_ids": [rec.figure_id],
                "visual_class": rec.visual_class,
            }
        )
    return gold
