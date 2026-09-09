"""Hybrid BM25 + dense + figure-image retrieval (RRF)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from abcite_core.models import Chunk
from abcite_core.vision import DIM, VisionEncoder, hashing_vector

TOKEN = re.compile(r"[a-z0-9_+\-./]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class HashingEncoder:
    """Deterministic MD5-bucket bag-of-tokens. Swap for sentence-transformers."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([hashing_vector(t, self.dim) for t in texts], axis=0)


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def build_text_encoder():
    try:
        return SentenceTransformerEncoder()
    except Exception:
        return HashingEncoder()


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    dense: float
    image: float = 0.0


class HybridIndex:
    def __init__(self, chunks: list[Chunk], encoder=None, vision: VisionEncoder | None = None) -> None:
        self.chunks = chunks
        self.encoder = encoder or HashingEncoder()
        if vision is not None:
            self.vision = vision
        else:
            from abcite_core.clip_encoder import build_vision_encoder

            self.vision = build_vision_encoder()
        self.docs = [tokenize(self._blob(c)) for c in chunks]
        self.df: dict[str, int] = defaultdict(int)
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1
        self.n = len(chunks)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.n)
        self.k1 = 1.5
        self.b = 0.75
        blobs = [self._blob(c) for c in chunks]
        self.dense = self.encoder.encode(blobs)
        self.image_vecs = np.stack([self._image_vec(c) for c in chunks], axis=0)

    def _image_vec(self, chunk: Chunk) -> np.ndarray:
        if chunk.image_path:
            path = Path(chunk.image_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            if path.exists():
                return np.asarray(self.vision.encode_image(path), dtype=np.float32)
        cap = f"{chunk.caption or ''} {chunk.title}"
        encode_text = getattr(self.vision, "encode_text", None)
        if callable(encode_text):
            return np.asarray(encode_text(cap), dtype=np.float32)
        return hashing_vector(cap)

    @staticmethod
    def _blob(chunk: Chunk) -> str:
        extra = " ".join(chunk.residues + chunk.tags)
        cap = chunk.caption or ""
        return (
            f"{chunk.title} {chunk.text} {cap} {extra} {chunk.pdb_id or ''} "
            f"{chunk.antigen or ''} {chunk.antibody or ''} {chunk.modality.value}"
        )

    def bm25_scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)
        qn = Counter(q)
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            dl = len(doc) or 1
            s = 0.0
            for term, qtf in qn.items():
                n_q = self.df.get(term, 0)
                if n_q == 0:
                    continue
                idf = math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * (freq * (self.k1 + 1) / denom) * qtf
            scores[i] = s
        return scores

    def dense_scores(self, query: str) -> np.ndarray:
        qv = self.encoder.encode([query])[0]
        return self.dense @ qv

    def image_scores(self, query: str) -> np.ndarray:
        encode_text = getattr(self.vision, "encode_text", None)
        if callable(encode_text):
            qv = np.asarray(encode_text(query), dtype=np.float32)
        else:
            qv = hashing_vector(query)
        if qv.shape[0] != self.image_vecs.shape[1]:
            return np.zeros(self.n, dtype=np.float32)
        return self.image_vecs @ qv

    def search(self, query: str, top_k: int = 6) -> list[Hit]:
        bm25 = self.bm25_scores(query)
        dense = self.dense_scores(query)
        image = self.image_scores(query)
        rrf = _rrf(bm25) + _rrf(dense) + 0.5 * _rrf(image)
        qlow = query.lower()
        if any(w in qlow for w in ("figure", "diagram", "caption", "snapshot", "residue map", "schematic")):
            for i, c in enumerate(self.chunks):
                if c.modality.value in {"figure", "structure"}:
                    rrf[i] += 0.03
        order = np.argsort(-rrf)[:top_k]
        hits: list[Hit] = []
        for i in order:
            hits.append(
                Hit(
                    chunk=self.chunks[int(i)],
                    score=float(rrf[i]),
                    bm25=float(bm25[i]),
                    dense=float(dense[i]),
                    image=float(image[i]),
                )
            )
        return hits

    def search_by_image(self, path: Path, top_k: int = 5) -> list[Hit]:
        qv = np.asarray(self.vision.encode_image(path), dtype=np.float32)
        if qv.shape[0] != self.image_vecs.shape[1]:
            return []
        sims = self.image_vecs @ qv
        order = np.argsort(-sims)[:top_k]
        hits: list[Hit] = []
        for i in order:
            hits.append(
                Hit(
                    chunk=self.chunks[int(i)],
                    score=float(sims[i]),
                    bm25=0.0,
                    dense=0.0,
                    image=float(sims[i]),
                )
            )
        return hits


def _rrf(scores: np.ndarray, k: int = 60) -> np.ndarray:
    order = np.argsort(-scores)
    out = np.zeros_like(scores, dtype=np.float32)
    for rank, idx in enumerate(order, start=1):
        out[idx] = 1.0 / (k + rank)
    return out
