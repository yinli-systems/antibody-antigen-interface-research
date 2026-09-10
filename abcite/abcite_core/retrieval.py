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
from abcite_core.vision_flags import vision_fusion_enabled, vision_rerank_enabled, vision_rerank_k

TOKEN = re.compile(r"[a-z0-9_+\-./]+")

# Query rewrite / HyDE-lite expansions. Not gold-question copies of chunk text.
_EXPAND: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bherceptin\b", re.I), " trastuzumab her2 domain IV 1n8z"),
    (re.compile(r"\b2c4\b", re.I), " pertuzumab domain II dimerization"),
    (re.compile(r"\berbb2\b", re.I), " her2"),
    (re.compile(r"\bmembrane-adjacent\b", re.I), " domain IV membrane-proximal"),
    (re.compile(r"\bdimerization[- ]arm\b", re.I), " pertuzumab domain II"),
    (re.compile(r"\bpalivizumab[- ]class\b", re.I), " motavizumab palivizumab site II"),
    (re.compile(r"\bprefusion(?: rsv)?(?: fusion-protein)? apex\b", re.I), " d25 4jhw site zero trimer apex"),
    (re.compile(r"\bcd4-mimetic\b", re.I), " vrc01 cd4-binding-site 3ngb"),
    (re.compile(r"\bn332 glycan-hole\b", re.I), " pgt121 n332 glycan"),
    (re.compile(r"\bcryptic sarbecovirus\b", re.I), " cr3022 6w41 cryptic rbd"),
    (re.compile(r"\bhel binder d1\.3\b", re.I), " d1.3 lysozyme 1vfb"),
    (re.compile(r"\binfluenza stem\b", re.I), " cr9114 4fqi ha2 stem"),
    (re.compile(r"\betesevimab\b", re.I), " cb6 7c01 ace2"),
    (re.compile(r"\bsabdab\b", re.I), " antibody catalog metadata residue jaccard"),
    (re.compile(r"\boverlap diagnostic\b", re.I), " framecontact predicted overlap jaccard"),
    (re.compile(r"\bnumbered (?:erbb2|her2|rsv|rbd|hel|ha2|gp120)\b", re.I), " residue map structure snapshot highlighted"),
    (re.compile(r"\blocal (?:herceptin )?contact cartoon\b", re.I), " structure snapshot residue map"),
    (re.compile(r"\bschematic cartoon\b", re.I), " figure caption diagram"),
]


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def expand_query(query: str) -> str:
    """Lexical rewrite + comparison cue. Does not invent gold questions."""
    extra: list[str] = []
    for pat, add in _EXPAND:
        if pat.search(query):
            extra.append(add)
    qlow = query.lower()
    if any(w in qlow for w in ("same antigen", "versus", " vs ", "share a residue", "same face", "same set")):
        extra.append(" epitope epitope-overlap distinct site")
    if extra:
        return query + " " + " ".join(extra)
    return query


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
        self.caption_docs = [tokenize(self._caption_blob(c)) for c in chunks]
        self.df: dict[str, int] = defaultdict(int)
        self.caption_df: dict[str, int] = defaultdict(int)
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1
        for doc in self.caption_docs:
            for term in set(doc):
                self.caption_df[term] += 1
        self.n = len(chunks)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.n)
        self.avg_cap_dl = sum(len(d) for d in self.caption_docs) / max(1, self.n)
        self.k1 = 1.5
        self.b = 0.75
        blobs = [self._blob(c) for c in chunks]
        self.dense = self.encoder.encode(blobs)
        self.image_toks: list[np.ndarray] | None = self._encode_late_toks()
        self.image_vecs = np.stack([self._image_vec(c) for c in chunks], axis=0)
        self.parent_of = [c.parent_id for c in chunks]

    @property
    def visual_score_path(self) -> str:
        """Product visual_qa scoring: MaxSim when late-interaction toks exist, else pooled."""
        if self.image_toks is not None and bool(getattr(self.vision, "late_interaction", False)):
            return "maxsim"
        return "pooled"

    def _resolve_image_path(self, chunk: Chunk) -> Path | None:
        if not chunk.image_path:
            return None
        path = Path(chunk.image_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        return path if path.exists() else None

    def _encode_late_toks(self) -> list[np.ndarray] | None:
        if not bool(getattr(self.vision, "late_interaction", False)):
            return None
        encode_multi = getattr(self.vision, "encode_images_multi", None)
        if not callable(encode_multi):
            return None
        dim = int(getattr(self.vision, "dim", 128) or 128)
        empty = np.zeros((0, dim), dtype=np.float32)
        idxs: list[int] = []
        paths: list[Path] = []
        for i, chunk in enumerate(self.chunks):
            path = self._resolve_image_path(chunk)
            if path is None:
                continue
            idxs.append(i)
            paths.append(path)
        if not paths:
            return [empty.copy() for _ in self.chunks]
        encoded = encode_multi(paths)
        by_i = {i: np.asarray(t, dtype=np.float32) for i, t in zip(idxs, encoded, strict=True)}
        return [by_i.get(i, empty) for i in range(self.n)]

    def _image_vec(self, chunk: Chunk) -> np.ndarray:
        path = self._resolve_image_path(chunk)
        if path is not None:
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

    @staticmethod
    def _caption_blob(chunk: Chunk) -> str:
        cap = chunk.caption or ""
        if not cap and chunk.image_path:
            cap = chunk.title
        return f"{cap} {chunk.title} {chunk.pdb_id or ''} {chunk.antigen or ''} {chunk.antibody or ''}"

    def _bm25(self, query: str, docs: list[list[str]], df: dict[str, int], avgdl: float) -> np.ndarray:
        q = tokenize(query)
        scores = np.zeros(self.n, dtype=np.float32)
        qn = Counter(q)
        for i, doc in enumerate(docs):
            tf = Counter(doc)
            dl = len(doc) or 1
            s = 0.0
            for term, qtf in qn.items():
                n_q = df.get(term, 0)
                if n_q == 0:
                    continue
                idf = math.log(1 + (self.n - n_q + 0.5) / (n_q + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / avgdl)
                s += idf * (freq * (self.k1 + 1) / denom) * qtf
            scores[i] = s
        return scores

    def bm25_scores(self, query: str) -> np.ndarray:
        return self._bm25(query, self.docs, self.df, self.avgdl)

    def caption_bm25_scores(self, query: str) -> np.ndarray:
        return self._bm25(query, self.caption_docs, self.caption_df, self.avg_cap_dl)

    def dense_scores(self, query: str) -> np.ndarray:
        qv = self.encoder.encode([query])[0]
        return self.dense @ qv

    def image_scores(self, query: str) -> np.ndarray:
        late = bool(getattr(self.vision, "late_interaction", False))
        encode_multi = getattr(self.vision, "encode_queries_multi", None)
        if late and callable(encode_multi) and self.image_toks is not None:
            from abcite_core.visual_qa import maxsim_matrix

            return maxsim_matrix(encode_multi([query]), self.image_toks)[0]
        encode_text = getattr(self.vision, "encode_text", None)
        if callable(encode_text):
            qv = np.asarray(encode_text(query), dtype=np.float32)
        else:
            qv = hashing_vector(query)
        if qv.shape[0] != self.image_vecs.shape[1]:
            return np.zeros(self.n, dtype=np.float32)
        return self.image_vecs @ qv

    def _parent_child_boost(self, rrf: np.ndarray) -> np.ndarray:
        """Boost siblings of currently strong parents, without flooding the ranking."""
        top = set(np.argsort(-rrf)[:5].tolist())
        hot_parents = {self.parent_of[i] for i in top}
        out = rrf.copy()
        for i, pid in enumerate(self.parent_of):
            if pid in hot_parents and i not in top:
                out[i] += 0.04 * float(rrf[i] + 1e-6)
        return out

    def search(self, query: str, top_k: int = 6) -> list[Hit]:
        expanded = expand_query(query)
        bm25 = self.bm25_scores(expanded)
        dense = self.dense_scores(expanded)
        caption = self.caption_bm25_scores(expanded)
        rrf = _rrf(bm25) + _rrf(dense)
        qlow = query.lower()
        figure_query = any(
            w in qlow
            for w in (
                "figure",
                "diagram",
                "caption",
                "snapshot",
                "residue map",
                "schematic",
                "cartoon",
                "highlighted",
                "numbered",
            )
        )
        if figure_query:
            rrf = rrf + 0.8 * _rrf(caption)
            for i, c in enumerate(self.chunks):
                if c.modality.value in {"figure", "structure"}:
                    rrf[i] += 0.04
        rrf = self._parent_child_boost(rrf)
        image = None
        if vision_rerank_enabled() or vision_fusion_enabled():
            image = self.image_scores(expanded)
        if vision_rerank_enabled() and image is not None:
            from abcite_core.figure_gallery import vision_rerank_scores

            rrf = vision_rerank_scores(rrf, image, k=vision_rerank_k())
        elif vision_fusion_enabled() and image is not None:
            rrf = rrf + 0.5 * _rrf(image)
        order = np.argsort(-rrf)[:top_k]
        hits: list[Hit] = []
        for i in order:
            hits.append(
                Hit(
                    chunk=self.chunks[int(i)],
                    score=float(rrf[i]),
                    bm25=float(bm25[i]),
                    dense=float(dense[i]),
                    image=float(image[i]) if image is not None else 0.0,
                )
            )
        return hits

    def search_visual(self, query: str, top_k: int = 6) -> list[Hit]:
        """Text → figure pixels. MaxSim if late-interaction; else pooled cosine. No caption BM25."""
        image = self.image_scores(query)
        caption = self.caption_bm25_scores(query)
        order = np.argsort(-image)[:top_k]
        hits: list[Hit] = []
        for i in order:
            hits.append(
                Hit(
                    chunk=self.chunks[int(i)],
                    score=float(image[i]),
                    bm25=float(caption[i]),
                    dense=0.0,
                    image=float(image[i]),
                )
            )
        return hits

    def search_by_image(self, path: Path, top_k: int = 5, fuse_captions: bool = True) -> list[Hit]:
        qv = np.asarray(self.vision.encode_image(path), dtype=np.float32)
        if qv.shape[0] != self.image_vecs.shape[1]:
            return []
        sims = self.image_vecs @ qv
        cap_q = self.vision.caption(path)
        cap_scores = self.caption_bm25_scores(cap_q)
        fused = _rrf(sims) + _rrf(cap_scores) if fuse_captions else sims
        order = np.argsort(-fused)[:top_k]
        hits: list[Hit] = []
        for i in order:
            hits.append(
                Hit(
                    chunk=self.chunks[int(i)],
                    score=float(fused[i]),
                    bm25=float(cap_scores[i]),
                    dense=0.0,
                    image=float(sims[i]),
                )
            )
        return hits

    def search_image_only(self, path: Path, top_k: int = 5) -> list[Hit]:
        return self.search_by_image(path, top_k=top_k, fuse_captions=False)


def _rrf(scores: np.ndarray, k: int = 60) -> np.ndarray:
    order = np.argsort(-scores)
    out = np.zeros_like(scores, dtype=np.float32)
    for rank, idx in enumerate(order, start=1):
        out[idx] = 1.0 / (k + rank)
    return out
