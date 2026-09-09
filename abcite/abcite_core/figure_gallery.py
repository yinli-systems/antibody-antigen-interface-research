"""Image gallery for multimodal figure retrieval (not the 14-SVG cite-id demo).

Image vectors are CLIP/hash pixels only — captions and PDB ids never enter encode_image.
Caption-as-query uses stripped captions as text queries (SciMMIR-style pair retrieval).
Unique-gold visual questions stay caption-free and PDB-free.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from abcite_core.holdout import strip_accession_tokens, strip_caption_leak, visual_query_ok
from abcite_core.vision import hashing_vector
from abcite_core.vision_flags import DEFAULT_RERANK_K

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
    href: str = ""


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9_\-]+", text.lower())


def load_figure_records(prefer_cache: bool = True) -> list[FigureRec]:
    cache_meta = FIG_ROOT / "cache" / "metadata.jsonl"
    fixture_meta = FIG_ROOT / "fixture" / "metadata.jsonl"
    meta = cache_meta if prefer_cache and cache_meta.exists() else fixture_meta
    if not meta.exists():
        raise FileNotFoundError(
            "No figure metadata. Run: python evals/download_figures.py --max 1200\n"
            f"looked for {cache_meta} and {fixture_meta}"
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
                href=str(raw.get("href") or ""),
            )
        )
    if not recs:
        raise FileNotFoundError(f"metadata at {meta} has no readable images")
    return recs


def fix_to_gallery_ids(recs: list[FigureRec]) -> dict[str, str]:
    """Map committed fixture `fix-*` ids onto gallery ids via href, then caption."""
    fixture_meta = FIG_ROOT / "fixture" / "metadata.jsonl"
    if not fixture_meta.exists():
        return {}
    href_to_gid: dict[str, str] = {}
    cap_to_gids: dict[str, list[str]] = defaultdict(list)
    for rec in recs:
        if rec.href:
            href_to_gid[rec.href] = rec.figure_id
        cap = (rec.caption or "").strip()
        if cap:
            cap_to_gids[cap].append(rec.figure_id)
    alias: dict[str, str] = {}
    for line in fixture_meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        fid = str(raw.get("figure_id") or "")
        href = str(raw.get("href") or "")
        cap = str(raw.get("caption") or "").strip()
        if href and href in href_to_gid:
            alias[fid] = href_to_gid[href]
        elif cap and cap_to_gids.get(cap):
            alias[fid] = cap_to_gids[cap][0]
    return alias


class FigureGallery:
    def __init__(self, recs: list[FigureRec], vision) -> None:
        self.recs = recs
        self.vision = vision
        self.ids = [r.figure_id for r in recs]
        self.captions = [r.caption for r in recs]
        self.cap_docs = [_tokenize(strip_accession_tokens(c)) for c in self.captions]
        self.df: dict[str, int] = defaultdict(int)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(self.cap_docs):
            tf = Counter(doc)
            for t, f in tf.items():
                self.df[t] += 1
                self.postings[t].append((i, f))
        self.n = len(recs)
        self.avgdl = sum(len(d) for d in self.cap_docs) / max(1, self.n)
        self.dl = [len(d) or 1 for d in self.cap_docs]
        self.image_vecs = self._load_or_encode()

    def _cache_key(self) -> Path:
        name = getattr(self.vision, "name", "enc")
        return FIG_ROOT / "cache" / f"embeddings-{name}-n{self.n}.npy"

    def _committed_key(self) -> Path:
        name = getattr(self.vision, "name", "enc")
        return ROOT / "evals" / f"embeddings-{name}-n{self.n}.npy"

    def _try_load_cache(self, cache: Path) -> np.ndarray | None:
        ids_path = cache.with_suffix(".ids.json")
        if not (cache.exists() and ids_path.exists()):
            return None
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if ids != self.ids:
            return None
        arr = np.load(cache)
        if arr.shape[0] != self.n:
            return None
        return arr.astype(np.float32)

    def _cached_by_id(self) -> dict[str, np.ndarray]:
        """Reuse any same-encoder npy (resume 1196 → 16k without re-encoding)."""
        name = getattr(self.vision, "name", "enc")
        found: dict[str, np.ndarray] = {}
        candidates = [
            *sorted((FIG_ROOT / "cache").glob(f"embeddings-{name}-n*.npy")),
            *sorted((ROOT / "evals").glob(f"embeddings-{name}-n*.npy")),
        ]
        candidates.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        for cache in candidates:
            ids_path = cache.with_suffix(".ids.json")
            if not ids_path.exists():
                continue
            try:
                ids = json.loads(ids_path.read_text(encoding="utf-8"))
                arr = np.load(cache)
            except Exception:
                continue
            if arr.shape[0] != len(ids):
                continue
            for i, fid in enumerate(ids):
                if fid not in found:
                    found[fid] = np.asarray(arr[i], dtype=np.float32)
        return found

    def _encode_paths(self, paths: list[Path]) -> np.ndarray:
        encode_batch = getattr(self.vision, "encode_images", None)
        if callable(encode_batch) and paths:
            return np.asarray(encode_batch(paths), dtype=np.float32)
        encode_image = self.vision.encode_image
        vecs = [np.asarray(encode_image(p), dtype=np.float32) for p in paths]
        if not vecs:
            dim = int(getattr(self.vision, "dim", 256) or 256)
            return np.zeros((0, dim), dtype=np.float32)
        return np.stack(vecs, axis=0)

    def _load_or_encode(self) -> np.ndarray:
        exact = self._try_load_cache(self._cache_key())
        if exact is None:
            exact = self._try_load_cache(self._committed_key())
        if exact is not None:
            return exact
        by_id = self._cached_by_id()
        dim: int | None = None
        if by_id:
            dim = int(next(iter(by_id.values())).shape[0])
        missing_idx = [i for i, fid in enumerate(self.ids) if fid not in by_id]
        checkpoint = bool(self.recs) and "fixture" not in str(self.recs[0].path)
        if missing_idx:
            name = getattr(self.vision, "name", "enc")
            print(
                f"encoding {len(missing_idx)}/{self.n} images "
                f"encoder={name}",
                flush=True,
            )
            chunk = 1024
            if dim is None:
                dim = int(getattr(self.vision, "dim", 256) or 256)
            for start in range(0, len(missing_idx), chunk):
                sl = missing_idx[start : start + chunk]
                paths = [self.recs[i].path for i in sl]
                encoded = self._encode_paths(paths)
                dim = int(encoded.shape[1]) if encoded.size else dim
                for row, i in enumerate(sl):
                    by_id[self.ids[i]] = encoded[row]
                if checkpoint:
                    done_ids = [fid for fid in self.ids if fid in by_id]
                    ck = FIG_ROOT / "cache" / f"embeddings-{name}-n{len(done_ids)}.npy"
                    np.save(ck, np.stack([by_id[fid] for fid in done_ids], axis=0))
                    ck.with_suffix(".ids.json").write_text(
                        json.dumps(done_ids), encoding="utf-8"
                    )
                    print(f"  checkpoint {len(done_ids)}/{self.n}", flush=True)
        if dim is None:
            dim = int(getattr(self.vision, "dim", 256) or 256)
        arr = np.stack(
            [by_id.get(fid, np.zeros(dim, dtype=np.float32)) for fid in self.ids],
            axis=0,
        ).astype(np.float32)
        cache = self._cache_key()
        cache.parent.mkdir(parents=True, exist_ok=True)
        if self.recs and "fixture" not in str(self.recs[0].path):
            np.save(cache, arr)
            cache.with_suffix(".ids.json").write_text(json.dumps(self.ids), encoding="utf-8")
        return arr

    def _bm25(self, query: str) -> np.ndarray:
        q = _tokenize(strip_accession_tokens(query))
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

    def encode_queries(self, questions: list[str], *, holdout_captions: bool = False) -> np.ndarray:
        texts: list[str] = []
        for i, question in enumerate(questions):
            cap = self.captions[i] if holdout_captions and i < len(self.captions) else None
            texts.append(strip_accession_tokens(strip_caption_leak(question, cap)))
        encode_batch = getattr(self.vision, "encode_texts", None)
        if callable(encode_batch):
            return np.asarray(encode_batch(texts), dtype=np.float32)
        encode_text = getattr(self.vision, "encode_text", None)
        vecs = []
        for text in texts:
            if callable(encode_text):
                vecs.append(np.asarray(encode_text(text), dtype=np.float32))
            else:
                vecs.append(hashing_vector(text))
        if not vecs:
            return np.zeros((0, self.image_vecs.shape[1]), dtype=np.float32)
        return np.stack(vecs, axis=0)

    def clip_score_matrix(self, questions: list[str]) -> np.ndarray:
        qv = self.encode_queries(questions, holdout_captions=False)
        if qv.size == 0 or qv.shape[1] != self.image_vecs.shape[1]:
            return np.zeros((len(questions), self.n), dtype=np.float32)
        return qv @ self.image_vecs.T

    def _query_vec(self, question: str, caption_for_holdout: str | None) -> np.ndarray:
        text = strip_accession_tokens(strip_caption_leak(question, caption_for_holdout))
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

    def vision_rerank_scores(
        self, question: str, k: int = DEFAULT_RERANK_K, text_scores: np.ndarray | None = None
    ) -> np.ndarray:
        text = self.caption_bm25_scores(question) if text_scores is None else text_scores
        img = self.clip_image_only_scores(question)
        return vision_rerank_scores(text, img, k=k)

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


def vision_rerank_scores(
    text_scores: np.ndarray, image_scores: np.ndarray, k: int = DEFAULT_RERANK_K
) -> np.ndarray:
    """Reorder BM25 top-K by CLIP cosine; keep ranks >K in original text order.

    Gold outside the text top-K cannot enter R@1/5/10 via rerank. That is the point.
    """
    n = int(text_scores.shape[0])
    k = max(1, min(int(k), n))
    text_order = np.argsort(-text_scores, kind="mergesort")
    top = text_order[:k]
    rest = text_order[k:]
    top_reranked = top[np.argsort(-image_scores[top], kind="mergesort")]
    new_order = np.concatenate([top_reranked, rest]) if rest.size else top_reranked
    out = np.empty(n, dtype=np.float32)
    for rank, idx in enumerate(new_order):
        out[int(idx)] = float(n - rank)
    return out


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    g = set(gold)
    return 1.0 if any(x in g for x in ranked[:k]) else 0.0


def mean_text_gap(gallery: FigureGallery, questions: list[str]) -> float:
    if not questions:
        return 0.0
    mat = gallery.clip_score_matrix(questions)
    gaps: list[float] = []
    for row in mat:
        order = np.argsort(-row)
        if len(order) < 2:
            continue
        gaps.append(float(row[order[0]] - row[order[1]]))
    return float(sum(gaps) / len(gaps)) if gaps else 0.0


def caption_as_query_items(recs: list[FigureRec], min_tokens: int = 6) -> list[dict]:
    """SciMMIR-style pair retrieval: query = accession-stripped caption, gold = that figure."""
    items: list[dict] = []
    for rec in recs:
        q = strip_accession_tokens(rec.caption or "")
        if len(_tokenize(q)) < min_tokens:
            continue
        items.append(
            {
                "id": f"cap-{rec.figure_id}",
                "question": q,
                "gold_figure_ids": [rec.figure_id],
                "protocol": "caption_as_query",
                "pmcid": rec.pmcid,
            }
        )
    return items


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
