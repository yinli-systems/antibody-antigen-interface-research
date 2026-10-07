"""Image gallery for multimodal figure retrieval (not the 14-SVG cite-id demo).

Image vectors are CLIP/hash pixels only — captions and PDB ids never enter encode_image.
Caption-as-query uses stripped captions as text queries (SciMMIR-style pair retrieval).
Unique-gold visual questions stay caption-free and PDB-free.
"""

from __future__ import annotations

import json
import math
import os
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


def pmc_gallery_flag() -> str:
    return (os.environ.get("ABCITE_PMC_GALLERY") or "0").strip()


def pmc_gallery_requested() -> bool:
    """Opt-in 16k PMC serving. Default off so CI stays on the 14-SVG fixture."""
    flag = pmc_gallery_flag().lower()
    if flag in {"0", "false", "no", "off", ""}:
        return False
    return True


def _encoder_npz_stems(vision) -> list[str]:
    from abcite_core.colpali_encoder import encoder_cache_name

    name = str(getattr(vision, "name", "") or "") if vision is not None else ""
    stem = encoder_cache_name(name) if name else ""
    if stem in {"colqwen2.5", "colqwen2"}:
        return [stem]
    return ["colqwen2.5", "colqwen2"]


def _discover_pmc_npz(vision, root: Path) -> Path | None:
    dirs = [root / "evals", root / "evals" / "figures" / "cache"]
    for stem in _encoder_npz_stems(vision):
        found: list[Path] = []
        for folder in dirs:
            if folder.is_dir():
                found.extend(folder.glob(f"embeddings-{stem}-n*.npz"))
        if not found:
            continue
        exact = [p for p in found if p.name == f"embeddings-{stem}-n15949.npz"]
        if exact:
            return exact[0]
        found.sort(key=lambda p: p.stat().st_size if p.is_file() else 0, reverse=True)
        return found[0]
    return None


def resolve_pmc_gallery_npz(vision=None, *, root: Path | None = None) -> Path | None:
    """npz for product visual_qa. None → keep the fixture HybridIndex."""
    if not pmc_gallery_requested():
        return None
    root = Path(root or ROOT)
    explicit = (os.environ.get("ABCITE_PMC_GALLERY_NPZ") or "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    flag = pmc_gallery_flag()
    as_path = Path(flag)
    if flag.lower() not in {"1", "true", "yes", "on", "auto"} and as_path.suffix.lower() == ".npz":
        return as_path if as_path.is_file() else None
    return _discover_pmc_npz(vision, root)


def read_late_npz(npz: Path) -> tuple[list[str], list[np.ndarray]]:
    npz = Path(npz)
    ids_path = npz.with_name(npz.stem + ".ids.json")
    if not npz.is_file() or not ids_path.is_file():
        raise FileNotFoundError(f"late npz needs {npz.name} and {ids_path.name}")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    blob = np.load(npz, allow_pickle=True)
    toks = [np.asarray(blob[f"t{i}"], dtype=np.float32) for i in range(len(ids))]
    if len(toks) != len(ids):
        raise ValueError(f"late npz token count {len(toks)} != ids {len(ids)}")
    return [str(i) for i in ids], toks


def _figure_meta_index() -> dict[str, dict]:
    """figure_id → metadata row. Does not require the jpg to exist."""
    found: dict[str, dict] = {}
    for meta in (FIG_ROOT / "cache" / "metadata.jsonl", FIG_ROOT / "fixture" / "metadata.jsonl"):
        if not meta.is_file():
            continue
        for line in meta.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            fid = str(raw.get("figure_id") or "")
            if fid and fid not in found:
                found[fid] = raw
    return found


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def pmc_image_roots() -> list[Path]:
    """On-disk jpg roots. npz retrieval does not require these files."""
    roots: list[Path] = []
    env = (os.environ.get("ABCITE_PMC_IMAGE_DIR") or os.environ.get("ABCITE_FIGURES_CACHE") or "").strip()
    if env:
        roots.append(Path(env).expanduser())
    roots.extend(
        [
            FIG_ROOT / "cache" / "images",
            FIG_ROOT / "cache",
            Path("/workspace/abcite/evals/figures/cache/images"),
            Path("/workspace/abcite/evals/figures/cache"),
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def resolve_figure_pixels(figure_id: str, hint: Path | str | None = None) -> Path | None:
    """Find jpg/png for a gallery id. None → heuristic VQA must abstain."""
    candidates: list[Path] = []
    if hint is not None and str(hint) not in {"", ".", "None"}:
        p = Path(str(hint))
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(ROOT / p)
            candidates.append(FIG_ROOT / p)
    fid = str(figure_id or "").strip()
    if fid:
        for root in pmc_image_roots():
            for ext in _IMAGE_EXTS:
                candidates.append(root / f"{fid}{ext}")
            candidates.append(root / "images" / f"{fid}.jpg")
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.is_file() and cand.stat().st_size > 32:
                return cand
        except OSError:
            continue
    return None


def recs_for_npz_ids(ids: list[str]) -> list[FigureRec]:
    meta = _figure_meta_index()
    recs: list[FigureRec] = []
    for fid in ids:
        raw = meta.get(fid, {})
        rel = str(raw.get("path") or "")
        hint = FIG_ROOT / rel if rel else None
        path = resolve_figure_pixels(fid, hint) or hint or Path(".")
        recs.append(
            FigureRec(
                figure_id=fid,
                path=path,
                caption=str(raw.get("caption") or ""),
                pmcid=raw.get("pmcid"),
                split=str(raw.get("split") or ""),
                visual_class=str(raw.get("visual_class") or "scientific figure panel"),
                source=str(raw.get("source") or ""),
                href=str(raw.get("href") or ""),
            )
        )
    return recs


def load_late_npz_gallery(npz: Path, vision) -> FigureGallery:
    """Score MaxSim from a cached npz. Never encodes or downloads figures."""
    ids, toks = read_late_npz(npz)
    recs = recs_for_npz_ids(ids)
    gal = FigureGallery(recs, vision, image_toks=toks)
    gal.npz_path = Path(npz)
    return gal


def load_product_pmc_gallery(vision, *, root: Path | None = None) -> FigureGallery | None:
    """Opt-in PMC gallery for /query visual_qa. Same MaxSim as the eval receipt."""
    if not bool(getattr(vision, "late_interaction", False)):
        return None
    if not callable(getattr(vision, "encode_queries_multi", None)):
        return None
    npz = resolve_pmc_gallery_npz(vision, root=root)
    if npz is None:
        return None
    gal = load_late_npz_gallery(npz, vision)
    print(
        f"product PMC gallery npz={npz.name} n={gal.n} MaxSim (not the 14-SVG fixture)",
        flush=True,
    )
    return gal


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
    def __init__(
        self,
        recs: list[FigureRec],
        vision,
        image_toks: list[np.ndarray] | None = None,
    ) -> None:
        self.recs = recs
        self.vision = vision
        self.npz_path: Path | None = None
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
        if image_toks is not None:
            if len(image_toks) != self.n:
                raise ValueError(f"image_toks {len(image_toks)} != gallery n={self.n}")
            self.image_toks = [np.asarray(t, dtype=np.float32) for t in image_toks]
            self.image_vecs = self._means_from_toks(self.image_toks)
        else:
            self.image_toks = None
            self.image_vecs = self._load_or_encode()

    @property
    def visual_score_path(self) -> str:
        if self.image_toks is not None and bool(getattr(self.vision, "late_interaction", False)):
            return "maxsim"
        return "pooled"

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
        if bool(getattr(self.vision, "late_interaction", False)):
            return self._load_or_encode_late()
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

    def _late_shard_dir(self) -> Path:
        name = getattr(self.vision, "name", "colpali")
        return FIG_ROOT / "cache" / f"late-{name}"

    def _means_from_toks(self, toks: list[np.ndarray]) -> np.ndarray:
        dim = int(getattr(self.vision, "dim", 128) or 128)
        means: list[np.ndarray] = []
        for t in toks:
            if t is None or np.asarray(t).size == 0:
                means.append(np.zeros(dim, dtype=np.float32))
                continue
            v = np.asarray(t, dtype=np.float32).mean(axis=0)
            nrm = float(np.linalg.norm(v))
            means.append(v / nrm if nrm else v)
            dim = int(v.shape[0])
        return np.stack(means, axis=0).astype(np.float32)

    def _load_late_shards(self) -> dict[str, np.ndarray]:
        root = self._late_shard_dir()
        found: dict[str, np.ndarray] = {}
        if not root.is_dir():
            return found
        for npy in root.glob("*.npy"):
            found[npy.stem] = np.load(npy)
        return found

    def _save_late_shard(self, figure_id: str, toks: np.ndarray) -> None:
        root = self._late_shard_dir()
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / f"{figure_id}.npy", np.asarray(toks, dtype=np.float16))

    def _load_or_encode_late(self) -> np.ndarray:
        """ColPali/ColQwen2 multi-vector cache (per-id shards) plus mean-pooled vecs."""
        cache = self._cache_key().with_suffix(".npz")
        ids_path = cache.with_name(cache.stem + ".ids.json")
        encode_multi = getattr(self.vision, "encode_images_multi", None)
        if cache.exists() and ids_path.exists() and callable(encode_multi):
            try:
                ids = json.loads(ids_path.read_text(encoding="utf-8"))
                print(f"loading late npz {cache.name} n={len(ids)}", flush=True)
                blob = np.load(cache, allow_pickle=True)
                toks = [np.asarray(blob[f"t{i}"], dtype=np.float32) for i in range(len(ids))]
                if ids == self.ids and len(toks) == self.n:
                    self.image_toks = toks
                    print(f"late npz hit encoder={getattr(self.vision, 'name', '')}", flush=True)
                    return self._means_from_toks(toks)
                print(
                    f"late npz id mismatch cached={len(ids)} gallery={self.n}; shards fallback",
                    flush=True,
                )
            except Exception as exc:
                print(f"late npz failed ({type(exc).__name__}: {exc})", flush=True)
        print("loading late shards (slow if thousands of npy files)", flush=True)
        by_id = self._load_late_shards()
        if not callable(encode_multi):
            return self._encode_paths([r.path for r in self.recs])
        missing = [i for i, fid in enumerate(self.ids) if fid not in by_id]
        if missing:
            name = getattr(self.vision, "name", "colpali")
            print(
                f"encoding {len(missing)}/{self.n} images encoder={name} late-interaction",
                flush=True,
            )
            chunk = 16
            checkpoint = bool(self.recs) and "fixture" not in str(self.recs[0].path)
            for start in range(0, len(missing), chunk):
                sl = missing[start : start + chunk]
                paths = [self.recs[i].path for i in sl]
                encoded = encode_multi(paths, batch_size=1)
                for row, i in enumerate(sl):
                    vec = np.asarray(encoded[row], dtype=np.float32)
                    by_id[self.ids[i]] = vec
                    if checkpoint:
                        self._save_late_shard(self.ids[i], vec)
                print(f"  late {min(start + chunk, len(missing))}/{len(missing)}", flush=True)
        empty = np.zeros((0, int(getattr(self.vision, "dim", 128) or 128)), dtype=np.float32)
        toks = [np.asarray(by_id.get(fid, empty), dtype=np.float32) for fid in self.ids]
        self.image_toks = toks
        if self.recs and "fixture" not in str(self.recs[0].path) and len(by_id) == self.n:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache,
                **{f"t{i}": t.astype(np.float16) for i, t in enumerate(toks)},
            )
            ids_path.write_text(json.dumps(self.ids), encoding="utf-8")
        return self._means_from_toks(toks)

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
        late = bool(getattr(self.vision, "late_interaction", False))
        encode_multi = getattr(self.vision, "encode_queries_multi", None)
        if late and callable(encode_multi) and self.image_toks is not None:
            from abcite_core.visual_qa import maxsim_matrix

            return maxsim_matrix(encode_multi(questions), self.image_toks)
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
        late = bool(getattr(self.vision, "late_interaction", False))
        encode_multi = getattr(self.vision, "encode_queries_multi", None)
        if late and callable(encode_multi) and self.image_toks is not None:
            return self.clip_score_matrix([question])[0]
        qv = self._query_vec(question, None)
        if qv.shape[0] != self.image_vecs.shape[1]:
            return np.zeros(self.n, dtype=np.float32)
        return self.image_vecs @ qv

    def _hits_from_scores(
        self,
        rank: np.ndarray,
        image: np.ndarray,
        caption: np.ndarray,
        top_k: int,
    ):
        from abcite_core.models import Chunk, Modality
        from abcite_core.retrieval import Hit

        order = np.argsort(-rank)[:top_k]
        hits: list[Hit] = []
        for i in order:
            rec = self.recs[int(i)]
            path = resolve_figure_pixels(rec.figure_id, rec.path)
            hits.append(
                Hit(
                    chunk=Chunk(
                        chunk_id=rec.figure_id,
                        parent_id=rec.pmcid or rec.figure_id,
                        modality=Modality.FIGURE,
                        title=rec.figure_id,
                        text=rec.caption or rec.figure_id,
                        caption=rec.caption or "",
                        source_url=rec.href or "",
                        source_label=rec.source or rec.pmcid or rec.figure_id,
                        image_path=str(path) if path else None,
                    ),
                    score=float(rank[i]),
                    bm25=float(caption[i]),
                    dense=0.0,
                    image=float(image[i]),
                )
            )
        return hits

    def search_visual(self, query: str, top_k: int = 6):
        """Text → figure MaxSim (same scorer as eval). Hits use gallery figure ids."""
        image = self.clip_image_only_scores(query)
        caption = self.caption_bm25_scores(query)
        return self._hits_from_scores(image, image, caption, top_k)

    def search_by_image(self, path: Path, top_k: int = 5, fuse_captions: bool = True):
        """Pixels→pixels. MaxSim when late toks exist; else pooled. Same gallery as visual_qa."""
        path = Path(path)
        late = bool(getattr(self.vision, "late_interaction", False))
        encode_multi = getattr(self.vision, "encode_images_multi", None)
        if late and callable(encode_multi) and self.image_toks is not None:
            from abcite_core.visual_qa import maxsim_matrix

            sims = maxsim_matrix(encode_multi([path]), self.image_toks)[0]
        else:
            qv = np.asarray(self.vision.encode_image(path), dtype=np.float32)
            if qv.shape[0] != self.image_vecs.shape[1]:
                return []
            sims = self.image_vecs @ qv
        cap_q = ""
        caption_fn = getattr(self.vision, "caption", None)
        if callable(caption_fn):
            cap_q = caption_fn(path) or ""
        cap_scores = self.caption_bm25_scores(cap_q)
        rank = (_rrf(sims) + _rrf(cap_scores)) if fuse_captions else sims
        return self._hits_from_scores(rank, sims, cap_scores, top_k)

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
