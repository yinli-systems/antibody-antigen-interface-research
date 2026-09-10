"""Optional ColPali / ColQwen2 late-interaction encoder (GPU).

Install: `pip install -e ".[colpali]"` (colpali-engine). Weights from HF cache
(`/workspace/hf-cache` on the 5090). Never required for CI.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from abcite_core.raster import load_rgb
from abcite_core.vision import LocalImageEncoder, VisionEncoder

log = logging.getLogger("abcite.colpali")

COLPALI_MODELS = (
    "vidore/colqwen2-v1.0",
    "vidore/colpali-v1.3",
    "vidore/colpali-v1.2",
    "vidore/colpali",
)


def _hf_home() -> Path:
    raw = os.environ.get("HF_HOME")
    if raw:
        return Path(raw)
    if Path("/workspace/hf-cache").is_dir():
        return Path("/workspace/hf-cache")
    return Path.home() / ".cache" / "huggingface"


def _hub_snapshot(repo_dir: str) -> Path | None:
    root = _hf_home() / "hub" / repo_dir / "snapshots"
    if not root.is_dir():
        return None
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    return snaps[0] if snaps else None


def _patched_lora_adapter(adapter_repo: str, base_repo: str, dest: Path) -> Path | None:
    """Point a LoRA adapter at a local base snapshot (offline-friendly)."""
    adapter = _hub_snapshot(f"models--{adapter_repo.replace('/', '--')}")
    base = _hub_snapshot(f"models--{base_repo.replace('/', '--')}")
    if adapter is None or base is None:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    import json
    import shutil

    for src in adapter.iterdir():
        target = dest / src.name
        if target.exists() or target.is_symlink():
            continue
        if src.is_dir():
            shutil.copytree(src, target, symlinks=True)
        else:
            try:
                os.symlink(src.resolve(), target)
            except OSError:
                shutil.copy2(src, target)
    cfg_path = dest / "adapter_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["base_model_name_or_path"] = str(base)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return dest


def _patched_colqwen2_adapter() -> Path | None:
    """Point the LoRA adapter at a local vidore/colqwen2-base snapshot (offline)."""
    dest = Path(os.environ.get("ABCITE_COLPALI_PATCH", "/tmp/colqwen2-local"))
    return _patched_lora_adapter("vidore/colqwen2-v1.0", "vidore/colqwen2-base", dest)


def _patched_colpali_v13_adapter() -> Path | None:
    """Point colpali-v1.3 LoRA at a local colpaligemma-3b-pt-448-base snapshot."""
    dest = Path(os.environ.get("ABCITE_COLPALI_V13_PATCH", "/tmp/colpali-v1.3-local"))
    return _patched_lora_adapter(
        "vidore/colpali-v1.3",
        "vidore/colpaligemma-3b-pt-448-base",
        dest,
    )


def _patched_colqwen25_adapter() -> Path | None:
    """Point colqwen2.5-v0.2 LoRA at a local vidore/colqwen2.5-base snapshot."""
    dest = Path(os.environ.get("ABCITE_COLQWEN25_PATCH", "/tmp/colqwen2.5-local"))
    return _patched_lora_adapter(
        "vidore/colqwen2.5-v0.2",
        "vidore/colqwen2.5-base",
        dest,
    )


def encoder_cache_name(model_id: str, tag: str = "") -> str:
    """Stable FigureGallery cache stem. Never reuse embeddings-colqwen2-n*.npz."""
    blob = f"{model_id} {tag}".lower()
    if "colqwen2.5" in blob or "colqwen2_5" in blob:
        return "colqwen2.5"
    if "colqwen2" in blob:
        return "colqwen2"
    if "colpali-v1.3" in blob or "colpali_v1.3" in blob:
        return "colpali-v1.3"
    if "colpali" in blob:
        return "colpali"
    return tag or "colpali"


def ensure_query_prefix(processor, default: str = "Query: ") -> str:
    """colpali-engine>=0.3.13 dropped the training-time prefix."""
    old = getattr(processor, "query_prefix", None)
    if old in (None, ""):
        processor.query_prefix = default
    return str(getattr(processor, "query_prefix"))


def colpali_forced_off() -> bool:
    return os.environ.get("ABCITE_COLPALI", "auto").lower() in {"0", "false", "no", "off"}


def colpali_requested() -> bool:
    return os.environ.get("ABCITE_COLPALI", "auto").lower() in {"1", "true", "yes", "on", "colpali"}


class ColPaliEncoder(VisionEncoder):
    """Multi-vector page/figure encoder. Mean-pool is only a HybridIndex fallback."""

    name = "colqwen2"
    dim = 128
    late_interaction = True

    def __init__(self, model=None, processor=None, device=None) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self._tok_cache: dict[str, np.ndarray] = {}

    def encode_image(self, path: Path) -> list[float]:
        toks = self.encode_images_multi([path])[0]
        if toks.size == 0:
            return [0.0] * self.dim
        vec = toks.mean(axis=0)
        n = float(np.linalg.norm(vec))
        return (vec / n).tolist() if n else vec.tolist()

    def encode_text(self, text: str) -> list[float]:
        toks = self.encode_queries_multi([text])[0]
        if toks.size == 0:
            return [0.0] * self.dim
        vec = toks.mean(axis=0)
        n = float(np.linalg.norm(vec))
        return (vec / n).tolist() if n else vec.tolist()

    def caption(self, path: Path) -> str:
        return LocalImageEncoder().caption(path)

    def encode_images(self, paths: list[Path], batch_size: int = 2) -> np.ndarray:
        toks = self.encode_images_multi(paths, batch_size=batch_size)
        rows = []
        for t in toks:
            if t.size == 0:
                rows.append(np.zeros(self.dim, dtype=np.float32))
                continue
            v = t.mean(axis=0)
            n = float(np.linalg.norm(v))
            rows.append(v / n if n else v)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(rows, axis=0)

    def encode_texts(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        toks = self.encode_queries_multi(texts, batch_size=batch_size)
        rows = []
        for t in toks:
            if t.size == 0:
                rows.append(np.zeros(self.dim, dtype=np.float32))
                continue
            v = t.mean(axis=0)
            n = float(np.linalg.norm(v))
            rows.append(v / n if n else v)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(rows, axis=0)

    def encode_images_multi(self, paths: list[Path], batch_size: int = 1) -> list[np.ndarray]:
        import torch

        out: list[np.ndarray] = [np.zeros((0, self.dim), dtype=np.float32) for _ in paths]
        pending_i: list[int] = []
        pending_p: list[Path] = []
        for i, path in enumerate(paths):
            key = str(Path(path).resolve())
            cached = self._tok_cache.get(key)
            if cached is not None:
                out[i] = cached
                continue
            pending_i.append(i)
            pending_p.append(path)
            if len(pending_p) >= batch_size:
                self._encode_image_batch(pending_i, pending_p, out)
                pending_i, pending_p = [], []
        if pending_p:
            self._encode_image_batch(pending_i, pending_p, out)
        return out

    def _encode_image_batch(
        self, indices: list[int], paths: list[Path], out: list[np.ndarray]
    ) -> None:
        import torch

        images = [load_rgb(p) for p in paths]
        batch = self.processor.process_images(images).to(self.device)
        with torch.no_grad():
            emb = self.model(**batch)
        if hasattr(emb, "cpu"):
            arr = emb.float().cpu().numpy()
        else:
            arr = np.asarray(emb)
        for row, path, idx in zip(arr, paths, indices, strict=True):
            vec = np.asarray(row, dtype=np.float32)
            n = np.linalg.norm(vec, axis=-1, keepdims=True)
            vec = np.divide(vec, n, out=np.zeros_like(vec), where=n > 0)
            self.dim = int(vec.shape[-1])
            self._tok_cache[str(Path(path).resolve())] = vec
            out[idx] = vec

    def encode_queries_multi(self, texts: list[str], batch_size: int = 8) -> list[np.ndarray]:
        import torch

        if not texts:
            return []
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch_txt = texts[i : i + batch_size]
            batch = self.processor.process_queries(batch_txt).to(self.device)
            with torch.no_grad():
                emb = self.model(**batch)
            if hasattr(emb, "cpu"):
                arr = emb.float().cpu().numpy()
            else:
                arr = np.asarray(emb)
            for row in arr:
                vec = np.asarray(row, dtype=np.float32)
                n = np.linalg.norm(vec, axis=-1, keepdims=True)
                vec = np.divide(vec, n, out=np.zeros_like(vec), where=n > 0)
                self.dim = int(vec.shape[-1])
                chunks.append(vec)
        return chunks


def _class_pairs() -> list[tuple[str, object, object]]:
    """(name_substr, Model, Processor) in preference order."""
    pairs: list[tuple[str, object, object]] = []
    try:
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        pairs.append(("colqwen2.5", ColQwen2_5, ColQwen2_5_Processor))
    except Exception:
        pass
    try:
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        pairs.append(("colqwen2", ColQwen2, ColQwen2Processor))
    except Exception:
        pass
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor

        pairs.append(("colpali", ColPali, ColPaliProcessor))
    except Exception:
        pass
    return pairs


def try_load_colpali() -> tuple[VisionEncoder | None, str | None]:
    """Best-effort ColQwen2 / ColPali. Never raises."""
    if colpali_forced_off():
        return None, "ColPali skipped: ABCITE_COLPALI=0"
    try:
        import torch
    except Exception as exc:
        return None, f"ColPali skipped: {type(exc).__name__}: {exc}"
    pairs = _class_pairs()
    if not pairs:
        return None, "ColPali skipped: colpali-engine models not importable"

    os.environ.setdefault("HF_HOME", str(_hf_home()))
    os.environ.setdefault("HF_HUB_CACHE", str(_hf_home() / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf_home() / "hub"))
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
    local_only = offline or os.environ.get("ABCITE_COLPALI_LOCAL", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last: Exception | None = None
    names: list[tuple[str, str]] = []
    forced = (os.environ.get("ABCITE_COLPALI_MODEL") or "").strip()
    if forced:
        if "colpali-v1.3" in forced.lower():
            v13 = _patched_colpali_v13_adapter()
            v13_snap = _hub_snapshot("models--vidore--colpali-v1.3")
            if v13 is not None and v13_snap is not None:
                names.append((str(v13), str(v13_snap)))
        if "colqwen2.5" in forced.lower() or "colqwen2_5" in forced.lower():
            q25 = _patched_colqwen25_adapter()
            q25_snap = _hub_snapshot("models--vidore--colqwen2.5-v0.2")
            if q25 is not None and q25_snap is not None:
                names.append((str(q25), str(q25_snap)))
        elif "colqwen2" in forced.lower():
            patched = _patched_colqwen2_adapter()
            adapter_snap = _hub_snapshot("models--vidore--colqwen2-v1.0")
            if patched is not None and adapter_snap is not None:
                names.append((str(patched), str(adapter_snap)))
        names.append((forced, forced))
    else:
        patched = _patched_colqwen2_adapter()
        adapter_snap = _hub_snapshot("models--vidore--colqwen2-v1.0")
        if patched is not None and adapter_snap is not None:
            # Model from patched adapter (local base path); processor from the
            # original snapshot so chat-template relative symlinks resolve.
            names.append((str(patched), str(adapter_snap)))
        v13 = _patched_colpali_v13_adapter()
        v13_snap = _hub_snapshot("models--vidore--colpali-v1.3")
        if v13 is not None and v13_snap is not None:
            names.append((str(v13), str(v13_snap)))
        for hub_name in COLPALI_MODELS:
            names.append((hub_name, hub_name))
    seen: set[tuple[str, str]] = set()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for name, proc_name in names:
        if not name or (name, proc_name) in seen:
            continue
        seen.add((name, proc_name))
        low = name.lower() + " " + proc_name.lower()
        fl = forced.lower()
        if fl and ("colpali-v1.3" in fl or "colpali_v1.3" in fl):
            # Forced v1.3 must not fall through to ColQwen2 (would poison the v1.3 npz).
            ordered = [p for p in pairs if p[0] == "colpali"]
        elif fl and ("colqwen2.5" in fl or "colqwen2_5" in fl):
            ordered = [p for p in pairs if p[0] == "colqwen2.5"]
        elif fl and "colqwen2" in fl:
            ordered = [p for p in pairs if p[0] == "colqwen2"]
        elif "colqwen2" in low and "colqwen2.5" not in low and "colqwen2_5" not in low:
            ordered = [p for p in pairs if p[0] == "colqwen2"] + [p for p in pairs if p[0] != "colqwen2"]
        else:
            ordered = [p for p in pairs if p[0] in low] + [p for p in pairs if p[0] not in low]
        if not ordered:
            last = RuntimeError(f"no matching late-interaction class for {name!r} forced={forced!r}")
            continue
        for tag, model_cls, proc_cls in ordered:
            try:
                model = model_cls.from_pretrained(
                    name,
                    torch_dtype=dtype,
                    device_map=str(device),
                    local_files_only=local_only,
                ).eval()
                processor = proc_cls.from_pretrained(proc_name, local_files_only=local_only)
                prefix = ensure_query_prefix(processor)
                enc = ColPaliEncoder(model, processor, device)
                enc.name = encoder_cache_name(forced or name, tag)
                print(
                    f"loaded late-interaction encoder={enc.name} src={name} "
                    f"device={device} query_prefix={prefix!r}",
                    flush=True,
                )
                log.info("loaded ColPali encoder %s (%s) on %s", name, tag, device)
                return enc, None
            except Exception as exc:
                last = exc
                log.warning("ColPali load %s via %s failed: %s", name, tag, exc)
                continue
    return None, f"ColPali skipped: {last}"
