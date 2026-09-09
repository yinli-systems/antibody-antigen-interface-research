"""Real CLIP encoder (OpenCLIP ViT-B-32) with SVG→RGB rasterization.

Install: `pip install -e ".[clip]"` then `python -m abcite_core.download_clip`.
HybridIndex auto-uses CLIP when open_clip is importable **and** weights are cached,
unless ABCITE_CLIP=0. ABCITE_CLIP=1 downloads if needed and fails hard on error.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

from abcite_core.raster import load_rgb
from abcite_core.vision import LocalImageEncoder, VisionEncoder

log = logging.getLogger("abcite.clip")

CLIP_NAME = "clip-vit-b-32"
CLIP_ARCH = "ViT-B-32"
CLIP_TAG = "openai"
BIOMEDCLIP_STUB = "biomedclip-stub"
CLIP_MIN_BYTES = 340_000_000
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
CLIP_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    f"{CLIP_SHA256}/ViT-B-32.pt"
)
_MODEL = None
_VEC_CACHE: dict[str, list[float]] = {}


def clip_stack_available() -> bool:
    try:
        import open_clip  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


def clip_forced_off() -> bool:
    return os.environ.get("ABCITE_CLIP", "auto").lower() in {"0", "false", "no", "tiny"}


def clip_forced_on() -> bool:
    return os.environ.get("ABCITE_CLIP", "auto").lower() in {"1", "true", "yes", "clip"}


def clip_model_name() -> str:
    return (os.environ.get("ABCITE_CLIP_MODEL") or "openclip").strip().lower()


def biomedclip_requested() -> bool:
    return clip_model_name() in {"biomedclip", "biomed-clip", "pmc-clip"}


BIOMEDCLIP_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BIOMEDCLIP_HF_NAMES = (f"hf-hub:{BIOMEDCLIP_REPO}",)
BIOMEDCLIP_TEXT_REPOS = (
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
)
_BIOMEDCLIP_FILES = (
    "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.bin",
    "biomedclip.pt",
    "open_clip_pytorch_model_biomedclip.bin",
    "open_clip_pytorch_model.bin",
    "open_clip_model.safetensors",
)


def _hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    for path in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path("/root/.cache/huggingface/token"),
        Path("/workspace/hf-cache/token"),
        Path("/workspace/hf/token"),
        Path("/workspace/.huggingface/token"),
    ):
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
    return None


def _hf_home() -> Path:
    raw = os.environ.get("HF_HOME")
    if raw:
        return Path(raw)
    if Path("/workspace/hf-cache").is_dir():
        return Path("/workspace/hf-cache")
    if Path("/workspace/hf").is_dir():
        return Path("/workspace/hf")
    if Path("/workspace").is_dir():
        return Path("/workspace/hf-cache")
    return Path.home() / ".cache" / "huggingface"


def _biomedclip_cached_files() -> list[Path]:
    found: list[Path] = []
    roots = [clip_cache_dir(), _hf_home(), _hf_home() / "hub"]
    for root in roots:
        if not root.exists():
            continue
        for name in _BIOMEDCLIP_FILES:
            cand = root / name
            if cand.is_file() and cand.stat().st_size > 50_000_000:
                found.append(cand)
        for match in root.glob("**/models--microsoft--BiomedCLIP*/**/*"):
            if match.is_file() and match.stat().st_size > 50_000_000:
                found.append(match)
    return found


def biomedclip_weights_present() -> bool:
    if _biomedclip_cached_files():
        return True
    return os.environ.get("ABCITE_BIOMEDCLIP_DOWNLOAD", "").lower() in {"1", "true", "yes"}


def biomedclip_skip_reason() -> str | None:
    if not biomedclip_requested():
        return None
    if biomedclip_weights_present():
        return None
    return (
        "BiomedCLIP skipped: ABCITE_CLIP_MODEL=biomedclip but weights are not in "
        f"{clip_cache_dir()}. No download. Using OpenCLIP if cached, else local encoder."
    )


def clip_cache_dir() -> Path:
    return Path(os.environ.get("ABCITE_CLIP_CACHE", Path.home() / ".cache" / "clip"))


def clip_checkpoint_path() -> Path:
    return clip_cache_dir() / "ViT-B-32.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < CLIP_MIN_BYTES:
        return False
    return _sha256(path) == CLIP_SHA256


def clip_weights_cached() -> bool:
    dest = clip_checkpoint_path()
    if not dest.is_file() or dest.stat().st_size < CLIP_MIN_BYTES:
        return False
    return True


def _torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OpenCLIPEncoder(VisionEncoder):
    name = CLIP_NAME
    dim = 512

    def __init__(self, model=None, preprocess=None, tokenizer=None) -> None:
        if model is None:
            model, preprocess, tokenizer = _load_open_clip()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.last_preprocess: dict | None = None
        self.device = next(model.parameters()).device

    def _cache_key(self, path: Path) -> str:
        return f"{self.name}:{Path(path).resolve()}"

    def encode_image(self, path: Path) -> list[float]:
        key = self._cache_key(path)
        cached = _VEC_CACHE.get(key)
        if cached is not None:
            return list(cached)
        arr = self.encode_images([path])
        return arr[0].tolist()

    def encode_images(self, paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """Pixel-only batched encode. Captions never enter this path."""
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        pending_i: list[int] = []
        pending_p: list[Path] = []
        encoded = 0
        for i, path in enumerate(paths):
            key = self._cache_key(path)
            cached = _VEC_CACHE.get(key)
            if cached is not None:
                out[i] = np.asarray(cached, dtype=np.float32)
                continue
            pending_i.append(i)
            pending_p.append(path)
            if len(pending_p) >= batch_size:
                self._encode_image_batch(pending_i, pending_p, out)
                encoded += len(pending_p)
                if encoded % 256 < batch_size:
                    print(f"  {self.name} encoded {encoded}/{len(paths)}", flush=True)
                pending_i, pending_p = [], []
        if pending_p:
            self._encode_image_batch(pending_i, pending_p, out)
        return out

    def _encode_image_batch(
        self, indices: list[int], paths: list[Path], out: np.ndarray
    ) -> None:
        import torch
        from concurrent.futures import ThreadPoolExecutor

        def _prep(path: Path):
            rgb = load_rgb(path)
            tensor = self.preprocess(rgb)
            if not hasattr(tensor, "unsqueeze"):
                tensor = torch.as_tensor(np.asarray(tensor))
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            return path, rgb, tensor

        workers = min(8, max(1, len(paths)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            prepped = list(pool.map(_prep, paths))
        tensors = []
        for path, rgb, tensor in prepped:
            self.last_preprocess = {"mode": rgb.mode, "size": rgb.size, "path": str(path)}
            tensors.append(tensor)
        batch = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        vecs = feat.detach().cpu().numpy().astype(np.float32)
        self.dim = int(vecs.shape[1])
        for row, path, idx in zip(vecs, paths, indices, strict=True):
            _VEC_CACHE[self._cache_key(path)] = row.tolist()
            out[idx] = row

    def encode_text(self, text: str) -> list[float]:
        return self.encode_texts([text])[0].tolist()

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(batch)
            if hasattr(tokens, "to"):
                tokens = tokens.to(self.device)
            with torch.no_grad():
                feat = self.model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            chunks.append(feat.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    def caption(self, path: Path) -> str:
        return LocalImageEncoder().caption(path)


def download_openclip_weights() -> Path:
    """Fetch ViT-B-32 openai weights from the OpenAI CDN into ~/.cache/clip."""
    cache = clip_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dest = clip_checkpoint_path()
    if _checkpoint_ok(dest):
        return dest
    if dest.exists():
        dest.unlink()
    partial = dest.with_name("ViT-B-32.pt.partial")
    print(f"Downloading {CLIP_URL}", flush=True)
    _download_resumable(CLIP_URL, partial)
    digest = _sha256(partial)
    if digest != CLIP_SHA256:
        size = partial.stat().st_size
        raise RuntimeError(
            f"CLIP SHA256 mismatch after download ({size} bytes, sha={digest[:12]}…). "
            "Delete ~/.cache/clip/ViT-B-32.pt.partial and retry."
        )
    partial.replace(dest)
    return dest


def _download_resumable(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    while True:
        start = dest.stat().st_size if dest.exists() else 0
        headers = {"User-Agent": "abcite-clip"}
        if start:
            headers["Range"] = f"bytes={start}-"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as src:
            status = getattr(src, "status", 200)
            length = src.headers.get("Content-Length")
            if start and status == 200:
                dest.unlink(missing_ok=True)
                start = 0
            mode = "ab" if start and status == 206 else "wb"
            total = start + int(length) if length else None
            print(f"  resume_from={start} status={status} content_length={length}", flush=True)
            with dest.open(mode) as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    if total:
                        done = dest.stat().st_size
                        if done % (8 << 20) < (1 << 20):
                            print(f"  {done}/{total} bytes", flush=True)
        size = dest.stat().st_size
        if total and size < total:
            print(f"  incomplete {size}/{total}, retrying", flush=True)
            continue
        if size < CLIP_MIN_BYTES:
            raise RuntimeError(f"CLIP download too small: {size} bytes")
        return


def _load_open_clip():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import open_clip

    ckpt = download_openclip_weights()
    cfg_mean = (0.48145466, 0.4578275, 0.40821073)
    cfg_std = (0.26862954, 0.26130258, 0.27577711)
    device = _torch_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_ARCH,
        pretrained=str(ckpt),
        device=device,
        force_quick_gelu=True,
        image_mean=cfg_mean,
        image_std=cfg_std,
        weights_only=False,  # OpenAI ViT-B-32.pt is a TorchScript archive
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(CLIP_ARCH)
    _MODEL = (model, preprocess, tokenizer)
    return _MODEL


def load_open_clip() -> OpenCLIPEncoder:
    return OpenCLIPEncoder()


class BiomedCLIPEncoder(OpenCLIPEncoder):
    name = "biomedclip"
    dim = 512


def _allow_biomedclip_download() -> bool:
    return os.environ.get("ABCITE_BIOMEDCLIP_DOWNLOAD", "").lower() in {"1", "true", "yes"}


def _snapshot_repo(
    repo_id: str, *, token: str | None, cache_dir: Path, local_dir: Path | None
) -> Path | None:
    """Download one Hub repo with retries. Prefer huggingface.co when a token is set."""
    import shutil
    import subprocess

    endpoints: list[str] = []
    forced = os.environ.get("HF_ENDPOINT")
    if forced:
        endpoints.append(forced)
    if token:
        endpoints.append("https://huggingface.co")
    endpoints.extend(["https://huggingface.co", "https://hf-mirror.com"])
    seen: set[str] = set()
    ordered: list[str] = []
    for ep in endpoints:
        if ep and ep not in seen:
            seen.add(ep)
            ordered.append(ep)

    last_err: Exception | None = None
    try:
        from huggingface_hub import login, snapshot_download
    except ImportError:
        snapshot_download = None  # type: ignore[assignment]
        login = None  # type: ignore[assignment]
    if token and callable(login):
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as exc:
            log.warning("huggingface login skipped: %s", type(exc).__name__)

    kwargs: dict = {
        "repo_id": repo_id,
        "token": token,
        "cache_dir": str(cache_dir),
        "max_workers": 1,
    }
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)

    if snapshot_download is not None:
        for endpoint in ordered:
            os.environ["HF_ENDPOINT"] = endpoint
            for attempt in range(8):
                try:
                    path = snapshot_download(**kwargs)
                    p = Path(path)
                    if p.exists():
                        return p
                except Exception as exc:
                    last_err = exc
                    wait = min(120.0, 5.0 * (2**attempt))
                    log.warning(
                        "snapshot_download %s %s attempt %s: %s; sleep %.0fs",
                        repo_id,
                        endpoint,
                        attempt + 1,
                        exc,
                        wait,
                    )
                    time.sleep(wait)

    cli = shutil.which("huggingface-cli") or shutil.which("hf")
    if cli:
        dest = local_dir or (cache_dir / repo_id.replace("/", "--"))
        dest.mkdir(parents=True, exist_ok=True)
        for endpoint in ordered:
            env = os.environ.copy()
            env["HF_ENDPOINT"] = endpoint
            if token:
                env["HF_TOKEN"] = token
                env["HUGGING_FACE_HUB_TOKEN"] = token
            cmd = [cli, "download", repo_id, "--local-dir", str(dest)]
            try:
                subprocess.run(cmd, check=True, env=env, timeout=1200)
            except Exception as exc:
                last_err = exc
                log.warning("huggingface-cli %s %s: %s", repo_id, endpoint, exc)
                continue
            if dest.exists():
                return dest
    if last_err:
        log.warning("Hub download failed for %s: %s", repo_id, last_err)
    return None


def download_biomedclip_weights() -> Path | None:
    """Fetch BiomedCLIP (and PubMedBERT tokenizer) into the CLIP/HF cache."""
    existing = _biomedclip_cached_files()
    if existing:
        return existing[0]
    if not _allow_biomedclip_download():
        return None
    dest_dir = clip_cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    home = _hf_home()
    home.mkdir(parents=True, exist_ok=True)
    hub = home / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(home)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    token = _hf_token()
    local_dir = dest_dir / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    snapped = _snapshot_repo(BIOMEDCLIP_REPO, token=token, cache_dir=hub, local_dir=local_dir)
    for dep in BIOMEDCLIP_TEXT_REPOS:
        _snapshot_repo(dep, token=token, cache_dir=hub, local_dir=None)
    hits = _biomedclip_cached_files()
    if hits:
        return hits[0]
    if snapped is not None:
        for match in Path(snapped).rglob("*"):
            if match.is_file() and match.stat().st_size > 50_000_000:
                return match
    return None


def _load_biomedclip() -> BiomedCLIPEncoder:
    """Load BiomedCLIP via open_clip hf-hub; download only if ABCITE_BIOMEDCLIP_DOWNLOAD=1."""
    import open_clip

    device = _torch_device()
    cache = clip_cache_dir()
    downloaded = download_biomedclip_weights()
    cached = bool(_biomedclip_cached_files())
    allow = _allow_biomedclip_download() or downloaded is not None or cached
    if not allow:
        raise FileNotFoundError(f"no BiomedCLIP checkpoint in {cache}")

    names: list[str] = list(BIOMEDCLIP_HF_NAMES)
    local_dir = cache / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    if (local_dir / "open_clip_config.json").is_file():
        names.insert(0, f"hf-hub:{local_dir}")
    last: Exception | None = None
    for name in names:
        try:
            create = getattr(open_clip, "create_model_from_pretrained", None)
            if callable(create):
                model, preprocess = create(name, device=device)
                tok = open_clip.get_tokenizer(name)
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(name, device=device)
                tok = open_clip.get_tokenizer(name)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            enc = BiomedCLIPEncoder(model, preprocess, tok)
            enc.dim = int(getattr(model, "embed_dim", enc.dim) or enc.dim)
            return enc
        except Exception as exc:
            last = exc
            log.warning("BiomedCLIP hub load %s failed (%s)", name, exc)
    if last:
        raise last
    raise FileNotFoundError(f"no BiomedCLIP checkpoint in {cache}")


def try_load_biomedclip() -> tuple[VisionEncoder | None, str | None]:
    """Best-effort BiomedCLIP. Never raises; skip reason is for the eval receipt."""
    if not clip_stack_available():
        return None, "BiomedCLIP skipped: open_clip/torch not installed"
    try:
        enc = _load_biomedclip()
        return enc, None
    except Exception as exc:
        return None, f"BiomedCLIP skipped: {exc}"


def build_vision_encoder() -> VisionEncoder:
    """CLIP when `[clip]` + weights are available; local encoder otherwise.

    ABCITE_CLIP_MODEL=biomedclip loads only if weights are cached or
    ABCITE_BIOMEDCLIP_DOWNLOAD=1. OpenCLIP ViT-B-32 remains the default path.
    """
    skip = biomedclip_skip_reason()
    if skip:
        log.warning(skip)
    elif biomedclip_requested():
        enc, err = try_load_biomedclip()
        if enc is not None:
            return enc
        log.warning("%s; trying OpenCLIP.", err)
    if clip_forced_off():
        return LocalImageEncoder()
    if clip_stack_available() and (clip_forced_on() or clip_weights_cached()):
        try:
            return load_open_clip()
        except Exception as exc:
            if clip_forced_on():
                raise
            log.warning(
                "CLIP load failed (%s); using local encoder. "
                "Run: python -m abcite_core.download_clip",
                exc,
            )
            return LocalImageEncoder()
    if clip_forced_on():
        raise RuntimeError(
            "ABCITE_CLIP=1 but open_clip is not installed. "
            "Run: pip install -e '.[clip]' && python -m abcite_core.download_clip"
        )
    if clip_stack_available() and not clip_weights_cached():
        log.warning(
            "open_clip is installed but weights are missing; using local encoder. "
            "Run: python -m abcite_core.download_clip"
        )
    return LocalImageEncoder()
