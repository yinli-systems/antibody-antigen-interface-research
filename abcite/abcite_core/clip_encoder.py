"""Real CLIP encoder (OpenCLIP ViT-B-32) with SVG→RGB rasterization.

Install: `pip install -e ".[clip]"` then `python -m abcite_core.download_clip`.
HybridIndex auto-uses CLIP when open_clip is importable **and** weights are cached,
unless ABCITE_CLIP=0. ABCITE_CLIP=1 downloads if needed and fails hard on error.
"""

from __future__ import annotations

import hashlib
import logging
import os
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


def biomedclip_weights_present() -> bool:
    cache = clip_cache_dir()
    candidates = [
        cache / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.bin",
        cache / "biomedclip.pt",
        cache / "open_clip_pytorch_model_biomedclip.bin",
    ]
    if any(p.is_file() and p.stat().st_size > 50_000_000 for p in candidates):
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

    def encode_image(self, path: Path) -> list[float]:
        import torch

        key = str(Path(path).resolve())
        cached = _VEC_CACHE.get(key)
        if cached is not None:
            return list(cached)
        rgb = load_rgb(path)
        self.last_preprocess = {"mode": rgb.mode, "size": rgb.size, "path": str(path)}
        tensor = self.preprocess(rgb)
        if not hasattr(tensor, "unsqueeze"):
            tensor = torch.as_tensor(np.asarray(tensor))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        vec = feat[0].detach().cpu().numpy().astype(np.float32).tolist()
        _VEC_CACHE[key] = vec
        return vec

    def encode_text(self, text: str) -> list[float]:
        import torch

        tokens = self.tokenizer([text])
        if hasattr(tokens, "to"):
            tokens = tokens.to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].detach().cpu().numpy().astype(np.float32).tolist()

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


BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT-256-vit-base-patch16-224"


def _load_biomedclip() -> BiomedCLIPEncoder:
    """Load cached BiomedCLIP; download only if ABCITE_BIOMEDCLIP_DOWNLOAD=1."""
    import open_clip

    device = _torch_device()
    cache = clip_cache_dir()
    allow_download = os.environ.get("ABCITE_BIOMEDCLIP_DOWNLOAD", "").lower() in {
        "1",
        "true",
        "yes",
    }
    for name in (
        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.bin",
        "biomedclip.pt",
        "open_clip_pytorch_model_biomedclip.bin",
    ):
        ckpt = cache / name
        if ckpt.is_file() and ckpt.stat().st_size > 50_000_000:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained=str(ckpt), device=device
            )
            model.eval()
            tok = open_clip.get_tokenizer("ViT-B-16")
            return BiomedCLIPEncoder(model, preprocess, tok)
    if allow_download:
        model, _, preprocess = open_clip.create_model_and_transforms(
            BIOMEDCLIP_HF, device=device
        )
        model.eval()
        tok = open_clip.get_tokenizer(BIOMEDCLIP_HF)
        return BiomedCLIPEncoder(model, preprocess, tok)
    raise FileNotFoundError(f"no BiomedCLIP checkpoint in {cache}")


def build_vision_encoder() -> VisionEncoder:
    """CLIP when `[clip]` + weights are available; local encoder otherwise.

    ABCITE_CLIP_MODEL=biomedclip is an interface stub: we do not download BiomedCLIP.
    OpenCLIP ViT-B-32 remains the real encoder path when weights are cached.
    """
    skip = biomedclip_skip_reason()
    if skip:
        log.warning(skip)
    elif biomedclip_requested() and biomedclip_weights_present():
        try:
            return _load_biomedclip()
        except Exception as exc:
            log.warning("BiomedCLIP weights present but load failed (%s); trying OpenCLIP.", exc)
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
