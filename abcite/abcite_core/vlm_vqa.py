"""Cite-or-abstain visual QA via a real VLM generate path.

Retrieval (ColQwen2 MaxSim) is a different system. This module never scores
`custom_text_proj` / MaxSim as VQA. Product default stays extractive until
`ABCITE_VLM=1` and generate weights load.

Preferred weights: `Qwen/Qwen2-VL-2B-Instruct` (architecture
Qwen2VLForConditionalGeneration). Cached locally on the Mac Hub (~4.4 GB).

Not a generate path: `vidore/colqwen2-base` (architecture ColQwen2,
custom_text_proj) + LoRA `vidore/colqwen2-v1.0` (FEATURE_EXTRACTION). Those
stay the ViDoRe / PMC figure **retriever**.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from abcite_core.holdout import query_contains_pdb, strip_pdb_ids
from abcite_core.visual_qa import VQAResult

log = logging.getLogger("abcite.vlm_vqa")

DEFAULT_VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
HEURISTIC_PATH = "extractive/heuristic pixel cues; not a VLM"
VLM_PATH = "qwen2-vl-2b-instruct generate (cite-or-abstain); not MaxSim"

_STATUS = re.compile(r"STATUS:\s*(answered|abstain)\b", re.I)
_CITE = re.compile(r"CITE:\s*(\S+)", re.I)
_ANSWER = re.compile(r"ANSWER:\s*(.*?)(?:\n\s*EVIDENCE:|\Z)", re.I | re.S)
_EVIDENCE = re.compile(r"EVIDENCE:\s*(.*)\Z", re.I | re.S)

_VLM = None
_VLM_SKIP: str | None = None
_VLM_TRIED = False


def vlm_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def vlm_enabled() -> bool:
    return vlm_flag("ABCITE_VLM", "0")


def vlm_model_name() -> str:
    return (os.environ.get("ABCITE_VLM_MODEL") or DEFAULT_VLM_MODEL).strip()


def reset_vlm_loader() -> None:
    """Tests: drop the singleton after monkeypatching env."""
    global _VLM, _VLM_SKIP, _VLM_TRIED
    _VLM = None
    _VLM_SKIP = None
    _VLM_TRIED = False


def _hf_home() -> Path:
    raw = os.environ.get("HF_HOME")
    if raw:
        return Path(raw)
    if Path("/workspace/hf-cache").is_dir():
        return Path("/workspace/hf-cache")
    if Path("/workspace/hf").is_dir():
        return Path("/workspace/hf")
    return Path.home() / ".cache" / "huggingface"


def _hub_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _hub_snapshot(model_id: str) -> Path | None:
    root = _hf_home() / "hub" / _hub_dir_name(model_id) / "snapshots"
    if not root.is_dir():
        return None
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    return snaps[0] if snaps else None


def generate_vlm_ok(cfg: dict) -> tuple[bool, str]:
    """True only for a causal / conditional-generation VLM, not ColQwen2 retrieval."""
    arch = [str(a) for a in (cfg.get("architectures") or [])]
    joined = " ".join(arch)
    if "ColQwen" in joined or "ColPali" in joined:
        return False, (
            "vidore/colqwen2-base is a retrieval encoder (architecture ColQwen2, "
            "custom_text_proj, MaxSim). It is not a VQA generate path. "
            f"Use {DEFAULT_VLM_MODEL}."
        )
    if any("ForConditionalGeneration" in a or "ForCausalLM" in a for a in arch):
        return True, ""
    return False, f"not a generate VLM: architectures={arch or ['unknown']}"


def is_colqwen_retrieval_id(model_id: str) -> bool:
    low = model_id.lower()
    return "colqwen" in low or "colpali" in low


def sanitize_visual_query(question: str) -> str:
    """Strip PDB ids. Visual QA never uses PDB lookup in the query path."""
    return strip_pdb_ids(question or "")


def query_has_pdb(question: str) -> bool:
    return query_contains_pdb(question or "")


def build_vlm_prompt(question: str, figure_id: str) -> tuple[str, str]:
    """Return (prompt, sanitized_question). Sanitized query never contains PDB ids."""
    q = sanitize_visual_query(question)
    if query_contains_pdb(q):
        q = re.sub(r"\b[0-9][a-zA-Z0-9]{3}\b", " ", q)
        q = " ".join(q.split())
    prompt = (
        "You are AbCite visual QA. Look only at the figure pixels you are shown.\n"
        f"The figure id is {figure_id}. Cite that id if and only if the pixels "
        "support the question.\n"
        "Do not use structure-database identifiers, paper captions, or filenames "
        "as evidence. If the pixels do not show the requested visual evidence, abstain.\n\n"
        f"Question: {q}\n\n"
        "Reply with exactly these lines:\n"
        "STATUS: answered OR abstain\n"
        f"CITE: {figure_id} OR none\n"
        "ANSWER: one sentence about what is visible (empty if abstain)\n"
        "EVIDENCE: panel layout, stain, gel bands, colors, arrows, microscopy type; "
        "or why those cues are missing"
    )
    return prompt, q


def parse_vlm_generation(text: str, shown_figure_id: str, *, query_used: str = "") -> VQAResult:
    """Parse STATUS/CITE/ANSWER/EVIDENCE. Parse failure → abstain."""
    raw = (text or "").strip()
    status_m = _STATUS.search(raw)
    cite_m = _CITE.search(raw)
    ans_m = _ANSWER.search(raw)
    ev_m = _EVIDENCE.search(raw)
    status = (status_m.group(1).lower() if status_m else "")
    cite_tok = (cite_m.group(1).strip() if cite_m else "")
    answer = (ans_m.group(1).strip() if ans_m else "")
    evidence = (ev_m.group(1).strip() if ev_m else "")
    if cite_tok.lower() in {"none", "null", "n/a", "-", "no"}:
        cite_tok = ""
    if not status:
        low = raw.lower()
        if not raw or "abstain" in low[:80] or "cannot answer" in low or "not visible" in low:
            status = "abstain"
        else:
            status = "abstain"
    if status == "answered" and (not cite_tok or cite_tok.lower() == "none"):
        status = "abstain"
    if status == "answered":
        cite = shown_figure_id
        if not answer:
            answer = raw.splitlines()[0][:280] if raw else "Visible in the cited figure."
        text_out = f"{answer} Cite {cite}."
        if evidence:
            text_out = f"{text_out} Evidence: {evidence[:240]}"
        return VQAResult(
            status="answered",
            answer=text_out,
            figure_id=cite,
            cues=[],
            pixel_desc=evidence,
            evidence=evidence,
            path=VLM_PATH,
            query_used=query_used,
        )
    why = evidence or answer or "pixels do not support the question"
    return VQAResult(
        status="abstain",
        answer=(
            f"Retrieved figure {shown_figure_id} does not contain the requested "
            f"visual evidence ({why}). Abstaining."
        ),
        figure_id=None,
        cues=[],
        pixel_desc=evidence,
        evidence=evidence,
        path=VLM_PATH,
        query_used=query_used,
    )


def vlm_weights_cached(model_id: str | None = None) -> Path | None:
    mid = model_id or vlm_model_name()
    snap = _hub_snapshot(mid)
    if snap is None:
        return None
    cfg = snap / "config.json"
    if not cfg.is_file():
        return None
    shards = list(snap.glob("*.safetensors")) + list(snap.glob("*.bin"))
    if not shards:
        return None
    return snap


class VlmVqaAnswerer:
    """Qwen2-VL generate. One figure at a time. Never MaxSim."""

    path = VLM_PATH

    def __init__(self, model, processor, device, model_id: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.model_id = model_id
        self.max_new_tokens = int(os.environ.get("ABCITE_VLM_MAX_NEW_TOKENS", "160") or "160")

    def answer(self, question: str, rec, path: Path | None, **_kw) -> VQAResult:
        from abcite_core.visual_qa import heuristic_vqa

        if rec is None:
            return VQAResult(
                status="abstain",
                answer="No figure retrieved. Abstaining.",
                figure_id=None,
                path=self.path,
                query_used=sanitize_visual_query(question),
            )
        img_path = path
        if img_path is None or not Path(img_path).exists():
            return heuristic_vqa(question, rec, path)
        prompt, q_used = build_vlm_prompt(question, rec.figure_id)
        if query_contains_pdb(q_used):
            log.error("refusing VLM call: PDB token survived sanitization")
            return VQAResult(
                status="abstain",
                answer="Query still contained a structure id after sanitization. Abstaining.",
                figure_id=None,
                path=self.path,
                query_used=q_used,
            )
        try:
            text = self._generate(Path(img_path), prompt)
        except Exception as exc:
            log.warning("VLM generate failed (%s); abstain", exc)
            return VQAResult(
                status="abstain",
                answer=f"VLM generate failed ({type(exc).__name__}). Abstaining.",
                figure_id=None,
                path=self.path,
                query_used=q_used,
            )
        return parse_vlm_generation(text, rec.figure_id, query_used=q_used)

    def _generate(self, path: Path, prompt: str) -> str:
        import torch

        from abcite_core.raster import load_rgb

        image = load_rgb(path, size=None)
        w, h = image.size
        max_side = int(os.environ.get("ABCITE_VLM_MAX_SIDE", "1024") or "1024")
        m = max(w, h)
        if m > max_side:
            scale = max_side / float(m)
            image = image.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                resample=__import__("PIL").Image.Resampling.BICUBIC,
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        in_len = int(inputs["input_ids"].shape[1])
        gen = out[0, in_len:]
        tok = getattr(self.processor, "tokenizer", None) or self.processor
        return tok.decode(gen, skip_special_tokens=True).strip()


def try_load_vlm() -> tuple[VlmVqaAnswerer | None, str | None]:
    """Load Qwen2-VL-Instruct. Never raises. Rejects ColQwen2 retrieval checkpoints."""
    if not vlm_enabled():
        return None, "VLM skipped: ABCITE_VLM=0 (heuristic cite-or-abstain is the product default)"
    model_id = vlm_model_name()
    if is_colqwen_retrieval_id(model_id):
        return None, (
            f"VLM skipped: {model_id} is ColQwen2/ColPali retrieval (custom_text_proj / MaxSim), "
            f"not a generate VLM. Set ABCITE_VLM_MODEL={DEFAULT_VLM_MODEL}."
        )
    snap = vlm_weights_cached(model_id)
    local_only = (
        vlm_flag("HF_HUB_OFFLINE", "0")
        or vlm_flag("ABCITE_VLM_LOCAL", "0")
        or snap is not None
    )
    if snap is None and local_only:
        return None, (
            f"VLM skipped: {model_id} not in HF cache ({_hf_home()}). "
            "On the 5090 copy Qwen/Qwen2-VL-2B-Instruct into $HF_HOME/hub "
            "(Mac already has models--Qwen--Qwen2-VL-2B-Instruct, ~4.4 GB). "
            "Do not substitute vidore/colqwen2-base."
        )
    load_from = str(snap) if snap is not None else model_id
    cfg_path = Path(load_from) / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"VLM skipped: bad config.json ({exc})"
        ok, reason = generate_vlm_ok(cfg)
        if not ok:
            return None, f"VLM skipped: {reason}"
    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except Exception as exc:
        return None, (
            f"VLM skipped: {type(exc).__name__}: {exc}. "
            "Install: pip install -e '.[vlm]'"
        )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    default_pixels = 512 * 28 * 28
    max_pixels = int(os.environ.get("ABCITE_VLM_MAX_PIXELS", str(default_pixels)) or default_pixels)
    try:
        processor = AutoProcessor.from_pretrained(
            load_from,
            local_files_only=local_only,
            min_pixels=256 * 28 * 28,
            max_pixels=max_pixels,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            load_from,
            torch_dtype=dtype,
            device_map=str(device) if device.type == "cuda" else None,
            local_files_only=local_only,
        ).eval()
        if device.type != "cuda":
            model = model.to(device)
    except Exception as exc:
        log.warning("VLM load failed: %s", exc)
        return None, f"VLM skipped: load failed ({type(exc).__name__}: {exc})"
    log.info("loaded VLM %s on %s (generate, not MaxSim)", model_id, device)
    return VlmVqaAnswerer(model, processor, device, model_id), None


def get_vlm() -> tuple[VlmVqaAnswerer | None, str | None]:
    global _VLM, _VLM_SKIP, _VLM_TRIED
    if not vlm_enabled():
        return None, "VLM skipped: ABCITE_VLM=0 (heuristic cite-or-abstain is the product default)"
    if _VLM_TRIED:
        return _VLM, _VLM_SKIP
    _VLM_TRIED = True
    _VLM, _VLM_SKIP = try_load_vlm()
    return _VLM, _VLM_SKIP
