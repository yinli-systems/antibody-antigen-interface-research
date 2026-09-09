"""Hold out PDB ids and caption text from the image-only retrieval path."""

from __future__ import annotations

import re

# Real PDB ids are 4 chars: digit + alnum with at least one letter (not years like 2023).
_PDB_TOKEN = re.compile(r"\b[0-9][a-zA-Z0-9]{3}\b")
_PDB_PREFIX = re.compile(r"\b(?:pdb[:\s-]*)([0-9][a-zA-Z0-9]{3})\b", re.I)


def is_pdb_id(token: str) -> bool:
    t = token.strip()
    if len(t) != 4 or not t[0].isdigit():
        return False
    return any(c.isalpha() for c in t)


def strip_pdb_ids(text: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        return " " if is_pdb_id(m.group(0)) else m.group(0)

    out = _PDB_PREFIX.sub(" ", text)
    out = _PDB_TOKEN.sub(_sub, out)
    return " ".join(out.split())


def query_contains_pdb(text: str) -> bool:
    if _PDB_PREFIX.search(text):
        tok = _PDB_PREFIX.search(text)
        if tok and is_pdb_id(tok.group(1)):
            return True
    return any(is_pdb_id(m.group(0)) for m in _PDB_TOKEN.finditer(text))


def strip_caption_leak(text: str, caption: str | None) -> str:
    """Remove caption n-grams so BM25/CLIP text is not a caption copy."""
    out = strip_pdb_ids(text)
    if not caption:
        return out
    cap = strip_pdb_ids(caption).lower()
    cap_tokens = cap.split()
    if len(cap_tokens) < 4:
        return out
    lowered = out.lower()
    # Drop 6-gram overlaps with the held-out caption.
    grams = {" ".join(cap_tokens[i : i + 6]) for i in range(max(1, len(cap_tokens) - 5))}
    for g in grams:
        if len(g) > 20 and g in lowered:
            lowered = lowered.replace(g, " ")
    return " ".join(lowered.split())


def visual_query_ok(question: str) -> bool:
    if query_contains_pdb(question):
        return False
    q = question.lower()
    cues = (
        "blot",
        "microscop",
        "ribbon",
        "schematic",
        "plot",
        "panel",
        "histogram",
        "heatmap",
        "gel",
        "fluorescence",
        "cartoon",
        "diagram",
        "scatter",
        "workflow",
        "table",
        "micrograph",
        "stain",
        "crystal",
        "photo",
        "spectra",
        "bar chart",
        "structure snapshot",
        "residue schematic",
    )
    return any(c in q for c in cues)
