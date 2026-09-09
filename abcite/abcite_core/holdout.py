"""Hold out PDB ids, accessions, and caption text from retrieval queries."""

from __future__ import annotations

import re

# Real PDB ids are 4 chars: digit + alnum with at least one letter (not years like 2023).
_PDB_TOKEN = re.compile(r"\b[0-9][a-zA-Z0-9]{3}\b")
_PDB_PREFIX = re.compile(r"\b(?:pdb[:\s-]*)([0-9][a-zA-Z0-9]{3})\b", re.I)
_NOT_PDB = re.compile(
    r"^\d+(?:st|nd|rd|th|hrs?|mins?|secs?|days?|wks?|gy|bp|kb|mb|nm|um|mm|cm|"
    r"kg|mg|ug|ml|ul)$",
    re.I,
)
# UniProtKB accession: 6 or 10 chars (P04626, A0A024R1R8).
_UNIPROT = re.compile(
    r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]"
    r"(?:[A-Z][A-Z0-9]{2}[0-9])?)\b"
)
_REFSEQ = re.compile(r"\b(?:NP|XP|NM|NR|NC|WP|XM|XR)_\d+(?:\.\d+)?\b", re.I)
_ID_WORDS = re.compile(r"\b(?:pdb(?:\s*id)?|uniprot(?:\s*acc(?:ession)?)?|accession)\b", re.I)


def is_pdb_id(token: str) -> bool:
    t = token.strip()
    if _NOT_PDB.match(t):
        return False
    if len(t) != 4 or not t[0].isdigit():
        return False
    return any(c.isalpha() for c in t)


def strip_pdb_ids(text: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        return " " if is_pdb_id(m.group(0)) else m.group(0)

    out = _PDB_PREFIX.sub(" ", text)
    out = _PDB_TOKEN.sub(_sub, out)
    return " ".join(out.split())


def strip_accession_tokens(text: str) -> str:
    """Strip PDB ids, UniProt/RefSeq accessions, and the words PDB / UniProt.

    Caption-as-query must remain SciMMIR-style pair retrieval, not ID lookup.
    """
    out = strip_pdb_ids(text)
    out = _UNIPROT.sub(" ", out)
    out = _REFSEQ.sub(" ", out)
    out = _ID_WORDS.sub(" ", out)
    return " ".join(out.split())


def query_contains_pdb(text: str) -> bool:
    if _PDB_PREFIX.search(text):
        tok = _PDB_PREFIX.search(text)
        if tok and is_pdb_id(tok.group(1)):
            return True
    return any(is_pdb_id(m.group(0)) for m in _PDB_TOKEN.finditer(text))


def strip_caption_leak(text: str, caption: str | None) -> str:
    """Remove caption n-grams so BM25/CLIP text is not a caption copy."""
    out = strip_accession_tokens(text)
    if not caption:
        return out
    cap = strip_accession_tokens(caption).lower()
    cap_tokens = cap.split()
    if len(cap_tokens) < 4:
        return out
    lowered = out.lower()
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
        "sds-page",
        "immunofluoresc",
        "ct scan",
    )
    return any(c in q for c in cues)
