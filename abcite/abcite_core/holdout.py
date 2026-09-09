"""Hold out PDB ids, accessions, and caption text from retrieval queries."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Sequence

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
        "phylogen",
        "forest plot",
        "confocal",
        "cryo",
        "immunoblot",
        "agarose",
        "histolog",
        "lane",
        "bands",
        "chevron",
        "grayscale",
        "wavenumber",
        "overlay",
        "whisker",
        "diamond",
        "tiles",
        "fibril",
        "gadolinium",
        "orbit",
        "scale bar",
        "counterstain",
        "cladogram",
        "radiology",
        "karyotype",
        "volcano",
        "microfluidic",
        "anaglyph",
        "dichroism",
        "western",
        "embryo",
        "confocal",
        "dark-field",
        "dark field",
        "scattered-light",
        "flowchart",
        "block diagram",
        "line chart",
        "bar chart",
        "time-course",
        "spinning-disk",
        "mitotic",
        "snapshot",
        "pointer",
        "overlaid",
        "speckle",
        "similarity matrix",
        "colored points",
        "theoretical curve",
        "scarlet",
        "crimson",
        "color-key",
        "fibril",
        "nephron",
        "chromosome",
        "phase-contrast",
        "kilodalton",
        "asterisks",
        "maltese",
    )
    return any(c in q for c in cues)


def paper_key(pmcid: str | None, figure_id: str) -> str:
    pmc = (pmcid or "").strip()
    return pmc if pmc else f"fig:{figure_id}"


@dataclass(frozen=True)
class PaperSplit:
    index_recs: tuple[Any, ...]
    query_recs: tuple[Any, ...]
    index_papers: frozenset[str]
    query_papers: frozenset[str]
    n_papers: int
    query_frac: float
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_papers": self.n_papers,
            "n_index_papers": len(self.index_papers),
            "n_query_papers": len(self.query_papers),
            "n_index_figures": len(self.index_recs),
            "n_query_figures": len(self.query_recs),
            "query_frac": self.query_frac,
            "seed": self.seed,
            "split_by": "pmcid",
            "note": (
                "Queries are captions from the query-paper split. Those papers are absent "
                "from the 80% index-paper set. Eval gallery mode decides whether query-paper "
                "images are included as retrieval candidates."
            ),
        }


def split_by_paper(
    recs: Sequence[Any],
    query_frac: float = 0.2,
    seed: int = 7,
) -> PaperSplit:
    """80/20 (default) split by PMCID so query papers are not the index-paper set.

    Figures without a PMCID are treated as their own paper (`fig:<id>`).
    """
    if not recs:
        return PaperSplit((), (), frozenset(), frozenset(), 0, query_frac, seed)
    grouped: dict[str, list[Any]] = {}
    for rec in recs:
        key = paper_key(getattr(rec, "pmcid", None), getattr(rec, "figure_id", ""))
        grouped.setdefault(key, []).append(rec)
    papers = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(papers)
    n_q = max(1, min(len(papers) - 1 if len(papers) > 1 else 1, round(len(papers) * query_frac)))
    if len(papers) == 1:
        n_q = 1
    query_papers = frozenset(papers[:n_q])
    index_papers = frozenset(papers[n_q:])
    query_recs = tuple(rec for p in papers[:n_q] for rec in grouped[p])
    index_recs = tuple(rec for p in papers[n_q:] for rec in grouped[p])
    return PaperSplit(
        index_recs=index_recs,
        query_recs=query_recs,
        index_papers=index_papers,
        query_papers=query_papers,
        n_papers=len(papers),
        query_frac=query_frac,
        seed=seed,
    )
