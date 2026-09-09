"""Parent-child chunking for papers, captions, and PDB residue rows."""

from __future__ import annotations

import json
import re
from pathlib import Path

from abcite_core.models import Chunk, Modality

WORD_RE = re.compile(r"\w+")


def split_parent_child(parent_id: str, title: str, text: str, *, child_words: int = 80) -> list[tuple[str, str, int, int]]:
    words = list(WORD_RE.finditer(text))
    if not words:
        return [(f"{parent_id}::c0", text, 0, len(text))]
    out: list[tuple[str, str, int, int]] = []
    i = 0
    n = 0
    while i < len(words):
        chunk_words = words[i : i + child_words]
        start = chunk_words[0].start()
        end = chunk_words[-1].end()
        out.append((f"{parent_id}::c{n}", text[start:end], start, end))
        i += max(1, child_words // 2)
        n += 1
        if n > 40:
            break
    return out


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    paths = sorted(corpus_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no jsonl corpus under {corpus_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                raw["modality"] = Modality(raw["modality"])
                chunks.append(Chunk.model_validate(raw))
    return chunks
