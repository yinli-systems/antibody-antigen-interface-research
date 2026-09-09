from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from abcite_core.models import Chunk, Citation
from abcite_core.vision import VisionEncoder

__all__ = ["LLMProvider", "VisionEncoder", "citations_from_chunks", "format_evidence"]


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


def format_evidence(chunks: list[Chunk]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c.chunk_id}|{c.modality.value}] {c.title}: {c.text[:600]}")
    return "\n".join(lines)


def citations_from_chunks(chunks: list[Chunk], claim: str) -> list[Citation]:
    out: list[Citation] = []
    for c in chunks:
        quote = c.text[:280]
        thumb = None
        if c.image_path:
            thumb = f"/media/{Path(c.image_path).name}"
        out.append(
            Citation(
                chunk_id=c.chunk_id,
                claim=claim,
                quote=quote,
                char_start=c.char_start,
                char_end=min(c.char_end or len(quote), c.char_start + len(quote)),
                modality=c.modality,
                source_url=c.source_url,
                source_label=c.source_label,
                thumbnail_url=thumb,
                pdb_id=c.pdb_id,
                residues=c.residues,
            )
        )
    return out
