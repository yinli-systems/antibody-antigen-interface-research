"""Hit@5, citation match, and extractive faithfulness over gold items."""

from __future__ import annotations

import re
from typing import Any

WORD = re.compile(r"[a-z0-9]{3,}")


def hit_at_k(retrieved_ids: list[str], required: list[str], k: int = 5) -> bool:
    if not required:
        return True
    top = set(retrieved_ids[:k])
    return any(r in top for r in required)


def recall_at_k(retrieved_ids: list[str], required: list[str], k: int = 5) -> float:
    if not required:
        return 1.0
    top = set(retrieved_ids[:k])
    return sum(1 for r in required if r in top) / len(required)


def citation_id_match(cited_ids: list[str], required_image_id: str | None) -> bool:
    if not required_image_id:
        return True
    return required_image_id in cited_ids


def faithfulness(answer: str, cited_texts: list[str], status: str) -> float:
    """Simple lexical faithfulness: abstain with no cites is 1.0; else overlap of answer vs cites."""
    if status == "abstain":
        return 1.0 if not cited_texts else 0.0
    if not cited_texts or not answer.strip():
        return 0.0
    a = set(WORD.findall(answer.lower()))
    c = set(WORD.findall(" ".join(cited_texts).lower()))
    if not a or not c:
        return 0.0
    overlap = len(a & c) / max(1, len(a))
    return float(min(1.0, overlap))
