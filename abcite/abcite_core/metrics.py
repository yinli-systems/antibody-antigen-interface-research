"""Hit@k, ranking, lexical + claim-level faithfulness, failure taxonomy."""

from __future__ import annotations

import math
import re
from typing import Any

WORD = re.compile(r"[a-z0-9]{3,}")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


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


def mrr(retrieved_ids: list[str], required: list[str]) -> float:
    if not required:
        return 1.0
    needed = set(required)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in needed:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], required: list[str], k: int = 10) -> float:
    """Binary nDCG@k over required chunk ids."""
    if not required:
        return 1.0
    rel = [1.0 if cid in set(required) else 0.0 for cid in retrieved_ids[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal_hits = min(len(required), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def citation_id_match(cited_ids: list[str], required_image_id: str | None) -> bool:
    if not required_image_id:
        return True
    return required_image_id in cited_ids


def faithfulness(answer: str, cited_texts: list[str], status: str) -> float:
    """Lexical smoke test only: abstain with no cites is 1.0; else overlap of answer vs cites."""
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


def split_claims(answer: str) -> list[str]:
    text = (answer or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in SENTENCE.split(text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"[\n;]+", text) if p.strip()]
    out: list[str] = []
    for p in parts:
        if len(WORD.findall(p.lower())) >= 3:
            out.append(p)
    return out or ([text] if text else [])


def _span_covers(claim: str, span: str, min_overlap: float = 0.35) -> bool:
    cw = set(WORD.findall(claim.lower()))
    sw = set(WORD.findall(span.lower()))
    if not cw or not sw:
        return False
    return (len(cw & sw) / len(cw)) >= min_overlap


def align_claims_to_spans(
    answer: str, cited_texts: list[str], status: str
) -> list[dict[str, Any]]:
    """Each claim maps to a cited span or is marked unsupported. Abstain+no claims → empty."""
    if status == "abstain":
        return []
    claims = split_claims(answer)
    aligned: list[dict[str, Any]] = []
    for claim in claims:
        best_span = None
        best = 0.0
        for span in cited_texts:
            cw = set(WORD.findall(claim.lower()))
            sw = set(WORD.findall(span.lower()))
            ov = (len(cw & sw) / max(1, len(cw))) if cw else 0.0
            if ov > best:
                best = ov
                best_span = span
        supported = best_span is not None and _span_covers(claim, best_span)
        aligned.append(
            {
                "claim": claim,
                "supported": supported,
                "span": best_span if supported else None,
                "overlap": round(best, 4),
            }
        )
    return aligned


def claim_faithfulness(answer: str, cited_texts: list[str], status: str) -> float:
    """Supported claims / total claims. Abstain with no claims = 1.0."""
    if status == "abstain":
        return 1.0 if not cited_texts else 0.0
    rows = align_claims_to_spans(answer, cited_texts, status)
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["supported"]) / len(rows)


def claim_nli_heuristic(answer: str, cited_texts: list[str], status: str) -> float:
    """Cheap lexical NLI: claim tokens mostly entailed by union of cites (not a neural NLI)."""
    if status == "abstain":
        return 1.0 if not cited_texts else 0.0
    context = set(WORD.findall(" ".join(cited_texts).lower()))
    claims = split_claims(answer)
    if not claims:
        return 0.0
    scores = []
    for claim in claims:
        cw = set(WORD.findall(claim.lower()))
        if not cw:
            scores.append(0.0)
            continue
        scores.append(len(cw & context) / len(cw))
    return float(sum(scores) / len(scores))


def cohens_kappa(a: list[int], b: list[int]) -> float | None:
    """Binary Cohen's κ. Returns None if lists are empty or length-mismatched."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if abs(1 - pe) < 1e-9:
        return 1.0 if agree >= 1.0 - 1e-9 else 0.0
    return float((agree - pe) / (1 - pe))


def label_failure(
    *,
    expected_status: str,
    actual_status: str,
    cited_ids: list[str],
    retrieved_ids: list[str],
    required_ids: list[str],
) -> str | None:
    """Auto-label one gold item. None = no taxonomy hit."""
    if expected_status == "abstain" and actual_status != "abstain":
        return "missed_abstain"
    if expected_status in {"answered", "hitl_pending"} and actual_status == "abstain":
        return "over_abstain"
    if actual_status != "abstain" and cited_ids:
        retrieved = set(retrieved_ids)
        required = set(required_ids)
        for cid in cited_ids:
            if cid not in retrieved and cid not in required:
                return "hallucinated_cite"
        if required and not (set(cited_ids) & required) and set(cited_ids) - retrieved:
            return "hallucinated_cite"
    return None
