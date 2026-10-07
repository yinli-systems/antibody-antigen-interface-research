"""Figure-id holdout for ColQwen2.5 LoRA. Split by gold figure, not query string."""

from __future__ import annotations

from typing import Any

import numpy as np

HOLD_SEED = 7
TRAIN_FRAC = 0.8


def primary_gold_id(item: dict) -> str | None:
    gold = item.get("gold_figure_ids") or []
    return str(gold[0]) if gold else None


def answerable_fail_items(items: list[dict]) -> list[dict]:
    """Answerable unique-gold items flagged caption_bm25_should_fail."""
    out: list[dict] = []
    for it in items:
        if not it.get("gold_figure_ids"):
            continue
        if it.get("expected_status") == "abstain":
            continue
        if not it.get("caption_bm25_should_fail"):
            continue
        out.append(it)
    return out


def split_by_figure_id(
    items: list[dict],
    *,
    train_frac: float = TRAIN_FRAC,
    seed: int = HOLD_SEED,
) -> dict[str, Any]:
    """80/20 by unique primary gold figure id. Queries follow their figure."""
    pool = answerable_fail_items(items)
    fids = sorted({primary_gold_id(it) for it in pool if primary_gold_id(it)})
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(fids))
    n_train = int(len(fids) * float(train_frac))
    train_figs = {fids[i] for i in order[:n_train]}
    hold_figs = {fids[i] for i in order[n_train:]}
    train = [it for it in pool if primary_gold_id(it) in train_figs]
    hold = [it for it in pool if primary_gold_id(it) in hold_figs]
    overlap = train_figs & hold_figs
    if overlap:
        raise RuntimeError(f"figure-id leak in split: {sorted(overlap)[:8]}")
    return {
        "seed": int(seed),
        "train_frac": float(train_frac),
        "train_items": train,
        "holdout_items": hold,
        "train_figure_ids": sorted(train_figs),
        "holdout_figure_ids": sorted(hold_figs),
        "n_train_figures": len(train_figs),
        "n_holdout_figures": len(hold_figs),
        "n_train_queries": len(train),
        "n_holdout_queries": len(hold),
    }


def negative_index_pool(
    gallery_ids: list[str],
    *,
    holdout_figure_ids: set[str],
    positive_id: str,
) -> list[int]:
    """In-gallery negatives: never the positive, never a held-out gold figure."""
    blocked = set(holdout_figure_ids)
    blocked.add(positive_id)
    return [i for i, fid in enumerate(gallery_ids) if fid not in blocked]
