"""CLIP rank-1/rank-2 gap and hubness over figure/structure chunks.

Does not claim SOTA. Tiny hashing CLIP often has collapsed neighborhoods.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from abcite_core.retrieval import HybridIndex
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def figure_chunks(index: HybridIndex):
    return [
        c
        for c in index.chunks
        if c.modality.value in {"figure", "structure"} and c.image_path
    ]


def clip_contrast(index: HybridIndex | None = None) -> dict:
    if index is None:
        index = HybridIndex(load_corpus(ROOT / "corpus"))
    figs = figure_chunks(index)
    per_query: list[dict] = []
    rank1_ids: list[str] = []
    for fig in figs:
        path = Path(fig.image_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            continue
        hits = index.search_by_image(path, top_k=min(5, max(2, len(index.chunks))))
        if len(hits) < 2:
            continue
        r1, r2 = hits[0], hits[1]
        gap = float(r1.score - r2.score)
        rank1_ids.append(r1.chunk.chunk_id)
        per_query.append(
            {
                "query_fig": fig.chunk_id,
                "rank1_id": r1.chunk.chunk_id,
                "rank1_score": round(float(r1.score), 6),
                "rank2_id": r2.chunk.chunk_id,
                "rank2_score": round(float(r2.score), 6),
                "gap": round(gap, 6),
                "self_hit": r1.chunk.chunk_id == fig.chunk_id,
            }
        )
    hub = Counter(rank1_ids)
    hub_figs = {cid: n for cid, n in hub.items() if n > 1}
    gaps = [row["gap"] for row in per_query]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    return {
        "encoder": getattr(index.vision, "name", "unknown"),
        "n_figure_queries": len(per_query),
        "mean_gap_rank1_minus_rank2": round(mean_gap, 6),
        "min_gap": round(min(gaps), 6) if gaps else 0.0,
        "self_hit_rate": (
            sum(1 for r in per_query if r["self_hit"]) / len(per_query) if per_query else 0.0
        ),
        "hubness_rank1_counts": dict(hub),
        "hub_figures_rank1_for_others": hub_figs,
        "clip_sota_claim": False,
        "note": (
            "Gap and hubness are diagnostics. High cosine with tiny gap is collapse, "
            "not domain SOTA. ABCITE_CLIP_MODEL=biomedclip is a stub (no download)."
        ),
        "per_query": per_query,
    }


def main() -> None:
    report = clip_contrast()
    slim = {k: v for k, v in report.items() if k != "per_query"}
    print(slim)
    print(f"per_query n={len(report['per_query'])}")


if __name__ == "__main__":
    main()
