"""CLIP rank-1/rank-2 gap and hubness over figure/structure chunks.

Caption BM25 is fused with image cosine (RRF). Tiny hashing CLIP often collapses;
caption fusion is the TEXT-side discrimination path in default CI.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from abcite_core.clip_encoder import biomedclip_skip_reason
from abcite_core.retrieval import HybridIndex
from ingest.chunking import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def figure_chunks(index: HybridIndex):
    return [
        c
        for c in index.chunks
        if c.modality.value in {"figure", "structure"} and c.image_path
    ]


def _gap_report(index: HybridIndex, fuse_captions: bool) -> dict:
    figs = figure_chunks(index)
    per_query: list[dict] = []
    rank1_ids: list[str] = []
    caption_gaps: list[float] = []
    for fig in figs:
        path = Path(fig.image_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            continue
        hits = index.search_by_image(
            path, top_k=min(5, max(2, len(index.chunks))), fuse_captions=fuse_captions
        )
        if len(hits) < 2:
            continue
        r1, r2 = hits[0], hits[1]
        gap = float(r1.score - r2.score)
        rank1_ids.append(r1.chunk.chunk_id)
        caption_gaps.append(float(r1.bm25 - r2.bm25))
        per_query.append(
            {
                "query_fig": fig.chunk_id,
                "rank1_id": r1.chunk.chunk_id,
                "rank1_score": round(float(r1.score), 6),
                "rank2_id": r2.chunk.chunk_id,
                "rank2_score": round(float(r2.score), 6),
                "gap": round(gap, 6),
                "self_hit": r1.chunk.chunk_id == fig.chunk_id,
                "rank1_image_cosine": round(float(r1.image), 6),
                "rank2_image_cosine": round(float(r2.image), 6),
            }
        )
    hub = Counter(rank1_ids)
    hub_figs = {cid: n for cid, n in hub.items() if n > 1}
    gaps = [row["gap"] for row in per_query]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    img_gaps = [
        float(r["rank1_image_cosine"] - r["rank2_image_cosine"]) for r in per_query
    ]
    return {
        "n_figure_queries": len(per_query),
        "mean_gap_rank1_minus_rank2": round(mean_gap, 6),
        "mean_image_cosine_gap": round(sum(img_gaps) / len(img_gaps), 6) if img_gaps else 0.0,
        "mean_caption_bm25_gap": round(sum(caption_gaps) / len(caption_gaps), 6) if caption_gaps else 0.0,
        "min_gap": round(min(gaps), 6) if gaps else 0.0,
        "self_hit_rate": (
            sum(1 for r in per_query if r["self_hit"]) / len(per_query) if per_query else 0.0
        ),
        "hubness_rank1_counts": dict(hub),
        "hub_figures_rank1_for_others": hub_figs,
        "per_query": per_query,
    }


def clip_contrast(index: HybridIndex | None = None) -> dict:
    if index is None:
        index = HybridIndex(load_corpus(ROOT / "corpus"))
    fused = _gap_report(index, fuse_captions=True)
    image_only = _gap_report(index, fuse_captions=False)
    skip = biomedclip_skip_reason()
    return {
        "encoder": getattr(index.vision, "name", "unknown"),
        "caption_fusion": True,
        "n_figure_queries": fused["n_figure_queries"],
        "mean_gap_rank1_minus_rank2": fused["mean_gap_rank1_minus_rank2"],
        "mean_gap_image_only": image_only["mean_gap_rank1_minus_rank2"],
        "mean_image_cosine_gap": fused["mean_image_cosine_gap"],
        "mean_caption_bm25_gap": fused["mean_caption_bm25_gap"],
        "min_gap": fused["min_gap"],
        "self_hit_rate": fused["self_hit_rate"],
        "hubness_rank1_counts": fused["hubness_rank1_counts"],
        "hub_figures_rank1_for_others": fused["hub_figures_rank1_for_others"],
        "clip_sota_claim": False,
        "biomedclip_skip": skip,
        "note": (
            "Gap and hubness are diagnostics. Image-only cosine on similar SVGs can collapse "
            "(~0.97 neighbors). Caption BM25 + image RRF is the contrast path. "
            "ABCITE_CLIP_MODEL=biomedclip loads only if weights are cached; otherwise skipped."
        ),
        "per_query": fused["per_query"],
        "image_only": {k: v for k, v in image_only.items() if k != "per_query"},
    }


def main() -> None:
    report = clip_contrast()
    slim = {k: v for k, v in report.items() if k not in {"per_query"}}
    print(slim)
    print(f"per_query n={len(report['per_query'])}")


if __name__ == "__main__":
    main()
