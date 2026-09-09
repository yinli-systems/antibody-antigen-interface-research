"""Visual multimodal retrieval: caption BM25 vs CLIP image-only vs fusion.

Not the 14-SVG cite-id demo. Queries are visual content (no PDB ids).
Image vectors are pixels only (captions held out of the encoder path).

Published ranges (different galleries; do not claim a match):
  SciMMIR CLIP-base FT T→I Hits@10 ~13.5% (ALL); many ZS VLMs Hits@10 <0.23%
  BiomedCLIP PMC-OA retrieval R@1 ~56% / R@5 ~77% on ~726k pairs
"""

from __future__ import annotations

import json
from pathlib import Path

from abcite_core.clip_encoder import biomedclip_skip_reason, build_vision_encoder
from abcite_core.figure_gallery import (
    FigureGallery,
    load_figure_records,
    mean_text_gap,
    recall_at_k,
    synthesize_visual_queries,
)
from abcite_core.holdout import query_contains_pdb, visual_query_ok

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold" / "multimodal_visual.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_visual_gold(gallery_ids: set[str]) -> list[dict]:
    rows = _read_jsonl(GOLD)
    ok = []
    for row in rows:
        q = row["question"]
        if query_contains_pdb(q):
            raise SystemExit(f"image-only protocol forbids PDB in query: {row.get('id')} {q!r}")
        if not visual_query_ok(q):
            raise SystemExit(f"query is not visual-content: {row.get('id')} {q!r}")
        gold_ids = [g for g in row["gold_figure_ids"] if g in gallery_ids]
        if gold_ids:
            row = dict(row)
            row["gold_figure_ids"] = gold_ids
            ok.append(row)
    return ok


def evaluate(gallery: FigureGallery, items: list[dict]) -> dict:
    cols = {
        "caption_bm25": gallery.caption_bm25_scores,
        "clip_image_only": gallery.clip_image_only_scores,
        "fusion": gallery.fusion_scores,
    }
    out: dict[str, dict] = {}
    questions = [it["question"] for it in items]
    for name, scorer in cols.items():
        r1 = r5 = r10 = 0.0
        for it in items:
            ranked = gallery.ranked_ids(scorer(it["question"]))
            gold = it["gold_figure_ids"]
            r1 += recall_at_k(ranked, gold, 1)
            r5 += recall_at_k(ranked, gold, 5)
            r10 += recall_at_k(ranked, gold, 10)
        n = max(1, len(items))
        out[name] = {
            "recall@1": round(r1 / n, 4),
            "recall@5": round(r5 / n, 4),
            "recall@10": round(r10 / n, 4),
        }
    return {
        "n_gallery": gallery.n,
        "n_queries": len(items),
        "encoder": getattr(gallery.vision, "name", "unknown"),
        "caption_bm25": out["caption_bm25"],
        "clip_image_only": out["clip_image_only"],
        "fusion": out["fusion"],
        "mean_gap_image_i2i": round(gallery.mean_rank_gap_image(), 6),
        "mean_gap_clip_text_to_image": round(mean_text_gap(gallery, questions), 6),
        "biomedclip_skip": biomedclip_skip_reason(),
        "note": (
            "14 SVG cite-id match is a demo, not this protocol. "
            "clip_image_only is T→I with captions/PDB held out of image vectors. "
            "caption_bm25 is lexical captions, not CLIP."
        ),
        "published_ranges": {
            "scimmir_clip_ft_hits@10_t2i": "~0.135 (ALL, different 530k gallery; not a target to match)",
            "scimmir_zs_vlm_hits@10": "<0.0023 typical",
            "biomedclip_pmc_r@1_r@5": "~0.56 / ~0.77 on ~726k pairs (not this gallery)",
        },
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true", help="use committed 8–16 OA thumbnails only")
    args = ap.parse_args()
    recs = load_figure_records(prefer_cache=not args.fixture)
    vision = build_vision_encoder()
    gallery = FigureGallery(recs, vision)
    items = load_visual_gold(set(gallery.ids))
    if len(items) < 5:
        items = synthesize_visual_queries(recs, limit=min(64, len(recs)))
    report = evaluate(gallery, items)
    dest = ROOT / "evals" / "multimodal_visual_latest.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        f"gallery={report['n_gallery']} encoder={report['encoder']}\n"
        f"caption_bm25 R@1/5/10={report['caption_bm25']}\n"
        f"clip_image_only R@1/5/10={report['clip_image_only']}\n"
        f"fusion R@1/5/10={report['fusion']}\n"
        f"mean_gap i2i={report['mean_gap_image_i2i']} t2i={report['mean_gap_clip_text_to_image']}"
    )


if __name__ == "__main__":
    main()
