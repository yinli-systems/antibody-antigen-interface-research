"""Two multimodal protocols on the same PMC OA gallery.

Protocol A — held-out caption-as-query (SciMMIR-style pair retrieval):
  split by PMCID (default 80/20). Query captions come from query papers.
  Gallery is the full image set so query-paper figures remain retrievable
  (index papers are distractors). Query = accession-stripped caption.
  Image encoder never sees caption text. Caption BM25 uses captions as documents.
  Product column is BM25; CLIP T→I is diagnostic; fusion is an ablation
  (off by default). vision_rerank reorders BM25 top-K by CLIP cosine.

Protocol B — unique-gold visual questions:
  caption-free visual queries; gold remapped from committed fix-* ids to pmc-*
  where possible. Small-n; not mixed with protocol A.

Published ranges (different galleries; do not claim a match):
  SciMMIR CLIP-base FT T→I Hits@10 ~13.5% (ALL); ZS CLIP Hits@10 ~0.67%
  BiomedCLIP PMC-OA retrieval R@1 ~56% / R@5 ~77% on ~726k pairs
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from abcite_core.clip_encoder import (
    biomedclip_skip_reason,
    build_vision_encoder,
    try_load_biomedclip,
)
from abcite_core.figure_gallery import (
    FigureGallery,
    FigureRec,
    _rrf,
    caption_as_query_items,
    fix_to_gallery_ids,
    load_figure_records,
    mean_text_gap,
    recall_at_k,
    vision_rerank_scores,
)
from abcite_core.holdout import query_contains_pdb, split_by_paper, visual_query_ok
from abcite_core.vision_flags import DEFAULT_RERANK_K

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold" / "multimodal_visual.jsonl"

_PUBLISHED = {
    "scimmir_clip_ft_hits@10_t2i": (
        "~0.135 (ALL, different 530k gallery; not a target to match)"
    ),
    "scimmir_zs_clip_hits@10": "~0.0067 on 16k test (pair retrieval, not class)",
    "scimmir_zs_vlm_hits@10": "<0.0023 typical",
    "biomedclip_pmc_r@1_r@5": "~0.56 / ~0.77 on ~726k pairs (not this gallery)",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def load_visual_gold(gallery_ids: set[str], recs: list[FigureRec] | None = None) -> list[dict]:
    rows = _read_jsonl(GOLD)
    alias = fix_to_gallery_ids(recs) if recs else {}
    ok = []
    for row in rows:
        q = row["question"]
        if query_contains_pdb(q):
            raise SystemExit(f"image-only protocol forbids PDB in query: {row.get('id')} {q!r}")
        if not visual_query_ok(q):
            raise SystemExit(f"query is not visual-content: {row.get('id')} {q!r}")
        gold_ids: list[str] = []
        seen: set[str] = set()
        for g in row["gold_figure_ids"]:
            mapped = g if g in gallery_ids else alias.get(g)
            if mapped and mapped in gallery_ids and mapped not in seen:
                gold_ids.append(mapped)
                seen.add(mapped)
        if gold_ids:
            row = dict(row)
            row["gold_figure_ids"] = gold_ids
            row["protocol"] = "unique_gold_visual"
            ok.append(row)
    return ok


def _recall_block(scores_list: list[np.ndarray], items: list[dict], gallery: FigureGallery) -> dict:
    r1 = r5 = r10 = 0.0
    for scores, it in zip(scores_list, items, strict=True):
        ranked = gallery.ranked_ids(scores)
        gold = it["gold_figure_ids"]
        r1 += recall_at_k(ranked, gold, 1)
        r5 += recall_at_k(ranked, gold, 5)
        r10 += recall_at_k(ranked, gold, 10)
    n = max(1, len(items))
    return {
        "recall@1": round(r1 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
    }


def _rerank_helps(bm25: dict, rerank: dict) -> dict:
    deltas = {}
    helps = {}
    for k in ("recall@1", "recall@5", "recall@10"):
        d = round(float(rerank[k]) - float(bm25[k]), 4)
        deltas[k] = d
        helps[k] = d > 0
    return {
        "helps_any": any(helps.values()),
        "hurts_any": any(d < 0 for d in deltas.values()),
        "helps": helps,
        "delta": deltas,
        "note": (
            "Positive delta means CLIP rerank of BM25 top-K beat caption BM25. "
            "Fusion RRF is a separate ablation and is off in the product default."
        ),
    }


def evaluate(
    gallery: FigureGallery,
    items: list[dict],
    *,
    protocol: str = "unique_gold_visual",
    rerank_k: int = DEFAULT_RERANK_K,
) -> dict:
    questions = [it["question"] for it in items]
    clip_mat = gallery.clip_score_matrix(questions) if items else None
    bm25_mat = (
        np.stack([gallery.caption_bm25_scores(q) for q in questions], axis=0)
        if items
        else None
    )
    cols: dict[str, dict] = {}
    if items and bm25_mat is not None and clip_mat is not None:
        bm25_rows = [bm25_mat[i] for i in range(len(items))]
        clip_rows = [clip_mat[i] for i in range(len(items))]
        rerank_rows = [
            vision_rerank_scores(bm25_mat[i], clip_mat[i], k=rerank_k) for i in range(len(items))
        ]
        fusion_rows = [_rrf(bm25_mat[i]) + _rrf(clip_mat[i]) for i in range(len(items))]
        cols["caption_bm25"] = _recall_block(bm25_rows, items, gallery)
        cols["clip_image_only"] = _recall_block(clip_rows, items, gallery)
        cols["vision_rerank"] = _recall_block(rerank_rows, items, gallery)
        cols["fusion"] = _recall_block(fusion_rows, items, gallery)
    else:
        empty = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0}
        cols = {
            "caption_bm25": dict(empty),
            "clip_image_only": dict(empty),
            "vision_rerank": dict(empty),
            "fusion": dict(empty),
        }
    notes = {
        "caption_as_query": (
            "Protocol A: held-out caption-as-query pair retrieval, split by PMCID. "
            "Query is the figure caption with PDB / UniProt / RefSeq tokens stripped. "
            "Gold is that figure id. clip_image_only is T→I over the eval gallery; "
            "captions never enter the image encoder. caption_bm25 is lexical captions. "
            "vision_rerank reorders BM25 top-K by CLIP cosine (K=50 unless overridden). "
            "fusion RRF is an ablation and is OFF in the product default."
        ),
        "unique_gold_visual": (
            "Protocol B: unique-gold visual questions (caption-free, no PDB). "
            "Committed fix-* gold is remapped to pmc-* via href when the PMC cache is used. "
            "Small-n. Not mixed with caption-as-query. 14 SVG cite-id match is a demo, "
            "not this protocol."
        ),
    }
    report = {
        "protocol": protocol,
        "n_gallery": gallery.n,
        "n_queries": len(items),
        "encoder": getattr(gallery.vision, "name", "unknown"),
        "gpu": _gpu_name(),
        "rerank_k": rerank_k,
        "caption_bm25": cols["caption_bm25"],
        "clip_image_only": cols["clip_image_only"],
        "vision_rerank": cols["vision_rerank"],
        "fusion": cols["fusion"],
        "fusion_default": "off",
        "rerank_helps_bm25": _rerank_helps(cols["caption_bm25"], cols["vision_rerank"]),
        "mean_gap_image_i2i": round(gallery.mean_rank_gap_image(), 6),
        "mean_gap_clip_text_to_image": round(mean_text_gap(gallery, questions), 6),
        "biomedclip_skip": biomedclip_skip_reason(),
        "note": notes.get(protocol, notes["unique_gold_visual"]),
        "published_ranges": _PUBLISHED,
    }
    return report


def _print_table(title: str, report: dict) -> None:
    print(title)
    print(json.dumps(report, indent=2))
    helps = report.get("rerank_helps_bm25") or {}
    print(
        f"protocol={report.get('protocol')} gallery={report['n_gallery']} "
        f"queries={report['n_queries']} encoder={report['encoder']} gpu={report.get('gpu')}\n"
        f"caption_bm25 R@1/5/10={report['caption_bm25']}\n"
        f"clip_image_only T→I R@1/5/10={report['clip_image_only']}\n"
        f"vision_rerank top{report.get('rerank_k')} R@1/5/10={report['vision_rerank']}\n"
        f"fusion ablation R@1/5/10={report['fusion']}\n"
        f"rerank_helps_bm25={helps.get('helps')} delta={helps.get('delta')}\n"
        f"mean_gap i2i={report['mean_gap_image_i2i']} t2i={report['mean_gap_clip_text_to_image']}\n"
        f"biomedclip={report.get('biomedclip') or report.get('biomedclip_skip')}"
    )


def _run_one(
    gallery: FigureGallery,
    items: list[dict],
    protocol: str,
    dest: Path,
    *,
    rerank_k: int,
) -> dict:
    report = evaluate(gallery, items, protocol=protocol, rerank_k=rerank_k)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(f"=== {protocol} ===", report)
    return report


def _caption_items(recs: list[FigureRec]) -> list[dict]:
    return [
        it
        for it in caption_as_query_items(recs)
        if not query_contains_pdb(it["question"]) and "1n8z" not in it["question"].lower()
    ]


def _four_column(openclip: dict, biomed: dict | None) -> dict:
    """BM25 / BiomedCLIP T→I / OpenCLIP T→I / rerank-on-BM25-top50."""
    bm25 = openclip["caption_bm25"]
    rerank_src = biomed if biomed is not None else openclip
    rerank_name = (
        "biomedclip_on_bm25_top50" if biomed is not None else "openclip_on_bm25_top50"
    )
    return {
        "caption_bm25": bm25,
        "biomedclip_t2i_full_gallery": (biomed or {}).get("clip_image_only"),
        "openclip_t2i_full_gallery": openclip["clip_image_only"],
        "vision_rerank": rerank_src["vision_rerank"],
        "vision_rerank_encoder": rerank_src.get("encoder"),
        "vision_rerank_name": rerank_name,
        "rerank_helps_bm25": rerank_src.get("rerank_helps_bm25"),
        "fusion_ablation_biomedclip": (biomed or {}).get("fusion"),
        "fusion_ablation_openclip": openclip.get("fusion"),
        "fusion_default": "off",
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true", help="use committed 8–16 OA thumbnails only")
    ap.add_argument(
        "--protocol",
        choices=["caption", "visual", "both"],
        default="both",
        help="caption-as-query, unique-gold visual, or both (default)",
    )
    ap.add_argument(
        "--try-biomedclip",
        action="store_true",
        help="attempt BiomedCLIP (also on if weights are already cached)",
    )
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--holdout-seed", type=int, default=7)
    ap.add_argument(
        "--gallery-mode",
        choices=["full", "query-papers"],
        default="full",
        help="full = all figures (query papers as needles + index-paper distractors)",
    )
    ap.add_argument("--rerank-k", type=int, default=DEFAULT_RERANK_K)
    ap.add_argument(
        "--no-holdout",
        action="store_true",
        help="transductive caption-as-query on the whole gallery (legacy ablation)",
    )
    args = ap.parse_args()
    recs = load_figure_records(prefer_cache=not args.fixture)
    split = split_by_paper(recs, query_frac=args.holdout_frac, seed=args.holdout_seed)
    if args.gallery_mode == "query-papers":
        gallery_recs = list(split.query_recs)
    else:
        gallery_recs = recs
    vision = build_vision_encoder()
    gallery = FigureGallery(gallery_recs, vision)
    holdout_info = split.as_dict()
    holdout_info["gallery_mode"] = args.gallery_mode
    holdout_info["held_out"] = not args.no_holdout
    combined: dict = {
        "n_gallery": gallery.n,
        "n_readable": len(recs),
        "encoder": getattr(gallery.vision, "name", "unknown"),
        "gpu": _gpu_name(),
        "source": recs[0].source if recs else "",
        "holdout": holdout_info,
        "fusion_default": "off",
        "rerank_k": args.rerank_k,
    }

    cap_pool = recs if args.no_holdout else list(split.query_recs)
    cap_items = _caption_items(cap_pool)
    # Gold must live in the eval gallery.
    gal_ids = set(gallery.ids)
    cap_items = [it for it in cap_items if it["gold_figure_ids"][0] in gal_ids]
    combined["holdout"]["n_queries"] = len(cap_items)

    if args.protocol in {"caption", "both"}:
        cap_report = _run_one(
            gallery,
            cap_items,
            "caption_as_query",
            ROOT / "evals" / "multimodal_caption_latest.json",
            rerank_k=args.rerank_k,
        )
        cap_report["holdout"] = combined["holdout"]
        (ROOT / "evals" / "multimodal_caption_latest.json").write_text(
            json.dumps(cap_report, indent=2), encoding="utf-8"
        )
        combined["caption_as_query"] = cap_report

    if args.protocol in {"visual", "both"}:
        vis_items = load_visual_gold(set(gallery.ids), recs)
        if not vis_items:
            print("unique-gold visual: no queries matched this gallery (no class-query fallback)")
        else:
            vis_report = _run_one(
                gallery,
                vis_items,
                "unique_gold_visual",
                ROOT / "evals" / "multimodal_visual_latest.json",
                rerank_k=args.rerank_k,
            )
            vis_report["small_n"] = True
            vis_report["n_queries_label"] = f"small-n n={len(vis_items)}"
            (ROOT / "evals" / "multimodal_visual_latest.json").write_text(
                json.dumps(vis_report, indent=2), encoding="utf-8"
            )
            combined["unique_gold_visual"] = vis_report

    weights_cached = bool(
        os.environ.get("ABCITE_BIOMEDCLIP_DOWNLOAD", "").lower() in {"1", "true", "yes"}
        or args.try_biomedclip
    )
    try_bio = args.try_biomedclip or weights_cached
    # Always try cached BiomedCLIP for the four-column table (no Hub download unless flagged).
    if args.protocol in {"caption", "both"} or args.protocol in {"visual", "both"}:
        bio, skip = try_load_biomedclip()
        if bio is None:
            combined["biomedclip_skip"] = skip
            print(f"BiomedCLIP: {skip}")
            if "caption_as_query" in combined:
                combined["caption_as_query"]["biomedclip_skip"] = skip
                combined["four_column"] = _four_column(combined["caption_as_query"], None)
        else:
            bio_gal = FigureGallery(gallery_recs, bio)
            if args.protocol in {"caption", "both"}:
                bio_report = evaluate(
                    bio_gal, cap_items, protocol="caption_as_query", rerank_k=args.rerank_k
                )
                bio_report["encoder"] = getattr(bio, "name", "biomedclip")
                bio_report["holdout"] = combined["holdout"]
                combined["biomedclip_caption_as_query"] = bio_report
                if "caption_as_query" in combined:
                    combined["caption_as_query"]["biomedclip"] = {
                        "encoder": bio_report["encoder"],
                        "caption_bm25": bio_report["caption_bm25"],
                        "clip_image_only": bio_report["clip_image_only"],
                        "vision_rerank": bio_report["vision_rerank"],
                        "fusion": bio_report["fusion"],
                        "rerank_helps_bm25": bio_report["rerank_helps_bm25"],
                        "mean_gap_image_i2i": bio_report["mean_gap_image_i2i"],
                        "mean_gap_clip_text_to_image": bio_report["mean_gap_clip_text_to_image"],
                    }
                    combined["caption_as_query"]["biomedclip_skip"] = None
                    combined["four_column"] = _four_column(combined["caption_as_query"], bio_report)
                    (ROOT / "evals" / "multimodal_caption_latest.json").write_text(
                        json.dumps(combined["caption_as_query"], indent=2), encoding="utf-8"
                    )
                _print_table("=== caption_as_query / BiomedCLIP ===", bio_report)
            if args.protocol in {"visual", "both"} and combined.get("unique_gold_visual"):
                vis_items = load_visual_gold(set(bio_gal.ids), recs)
                vis_bio = evaluate(
                    bio_gal, vis_items, protocol="unique_gold_visual", rerank_k=args.rerank_k
                )
                vis_bio["small_n"] = True
                combined["biomedclip_unique_gold_visual"] = vis_bio
                combined["unique_gold_visual"]["biomedclip"] = {
                    "encoder": vis_bio["encoder"],
                    "caption_bm25": vis_bio["caption_bm25"],
                    "clip_image_only": vis_bio["clip_image_only"],
                    "vision_rerank": vis_bio["vision_rerank"],
                    "fusion": vis_bio["fusion"],
                    "rerank_helps_bm25": vis_bio["rerank_helps_bm25"],
                }
                (ROOT / "evals" / "multimodal_visual_latest.json").write_text(
                    json.dumps(combined["unique_gold_visual"], indent=2), encoding="utf-8"
                )
        _ = try_bio

    dest = ROOT / "evals" / "multimodal_protocols_latest.json"
    dest.write_text(json.dumps(combined, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
