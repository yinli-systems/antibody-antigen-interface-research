"""Two multimodal protocols on the same PMC OA gallery.

Protocol A — caption-as-query (SciMMIR-style pair retrieval):
  query = accession-stripped figure caption; gold = that figure id.
  Image encoder never sees caption text. Caption BM25 uses captions as documents.

Protocol B — unique-gold visual questions:
  caption-free visual queries; gold remapped from committed fix-* ids to pmc-*
  where possible. Reported as a second table; not mixed with protocol A.

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
)
from abcite_core.holdout import query_contains_pdb, visual_query_ok

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold" / "multimodal_visual.jsonl"


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


def evaluate(
    gallery: FigureGallery, items: list[dict], *, protocol: str = "unique_gold_visual"
) -> dict:
    questions = [it["question"] for it in items]
    clip_mat = gallery.clip_score_matrix(questions) if items else None
    bm25_mat = (
        np.stack([gallery.caption_bm25_scores(q) for q in questions], axis=0)
        if items
        else None
    )
    out: dict[str, dict] = {}
    for name in ("caption_bm25", "clip_image_only", "fusion"):
        r1 = r5 = r10 = 0.0
        for qi, it in enumerate(items):
            if name == "clip_image_only" and clip_mat is not None:
                scores = clip_mat[qi]
            elif name == "caption_bm25" and bm25_mat is not None:
                scores = bm25_mat[qi]
            elif name == "fusion" and clip_mat is not None and bm25_mat is not None:
                scores = _rrf(bm25_mat[qi]) + _rrf(clip_mat[qi])
            else:
                scores = gallery.fusion_scores(it["question"])
            ranked = gallery.ranked_ids(scores)
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
    notes = {
        "caption_as_query": (
            "Protocol A: caption-as-query pair retrieval. Query is the figure caption "
            "with PDB / UniProt / RefSeq tokens stripped. Gold is that figure id. "
            "clip_image_only is T→I; captions never enter the image encoder. "
            "caption_bm25 is lexical captions (SciMMIR-like text baseline), not CLIP."
        ),
        "unique_gold_visual": (
            "Protocol B: unique-gold visual questions (caption-free, no PDB). "
            "Committed fix-* gold is remapped to pmc-* via href when the 1200 cache is used. "
            "Not mixed with caption-as-query. 14 SVG cite-id match is a demo, not this protocol."
        ),
    }
    return {
        "protocol": protocol,
        "n_gallery": gallery.n,
        "n_queries": len(items),
        "encoder": getattr(gallery.vision, "name", "unknown"),
        "gpu": _gpu_name(),
        "caption_bm25": out["caption_bm25"],
        "clip_image_only": out["clip_image_only"],
        "fusion": out["fusion"],
        "mean_gap_image_i2i": round(gallery.mean_rank_gap_image(), 6),
        "mean_gap_clip_text_to_image": round(mean_text_gap(gallery, questions), 6),
        "biomedclip_skip": biomedclip_skip_reason(),
        "note": notes.get(protocol, notes["unique_gold_visual"]),
        "published_ranges": {
            "scimmir_clip_ft_hits@10_t2i": (
                "~0.135 (ALL, different 530k gallery; not a target to match)"
            ),
            "scimmir_zs_clip_hits@10": "~0.0067 on 16k test (pair retrieval, not class)",
            "scimmir_zs_vlm_hits@10": "<0.0023 typical",
            "biomedclip_pmc_r@1_r@5": "~0.56 / ~0.77 on ~726k pairs (not this gallery)",
        },
    }


def _print_table(title: str, report: dict) -> None:
    print(title)
    print(json.dumps(report, indent=2))
    print(
        f"protocol={report.get('protocol')} gallery={report['n_gallery']} "
        f"queries={report['n_queries']} encoder={report['encoder']} gpu={report.get('gpu')}\n"
        f"caption_bm25 R@1/5/10={report['caption_bm25']}\n"
        f"clip_image_only R@1/5/10={report['clip_image_only']}\n"
        f"fusion R@1/5/10={report['fusion']}\n"
        f"mean_gap i2i={report['mean_gap_image_i2i']} t2i={report['mean_gap_clip_text_to_image']}\n"
        f"biomedclip={report.get('biomedclip') or report.get('biomedclip_skip')}"
    )


def _run_one(gallery: FigureGallery, items: list[dict], protocol: str, dest: Path) -> dict:
    report = evaluate(gallery, items, protocol=protocol)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(f"=== {protocol} ===", report)
    return report


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
        help="attempt BiomedCLIP on caption-as-query (download if ABCITE_BIOMEDCLIP_DOWNLOAD=1)",
    )
    args = ap.parse_args()
    recs = load_figure_records(prefer_cache=not args.fixture)
    vision = build_vision_encoder()
    gallery = FigureGallery(recs, vision)
    combined: dict = {
        "n_gallery": gallery.n,
        "encoder": getattr(gallery.vision, "name", "unknown"),
        "gpu": _gpu_name(),
        "source": recs[0].source if recs else "",
    }

    if args.protocol in {"caption", "both"}:
        cap_items = [
            it
            for it in caption_as_query_items(recs)
            if not query_contains_pdb(it["question"]) and "1n8z" not in it["question"].lower()
        ]
        cap_report = _run_one(
            gallery,
            cap_items,
            "caption_as_query",
            ROOT / "evals" / "multimodal_caption_latest.json",
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
            )
            combined["unique_gold_visual"] = vis_report

    try_bio = args.try_biomedclip or os.environ.get("ABCITE_BIOMEDCLIP_DOWNLOAD", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if try_bio and args.protocol in {"caption", "both"}:
        bio, skip = try_load_biomedclip()
        if bio is None:
            combined["biomedclip_skip"] = skip
            if "caption_as_query" in combined:
                combined["caption_as_query"]["biomedclip_skip"] = skip
                (ROOT / "evals" / "multimodal_caption_latest.json").write_text(
                    json.dumps(combined["caption_as_query"], indent=2), encoding="utf-8"
                )
            print(f"BiomedCLIP: {skip}")
        else:
            bio_gal = FigureGallery(recs, bio)
            bio_items = caption_as_query_items(recs)
            bio_report = evaluate(bio_gal, bio_items, protocol="caption_as_query")
            bio_report["encoder"] = getattr(bio, "name", "biomedclip")
            combined["biomedclip_caption_as_query"] = bio_report
            if "caption_as_query" in combined:
                combined["caption_as_query"]["biomedclip"] = {
                    "encoder": bio_report["encoder"],
                    "caption_bm25": bio_report["caption_bm25"],
                    "clip_image_only": bio_report["clip_image_only"],
                    "fusion": bio_report["fusion"],
                    "mean_gap_image_i2i": bio_report["mean_gap_image_i2i"],
                    "mean_gap_clip_text_to_image": bio_report["mean_gap_clip_text_to_image"],
                }
                combined["caption_as_query"]["biomedclip_skip"] = None
                (ROOT / "evals" / "multimodal_caption_latest.json").write_text(
                    json.dumps(combined["caption_as_query"], indent=2), encoding="utf-8"
                )
            _print_table("=== caption_as_query / BiomedCLIP ===", bio_report)

    dest = ROOT / "evals" / "multimodal_protocols_latest.json"
    dest.write_text(json.dumps(combined, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
