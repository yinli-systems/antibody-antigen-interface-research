"""Primary multimodal eval: visual figure QA + pixel retrieval.

Caption-as-query (Protocol A/B) stays in evals/run_multimodal.py as appendix.
This task: visual question → figure image, then cite-or-abstain VQA.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from abcite_core.clip_encoder import biomedclip_skip_reason, build_vision_encoder, try_load_biomedclip
from abcite_core.colpali_encoder import try_load_colpali
from abcite_core.figure_gallery import FigureGallery, load_figure_records
from abcite_core.visual_qa import evaluate_visual_qa, load_visual_qa_gold
from abcite_core.vision_flags import DEFAULT_RERANK_K

ROOT = Path(__file__).resolve().parents[1]


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def _print(report: dict) -> None:
    print(json.dumps(report, indent=2))
    print(
        f"protocol={report.get('protocol')} gallery={report.get('n_gallery')} "
        f"queries={report.get('n_queries')} answerable={report.get('n_answerable')} "
        f"encoder={report.get('encoder')} gpu={report.get('gpu')}\n"
        f"caption_bm25 R@1/5/10={report.get('caption_bm25')}\n"
        f"vision T→I R@1/5/10={report.get('clip_image_only')}\n"
        f"vqa={report.get('vqa')}\n"
        f"vision_beats_bm25_r@1={report.get('vision_beats_bm25_r@1')}"
    )


def _four_column(openclip: dict, biomed: dict | None, colpali: dict | None) -> dict:
    bm25 = openclip["caption_bm25"]
    return {
        "caption_bm25": bm25,
        "biomedclip_t2i": (biomed or {}).get("clip_image_only"),
        "openclip_t2i": openclip["clip_image_only"],
        "colpali_t2i": (colpali or {}).get("clip_image_only") if colpali else None,
        "vision_rerank_biomedclip": (biomed or openclip).get("vision_rerank"),
        "vqa": (biomed or openclip).get("vqa"),
        "caption_bm25_should_fail_subset": (biomed or openclip).get(
            "caption_bm25_should_fail_subset"
        ),
        "vision_should_fail_subset": (biomed or openclip).get("vision_should_fail_subset"),
        "vision_beats_bm25_r@1": bool(
            ((biomed or openclip).get("clip_image_only") or {}).get("recall@1", 0)
            > (bm25 or {}).get("recall@1", 0)
        ),
        "vision_beats_bm25_ndcg@5": bool(
            ((biomed or openclip).get("clip_image_only") or {}).get("ndcg@5", 0)
            > (bm25 or {}).get("ndcg@5", 0)
        ),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--rerank-k", type=int, default=DEFAULT_RERANK_K)
    ap.add_argument("--try-colpali", action="store_true")
    args = ap.parse_args()
    recs = load_figure_records(prefer_cache=not args.fixture)
    vision = build_vision_encoder()
    gallery = FigureGallery(recs, vision)
    items = load_visual_qa_gold(set(gallery.ids), recs)
    combined: dict = {
        "protocol": "visual_qa",
        "n_gallery": gallery.n,
        "n_readable": len(recs),
        "n_queries": len(items),
        "n_answerable": sum(1 for it in items if it.get("gold_figure_ids")),
        "encoder": getattr(gallery.vision, "name", "unknown"),
        "gpu": _gpu_name(),
        "source": recs[0].source if recs else "",
        "fusion_default": "off",
        "rerank_k": args.rerank_k,
    }
    report = evaluate_visual_qa(gallery, items, rerank_k=args.rerank_k)
    report["gpu"] = combined["gpu"]
    _print(report)
    combined["openclip"] = report
    dest_one = ROOT / "evals" / "visual_qa_latest.json"
    dest_one.write_text(json.dumps(report, indent=2), encoding="utf-8")

    bio, skip = try_load_biomedclip()
    bio_report = None
    if bio is None:
        combined["biomedclip_skip"] = skip or biomedclip_skip_reason()
        print(f"BiomedCLIP: {combined['biomedclip_skip']}")
    else:
        bio_gal = FigureGallery(recs, bio)
        bio_items = load_visual_qa_gold(set(bio_gal.ids), recs)
        bio_report = evaluate_visual_qa(bio_gal, bio_items, rerank_k=args.rerank_k)
        bio_report["gpu"] = combined["gpu"]
        bio_report["encoder"] = getattr(bio, "name", "biomedclip")
        combined["biomedclip"] = bio_report
        _print(bio_report)

    combined["four_column"] = _four_column(report, bio_report, None)
    dest = ROOT / "evals" / "visual_qa_protocols_latest.json"
    dest.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    dest_one.write_text(json.dumps(bio_report or report, indent=2), encoding="utf-8")

    col_report = None
    col_skip = None
    try_col = args.try_colpali or os.environ.get("ABCITE_COLPALI", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    encode_ok = os.environ.get("ABCITE_COLPALI_ENCODE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if try_col:
        col, col_skip = try_load_colpali()
        if col is None:
            combined["colpali_skip"] = col_skip
            print(f"ColPali: {col_skip}")
        elif gallery.n > 200 and not encode_ok:
            combined["colpali_skip"] = (
                f"ColPali loaded ({getattr(col, 'name', 'colpali')}) but late-interaction "
                f"encode of n={gallery.n} figures is the next GPU job. "
                "Re-run with ABCITE_COLPALI_ENCODE=1 python evals/run_visual_qa.py --try-colpali"
            )
            print(f"ColPali: {combined['colpali_skip']}")
        else:
            col_gal = FigureGallery(recs, col)
            col_items = load_visual_qa_gold(set(col_gal.ids), recs)
            col_report = evaluate_visual_qa(col_gal, col_items, rerank_k=args.rerank_k)
            col_report["gpu"] = combined["gpu"]
            col_report["encoder"] = getattr(col, "name", "colpali")
            combined["colpali"] = col_report
            _print(col_report)
    else:
        combined["colpali_skip"] = (
            "ColPali not requested this run. Next GPU job: ABCITE_COLPALI=1 "
            "ABCITE_COLPALI_ENCODE=1 python evals/run_visual_qa.py --try-colpali"
        )

    combined["four_column"] = _four_column(report, bio_report, col_report)
    dest = ROOT / "evals" / "visual_qa_protocols_latest.json"
    dest.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    dest_one.write_text(
        json.dumps(bio_report or report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
