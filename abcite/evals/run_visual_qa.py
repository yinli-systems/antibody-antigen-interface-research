"""Primary multimodal eval: visual figure QA + pixel retrieval.

Caption-as-query (Protocol A/B) stays in evals/run_multimodal.py as appendix.
This task: visual question → figure image, then cite-or-abstain VQA.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from abcite_core.clip_encoder import (
    biomedclip_skip_reason,
    build_vision_encoder,
    try_load_biomedclip,
)
from abcite_core.colpali_encoder import try_load_colpali
from abcite_core.figure_gallery import FigureGallery, load_figure_records
from abcite_core.vision_flags import DEFAULT_RERANK_K
from abcite_core.visual_qa import evaluate_visual_qa, load_visual_qa_gold
from abcite_core.vlm_vqa import DEFAULT_VLM_MODEL, vlm_enabled, vlm_weights_cached

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
    vqa = report.get("vqa") or {}
    print(
        f"protocol={report.get('protocol')} gallery={report.get('n_gallery')} "
        f"queries={report.get('n_queries')} answerable={report.get('n_answerable')} "
        f"encoder={report.get('encoder')} gpu={report.get('gpu')}\n"
        f"retrieval (T→I, not VQA, not ViDoRe) caption_bm25={report.get('caption_bm25')}\n"
        f"retrieval vision T→I={report.get('clip_image_only')}\n"
        f"vqa cite-or-abstain (separate from R@K): "
        f"e2e={vqa.get('end_to_end_accuracy')} "
        f"oracle={vqa.get('oracle_figure_accuracy')} "
        f"abstain={vqa.get('abstain_accuracy')} "
        f"cite@1={vqa.get('cite_top1')} "
        f"path={vqa.get('path')}\n"
        f"vision_beats_bm25_r@1={report.get('vision_beats_bm25_r@1')}"
    )


def _four_column(openclip: dict | None, biomed: dict | None, colpali: dict | None) -> dict:
    openclip = openclip or {}
    biomed = biomed or {}
    colpali = colpali or {}
    base = biomed or openclip or colpali
    bm25 = base.get("caption_bm25")
    vision = (biomed or openclip or colpali).get("clip_image_only")
    return {
        "caption_bm25": bm25,
        "biomedclip_t2i": biomed.get("clip_image_only") or None,
        "openclip_t2i": openclip.get("clip_image_only") or None,
        "colpali_t2i": colpali.get("clip_image_only") or None,
        "vision_rerank_biomedclip": (biomed or openclip or colpali).get("vision_rerank"),
        "vqa": (colpali or biomed or openclip).get("vqa"),
        "caption_bm25_should_fail_subset": (biomed or openclip or colpali).get(
            "caption_bm25_should_fail_subset"
        ),
        "vision_should_fail_subset": (biomed or openclip or colpali).get(
            "vision_should_fail_subset"
        ),
        "vision_beats_bm25_r@1": bool(
            (vision or {}).get("recall@1", 0) > (bm25 or {}).get("recall@1", 0)
        ),
        "vision_beats_bm25_ndcg@5": bool(
            (vision or {}).get("ndcg@5", 0) > (bm25 or {}).get("ndcg@5", 0)
        ),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--rerank-k", type=int, default=DEFAULT_RERANK_K)
    ap.add_argument("--try-colpali", action="store_true")
    ap.add_argument("--colpali-only", action="store_true", help="Skip CLIP/BiomedCLIP columns")
    ap.add_argument("--max-n", type=int, default=0, help="Cap gallery size (keep gold ids first)")
    ap.add_argument(
        "--vlm",
        action="store_true",
        help="Cite-or-abstain with Qwen2-VL generate (ABCITE_VLM=1). Not ColQwen2 MaxSim.",
    )
    args = ap.parse_args()
    if args.vlm:
        os.environ["ABCITE_VLM"] = "1"
    vlm_on = args.vlm or vlm_enabled()
    if vlm_on:
        cached = vlm_weights_cached()
        print(
            f"VLM VQA requested model={os.environ.get('ABCITE_VLM_MODEL', DEFAULT_VLM_MODEL)} "
            f"cached={'yes '+str(cached) if cached else 'NO — will fall back to heuristic'}",
            flush=True,
        )
    recs = load_figure_records(prefer_cache=not args.fixture)
    if args.max_n and args.max_n < len(recs):
        gold_ids: list[str] = []
        for it in load_visual_qa_gold({r.figure_id for r in recs}, recs, require_gold=False):
            gold_ids.extend(it.get("gold_figure_ids") or [])
        by = {r.figure_id: r for r in recs}
        keep: list = []
        seen: set[str] = set()
        for gid in gold_ids:
            if gid in by and gid not in seen:
                keep.append(by[gid])
                seen.add(gid)
        for rec in recs:
            if rec.figure_id not in seen:
                keep.append(rec)
                seen.add(rec.figure_id)
            if len(keep) >= args.max_n:
                break
        recs = keep
    dest_one = ROOT / "evals" / "visual_qa_latest.json"
    report = None
    bio_report = None
    gallery = None
    gpu = _gpu_name()
    if args.colpali_only:
        items = load_visual_qa_gold({r.figure_id for r in recs}, recs)
        n_gallery = len(recs)
        encoder_name = "colqwen2"
        print("colpali-only: skipping CLIP/BiomedCLIP", flush=True)
    else:
        vision = build_vision_encoder()
        gallery = FigureGallery(recs, vision)
        items = load_visual_qa_gold(set(gallery.ids), recs)
        n_gallery = gallery.n
        encoder_name = getattr(gallery.vision, "name", "unknown")
    combined: dict = {
        "protocol": "visual_qa",
        "n_gallery": n_gallery,
        "n_readable": len(recs),
        "n_queries": len(items),
        "n_answerable": sum(1 for it in items if it.get("gold_figure_ids")),
        "encoder": encoder_name,
        "gpu": gpu,
        "source": recs[0].source if recs else "",
        "fusion_default": "off",
        "rerank_k": args.rerank_k,
        "vlm": {
            "requested": vlm_on,
            "model": os.environ.get("ABCITE_VLM_MODEL", DEFAULT_VLM_MODEL),
            "weights_cached": bool(vlm_weights_cached()) if vlm_on else None,
            "note": (
                "VQA generate is Qwen2-VL-Instruct. Retrieval stays "
                "vidore/colqwen2-base + LoRA vidore/colqwen2-v1.0. "
                "Do not score Qwen2-VL as ColQwen2 (missing custom_text_proj). "
                "Do not mix PMC figure VQA with official ViDoRe 89.11."
            ),
        },
    }
    if args.colpali_only:
        combined["skipped_clip_columns"] = True
    else:
        # When VLM is on, skip expensive generate on CLIP columns; run it on ColQwen2.
        clip_score_vqa = not vlm_on
        report = evaluate_visual_qa(
            gallery, items, rerank_k=args.rerank_k, score_vqa=clip_score_vqa
        )
        report["gpu"] = combined["gpu"]
        _print(report)
        combined["openclip"] = report
        dest_one.write_text(json.dumps(report, indent=2), encoding="utf-8")

        bio, skip = try_load_biomedclip()
        if bio is None:
            combined["biomedclip_skip"] = skip or biomedclip_skip_reason()
            print(f"BiomedCLIP: {combined['biomedclip_skip']}")
        else:
            bio_gal = FigureGallery(recs, bio)
            bio_items = load_visual_qa_gold(set(bio_gal.ids), recs)
            bio_report = evaluate_visual_qa(
                bio_gal, bio_items, rerank_k=args.rerank_k, score_vqa=clip_score_vqa
            )
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
        elif n_gallery > 200 and not encode_ok:
            combined["colpali_skip"] = (
                f"ColPali loaded ({getattr(col, 'name', 'colpali')}) but late-interaction "
                f"encode of n={n_gallery} figures is the next GPU job. "
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
            "ABCITE_COLPALI_ENCODE=1 python evals/run_visual_qa.py --try-colpali. "
            "VLM cite-or-abstain (after GPU is free): ABCITE_VLM=1 "
            "python evals/run_visual_qa.py --try-colpali --colpali-only --vlm"
        )

    if vlm_on and col_report is None and gallery is not None and report is not None:
        if not (report.get("vqa") or {}).get("scored"):
            print("VLM: ColQwen2 column missing; scoring VQA on the CLIP gallery", flush=True)
            report = evaluate_visual_qa(gallery, items, rerank_k=args.rerank_k, score_vqa=True)
            report["gpu"] = combined["gpu"]
            combined["openclip"] = report
            _print(report)

    combined["four_column"] = _four_column(report, bio_report, col_report)
    dest = ROOT / "evals" / "visual_qa_protocols_latest.json"
    dest.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    primary = col_report or bio_report or report
    if primary is None:
        return
    snap = dict(primary)
    snap["four_column"] = combined["four_column"]
    if col_report:
        snap["colpali_t2i"] = col_report.get("clip_image_only")
        snap["colpali_encoder"] = col_report.get("encoder")
    dest_one.write_text(json.dumps(snap, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
