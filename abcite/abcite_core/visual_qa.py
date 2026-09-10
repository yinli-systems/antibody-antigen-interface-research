"""Visual figure QA: pixel retrieval + cite-or-abstain.

Primary multimodal product task. Query text must require looking at the
image (panel layout, stain, gel bands, colors, arrows, microscopy type).
Caption-as-query is a different protocol (appendix); BM25 already wins there.

Retrieval (CLIP / ColQwen2 MaxSim) is scored separately from VQA. The
product default answerer is extractive/heuristic. `ABCITE_VLM=1` loads
Qwen2-VL-2B-Instruct as a generate path; it is not ColQwen2 retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from abcite_core.figure_gallery import FigureGallery, FigureRec, fix_to_gallery_ids, recall_at_k
from abcite_core.holdout import query_contains_pdb, strip_pdb_ids, visual_query_ok
from abcite_core.models import AnswerStatus, Citation, QueryResponse
from abcite_core.vision_flags import DEFAULT_RERANK_K

HEURISTIC_PATH = "extractive/heuristic pixel cues; not a VLM"

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold" / "visual_qa.jsonl"
TOKEN = re.compile(r"[a-z0-9_\-]+")
STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "with",
        "and",
        "or",
        "to",
        "from",
        "for",
        "by",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "as",
        "vs",
        "versus",
        "each",
        "per",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "ten",
        "next",
        "beside",
        "above",
        "below",
        "left",
        "right",
        "top",
        "bottom",
        "into",
        "over",
        "under",
        "across",
        "against",
        "after",
        "before",
        "than",
        "its",
        "their",
        "this",
        "that",
        "these",
        "those",
        "shown",
        "show",
        "shows",
        "using",
        "used",
        "via",
        "small",
        "large",
        "high",
        "low",
        "long",
        "short",
    }
)

_LEXICON_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"gel", "lane", "lanes", "bands", "blot", "electrophoresis", "rungs", "ladder"}),
    frozenset(
        {
            "fluorescence",
            "fluorescent",
            "dapi",
            "fitc",
            "confocal",
            "foci",
            "speckles",
            "emerald",
            "indigo",
            "nuclei",
        }
    ),
    frozenset({"micrograph", "grayscale", "sem", "electron", "wrinkled", "polymer", "fibril"}),
    frozenset({"plot", "chart", "charts", "whiskers", "diamond", "curve", "curves", "traces"}),
    frozenset({"photo", "photographs", "snapshot", "underwater", "contest"}),
    frozenset({"heatmap", "matrix", "scarlet", "kinship"}),
    frozenset({"histology", "stain", "brown", "purple", "counterstain", "tissue"}),
    frozenset({"ribbon", "cartoons", "coiled", "bundle", "conformational"}),
    frozenset({"radiology", "orbit", "gadolinium", "slice", "ultrasound"}),
    frozenset({"flowchart", "checkmarks", "chevron", "chevrons", "pointer"}),
    frozenset({"cladogram", "branching", "taxa"}),
    frozenset({"spectra", "spectral", "wavenumber", "absorbance"}),
)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOP and len(t) >= 3}


def sanitize_visual_query(question: str) -> str:
    """Strip PDB ids. Visual QA never uses PDB lookup in the query path."""
    return strip_pdb_ids(question or "")


def caption_keyword_overlap(query: str, caption: str) -> dict:
    """Content-token overlap used to flag BM25-should-fail items."""
    q = content_tokens(query)
    c = content_tokens(caption)
    inter = q & c
    union = q | c
    return {
        "n_query_content": len(q),
        "n_overlap": len(inter),
        "overlap_tokens": sorted(inter),
        "jaccard": round(len(inter) / len(union), 4) if union else 0.0,
        "should_fail": len(inter) <= 1,
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_visual_qa_gold(
    gallery_ids: set[str],
    recs: list[FigureRec] | None = None,
    *,
    path: Path | None = None,
    require_gold: bool = True,
) -> list[dict]:
    """Load visual-QA items. Remap fix-* → gallery ids. Drop PDB queries."""
    rows = _read_jsonl(path or GOLD)
    alias = fix_to_gallery_ids(recs) if recs else {}
    ok: list[dict] = []
    seen_gold: set[str] = set()
    for row in rows:
        q = row["question"]
        if query_contains_pdb(q):
            raise SystemExit(f"visual QA forbids PDB in query: {row.get('id')} {q!r}")
        if not visual_query_ok(q):
            raise SystemExit(f"query is not visual-content: {row.get('id')} {q!r}")
        gold_ids: list[str] = []
        seen: set[str] = set()
        for g in row.get("gold_figure_ids") or []:
            mapped = g if g in gallery_ids else alias.get(g)
            if mapped and mapped in gallery_ids and mapped not in seen:
                gold_ids.append(mapped)
                seen.add(mapped)
        if require_gold and not gold_ids and row.get("expected_status") != "abstain":
            continue
        item = dict(row)
        item["gold_figure_ids"] = gold_ids
        item["protocol"] = "visual_qa"
        if gold_ids:
            primary = gold_ids[0]
            if primary in seen_gold and row.get("expected_status") != "abstain":
                raise SystemExit(f"duplicate gold figure {primary} in {row.get('id')}")
            if row.get("expected_status") != "abstain":
                seen_gold.add(primary)
        ok.append(item)
    return ok


def _n_runs(mask: np.ndarray) -> int:
    runs = 0
    prev = False
    for v in mask.tolist():
        flag = bool(v)
        if flag and not prev:
            runs += 1
        prev = flag
    return runs


def describe_pixels(path: Path) -> str:
    """Cheap pixel cues (colors, panels, gel-like lanes). Not a VLM."""
    from abcite_core.raster import load_rgb

    img = load_rgb(path, size=(224, 224))
    arr = np.asarray(img).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 1)
    mean_sat = float(sat.mean())
    tags: list[str] = []
    if mean_sat < 0.12:
        tags.append("grayscale")
    else:
        if float(((g > r + 15) & (g > b + 10)).mean()) > 0.06:
            tags.append("green")
            tags.append("emerald")
        if float((b > r + 15).mean()) > 0.06:
            tags.append("blue")
            tags.append("indigo")
        if float(((r > g + 20) & (r > b + 20)).mean()) > 0.05:
            tags.append("red")
            tags.append("crimson")
            tags.append("scarlet")
        if float(((r > 90) & (g > 50) & (b < 90)).mean()) > 0.08:
            tags.append("brown")
            tags.append("tan")
    gray = arr.mean(axis=2)
    col_mean = gray.mean(axis=0)
    row_mean = gray.mean(axis=1)
    n_v = _n_runs(col_mean > np.percentile(col_mean, 88))
    n_h = _n_runs(row_mean > np.percentile(row_mean, 88))
    n_panels = max(1, (n_v + 1) * (n_h + 1) if (n_v + n_h) else 1)
    if n_panels >= 2:
        tags.append(f"{min(n_panels, 12)}-panel")
        tags.append("multi-panel")
        tags.append("tiles")
    if float(np.std(col_mean[16:-16])) > 18 and mean_sat < 0.28:
        tags.append("lanes")
        tags.append("bands")
        tags.append("gel")
    if float((gray > 240).mean()) > 0.45:
        tags.append("scientific plot")
        tags.append("chart")
    if mean_sat < 0.12 and float((gray < 80).mean()) > 0.08:
        tags.append("micrograph")
    if mean_sat > 0.18 and float((gray < 120).mean()) > 0.2:
        tags.append("photo")
        tags.append("photographs")
    return " ".join(dict.fromkeys(tags))


def _lexicon_support(question: str, description: str) -> bool:
    q = content_tokens(question)
    d = content_tokens(description)
    for group in _LEXICON_GROUPS:
        if (q & group) and (d & group):
            return True
    return False


@dataclass
class VQAResult:
    status: str
    answer: str
    figure_id: str | None
    cues: list[str] = field(default_factory=list)
    pixel_desc: str = ""
    evidence: str = ""
    path: str = HEURISTIC_PATH
    query_used: str = ""


def heuristic_vqa(
    question: str,
    rec: FigureRec | None,
    path: Path | None,
    *,
    min_overlap: int = 2,
) -> VQAResult:
    """Extractive/heuristic cite-or-abstain. Uses pixels, not the caption."""
    q_used = sanitize_visual_query(question)
    if rec is None:
        return VQAResult(
            status="abstain",
            answer="No figure retrieved. Abstaining.",
            figure_id=None,
            path=HEURISTIC_PATH,
            query_used=q_used,
        )
    desc = ""
    if path is not None and path.exists():
        try:
            desc = describe_pixels(path)
        except Exception:
            desc = ""
    q = content_tokens(q_used)
    d = content_tokens(desc)
    inter = q & d
    supported = len(inter) >= min_overlap or _lexicon_support(q_used, desc)
    if supported:
        cite = rec.figure_id
        answer = (
            f"The retrieved figure {cite} matches the visual question "
            f"({desc or 'pixel cues'}). Cite {cite}."
        )
        return VQAResult(
            status="answered",
            answer=answer,
            figure_id=cite,
            cues=sorted(inter),
            pixel_desc=desc,
            evidence=desc,
            path=HEURISTIC_PATH,
            query_used=q_used,
        )
    return VQAResult(
        status="abstain",
        answer=(
            f"Retrieved figure {rec.figure_id} does not contain the requested "
            "visual evidence. Abstaining."
        ),
        figure_id=None,
        cues=sorted(inter),
        pixel_desc=desc,
        evidence=desc,
        path=HEURISTIC_PATH,
        query_used=q_used,
    )


def resolve_vqa_answerer(answerer=None) -> tuple[object, str, str | None]:
    """VLM generate when ABCITE_VLM=1 and weights load; else heuristic."""
    if answerer is not None:
        path = getattr(answerer, "path", None)
        if callable(answerer) and path is None:
            path = "injected answerer"
        if not callable(answerer) and hasattr(answerer, "answer"):
            return answerer.answer, getattr(answerer, "path", "injected answerer"), None
        return answerer, str(path or "injected answerer"), None
    try:
        from abcite_core.vlm_vqa import get_vlm, vlm_enabled
    except Exception as exc:
        return heuristic_vqa, HEURISTIC_PATH, f"VLM import failed: {exc}"
    if not vlm_enabled():
        return heuristic_vqa, HEURISTIC_PATH, None
    vlm, skip = get_vlm()
    if vlm is None:
        return heuristic_vqa, HEURISTIC_PATH, skip
    return vlm.answer, getattr(vlm, "path", "qwen2-vl generate"), skip


def answer_visual_question(
    question: str,
    rec: FigureRec | None,
    path: Path | None,
    *,
    answerer=None,
    min_overlap: int = 2,
) -> VQAResult:
    """Cite-or-abstain on one figure. PDB ids never enter the query path."""
    fn, path_label, _skip = resolve_vqa_answerer(answerer)
    q = sanitize_visual_query(question)
    try:
        pred = fn(q, rec, path, min_overlap=min_overlap)
    except TypeError:
        pred = fn(q, rec, path)
    if not getattr(pred, "path", None):
        pred.path = path_label
    if not getattr(pred, "query_used", None):
        pred.query_used = q
    return pred


def gold_answer_overlap(pred_answer: str, gold_answer: str) -> float:
    """Content-token overlap of a VQA answer vs gold (diagnostic, not R@K)."""
    if not (pred_answer or "").strip() or not (gold_answer or "").strip():
        return 0.0
    a = content_tokens(pred_answer)
    g = content_tokens(gold_answer)
    if not a or not g:
        return 0.0
    return round(len(a & g) / max(1, len(g)), 4)


def visual_text_supports_question(question: str, text: str, min_overlap: int = 2) -> bool:
    """Whether a VLM/heuristic description actually addresses the visual question."""
    if not (question or "").strip() or not (text or "").strip():
        return False
    inter = content_tokens(question) & content_tokens(text)
    return len(inter) >= min_overlap or _lexicon_support(question, text)


def _recall_block(scores_list: list[np.ndarray], items: list[dict], gallery: FigureGallery) -> dict:
    from abcite_core.metrics import ndcg_at_k

    r1 = r5 = r10 = ndcg5 = 0.0
    n = 0
    for scores, it in zip(scores_list, items, strict=True):
        gold = it.get("gold_figure_ids") or []
        if not gold:
            continue
        ranked = gallery.ranked_ids(scores)
        r1 += recall_at_k(ranked, gold, 1)
        r5 += recall_at_k(ranked, gold, 5)
        r10 += recall_at_k(ranked, gold, 10)
        ndcg5 += ndcg_at_k(ranked, gold, 5)
        n += 1
    n = max(1, n)
    return {
        "recall@1": round(r1 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "ndcg@5": round(ndcg5 / n, 4),
        "n": n,
    }


def evaluate_visual_qa(
    gallery: FigureGallery,
    items: list[dict],
    *,
    rerank_k: int = DEFAULT_RERANK_K,
    vqa_top_k: int = 1,
    answerer=None,
    score_vqa: bool = True,
) -> dict:
    """Retrieval R@K (BM25 vs vision) plus cite-or-abstain VQA.

    Retrieval and VQA are scored separately. Do not mix VQA accuracy with
    ViDoRe nDCG@5. PDB ids are stripped from every visual query.
    """
    from abcite_core.figure_gallery import _rrf, mean_text_gap, vision_rerank_scores

    answerable = [it for it in items if it.get("gold_figure_ids")]
    fail_subset = [it for it in answerable if it.get("caption_bm25_should_fail")]
    questions = [sanitize_visual_query(it["question"]) for it in answerable]
    clip_mat = gallery.clip_score_matrix(questions) if answerable else None
    bm25_mat = (
        np.stack([gallery.caption_bm25_scores(q) for q in questions], axis=0)
        if answerable
        else None
    )
    empty = {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "ndcg@5": 0.0, "n": 0}
    cols: dict[str, dict] = {}
    if answerable and bm25_mat is not None and clip_mat is not None:
        bm25_rows = [bm25_mat[i] for i in range(len(answerable))]
        clip_rows = [clip_mat[i] for i in range(len(answerable))]
        rerank_rows = [
            vision_rerank_scores(bm25_mat[i], clip_mat[i], k=rerank_k)
            for i in range(len(answerable))
        ]
        fusion_rows = [_rrf(bm25_mat[i]) + _rrf(clip_mat[i]) for i in range(len(answerable))]
        cols["caption_bm25"] = _recall_block(bm25_rows, answerable, gallery)
        cols["clip_image_only"] = _recall_block(clip_rows, answerable, gallery)
        cols["vision_rerank"] = _recall_block(rerank_rows, answerable, gallery)
        cols["fusion"] = _recall_block(fusion_rows, answerable, gallery)
        if fail_subset:
            idx = [i for i, it in enumerate(answerable) if it.get("caption_bm25_should_fail")]
            cols["caption_bm25_should_fail_subset"] = _recall_block(
                [bm25_rows[i] for i in idx], fail_subset, gallery
            )
            cols["vision_should_fail_subset"] = _recall_block(
                [clip_rows[i] for i in idx], fail_subset, gallery
            )
    else:
        cols = {
            "caption_bm25": dict(empty),
            "clip_image_only": dict(empty),
            "vision_rerank": dict(empty),
            "fusion": dict(empty),
        }

    vqa_e2e_ok = 0
    vqa_oracle_ok = 0
    vqa_n = 0
    cite_ok = 0
    abstain_ok = 0
    abstain_n = 0
    overlap_e2e: list[float] = []
    overlap_oracle: list[float] = []
    path_by_id = {r.figure_id: r.path for r in gallery.recs}
    rec_by_id = {r.figure_id: r for r in gallery.recs}
    clip_by_id: dict[str, np.ndarray] = {}
    if clip_mat is not None:
        for i, it in enumerate(answerable):
            clip_by_id[it["id"]] = clip_mat[i]
    fn, path_label, vlm_skip = (heuristic_vqa, HEURISTIC_PATH, None)
    if score_vqa:
        fn, path_label, vlm_skip = resolve_vqa_answerer(answerer)
    for it in items:
        q = sanitize_visual_query(it["question"])
        expected = it.get("expected_status") or (
            "answered" if it.get("gold_figure_ids") else "abstain"
        )
        gold = it.get("gold_figure_ids") or []
        gold_ans = str(it.get("answer") or "")
        scores = clip_by_id.get(it["id"])
        if scores is None:
            scores = gallery.clip_image_only_scores(q)
        ranked = gallery.ranked_ids(scores)
        top_ids = ranked[:vqa_top_k]
        retrieved = rec_by_id.get(top_ids[0]) if top_ids else None
        path = path_by_id.get(top_ids[0]) if top_ids else None
        if not score_vqa:
            vqa_n += 1
            continue
        try:
            pred = fn(q, retrieved, path, min_overlap=2)
        except TypeError:
            pred = fn(q, retrieved, path)
        vqa_n += 1
        if gold:
            gold_rec = rec_by_id.get(gold[0])
            try:
                oracle = fn(q, gold_rec, path_by_id.get(gold[0]), min_overlap=2)
            except TypeError:
                oracle = fn(q, gold_rec, path_by_id.get(gold[0]))
            if top_ids and top_ids[0] in gold:
                cite_ok += 1
            if expected == "answered":
                if pred.status == "answered" and pred.figure_id in gold:
                    vqa_e2e_ok += 1
                    if gold_ans:
                        overlap_e2e.append(gold_answer_overlap(pred.answer, gold_ans))
                if oracle.status == "answered" and oracle.figure_id in gold:
                    vqa_oracle_ok += 1
                    if gold_ans:
                        overlap_oracle.append(gold_answer_overlap(oracle.answer, gold_ans))
            else:
                abstain_n += 1
                if pred.status == "abstain":
                    vqa_e2e_ok += 1
                    abstain_ok += 1
                if oracle.status == "abstain":
                    vqa_oracle_ok += 1
        else:
            abstain_n += 1
            if pred.status == "abstain":
                vqa_e2e_ok += 1
                abstain_ok += 1
                vqa_oracle_ok += 1

    n_ans = max(1, len(answerable))
    vqa_block: dict = {
        "n": vqa_n,
        "end_to_end_accuracy": round(vqa_e2e_ok / max(1, vqa_n), 4) if score_vqa else None,
        "oracle_figure_accuracy": round(vqa_oracle_ok / max(1, vqa_n), 4) if score_vqa else None,
        "cite_top1": round(cite_ok / n_ans, 4) if score_vqa else None,
        "abstain_accuracy": (
            round(abstain_ok / max(1, abstain_n), 4) if abstain_n and score_vqa else None
        ),
        "n_abstain": abstain_n if score_vqa else None,
        "gold_answer_overlap_e2e": (
            round(sum(overlap_e2e) / len(overlap_e2e), 4) if overlap_e2e else None
        ),
        "gold_answer_overlap_oracle": (
            round(sum(overlap_oracle) / len(overlap_oracle), 4) if overlap_oracle else None
        ),
        "path": path_label if score_vqa else "skipped (retrieval-only column)",
        "vlm_skip": vlm_skip,
        "not_vidore": True,
        "not_maxsim": True,
        "scored": score_vqa,
    }
    report = {
        "protocol": "visual_qa",
        "n_gallery": gallery.n,
        "n_queries": len(items),
        "n_answerable": len(answerable),
        "n_caption_bm25_should_fail": len(fail_subset),
        "encoder": getattr(gallery.vision, "name", "unknown"),
        "rerank_k": rerank_k,
        "caption_bm25": cols.get("caption_bm25", empty),
        "clip_image_only": cols.get("clip_image_only", empty),
        "vision_rerank": cols.get("vision_rerank", empty),
        "fusion": cols.get("fusion", empty),
        "caption_bm25_should_fail_subset": cols.get("caption_bm25_should_fail_subset"),
        "vision_should_fail_subset": cols.get("vision_should_fail_subset"),
        "fusion_default": "off",
        "vqa": vqa_block,
        "mean_gap_image_i2i": round(gallery.mean_rank_gap_image(), 6),
        "mean_gap_clip_text_to_image": round(mean_text_gap(gallery, questions), 6),
        "note": (
            "Primary multimodal task: visual figure QA + pixel retrieval. "
            "Query is a visual question (panel layout, stain, gel bands, colors, arrows, "
            "microscopy type), not the figure caption. Gold figure is unique. "
            "caption_bm25_should_fail items have ≤1 content-token overlap with the caption. "
            "Vision column is T→I over pixels; captions never enter encode_image. "
            "Retrieval R@K is not VQA and not ViDoRe. "
            f"VQA is cite-or-abstain given the retrieved figure ({path_label})."
        ),
    }
    bm25_r1 = float(report["caption_bm25"].get("recall@1") or 0)
    vis_r1 = float(report["clip_image_only"].get("recall@1") or 0)
    bm25_ndcg = float(report["caption_bm25"].get("ndcg@5") or 0)
    vis_ndcg = float(report["clip_image_only"].get("ndcg@5") or 0)
    report["vision_beats_bm25_r@1"] = vis_r1 > bm25_r1
    report["vision_beats_bm25_ndcg@5"] = vis_ndcg > bm25_ndcg
    return report


def visual_qa_response(
    question: str,
    hits,
    *,
    provider: str,
    trace_id: str,
    latency_ms: float,
    image_threshold: float = 0.08,
    answerer=None,
) -> QueryResponse:
    """Turn visual retrieval hits into a cite-or-abstain QueryResponse."""
    q = sanitize_visual_query(question)
    scores = {h.chunk.chunk_id: h.score for h in hits}
    if not q.strip():
        return QueryResponse(
            trace_id=trace_id,
            status=AnswerStatus.ABSTAIN,
            answer=(
                "Visual QA query was empty after stripping structure ids. Abstaining."
            ),
            citations=[],
            provider=provider,
            retrieval_scores=scores,
            latency_ms=latency_ms,
        )
    if not hits or (getattr(hits[0], "image", 0.0) < image_threshold and hits[0].score <= 0):
        return QueryResponse(
            trace_id=trace_id,
            status=AnswerStatus.ABSTAIN,
            answer=(
                "Insufficient visual evidence in the retrieved figures. Abstaining. "
                "The question requires looking at panel layout, stain, bands, or colors."
            ),
            citations=[],
            provider=provider,
            retrieval_scores=scores,
            latency_ms=latency_ms,
        )
    top = hits[0]
    path = Path(top.chunk.image_path) if top.chunk.image_path else None
    if path is not None and not path.is_absolute():
        path = ROOT / path
    rec = FigureRec(
        figure_id=top.chunk.chunk_id,
        path=path or Path("."),
        caption=top.chunk.caption or "",
        pmcid=None,
        split="",
        visual_class=top.chunk.modality.value,
        source=top.chunk.source_label,
    )
    pred = answer_visual_question(q, rec, path, answerer=answerer)
    if pred.status == "abstain":
        return QueryResponse(
            trace_id=trace_id,
            status=AnswerStatus.ABSTAIN,
            answer=pred.answer,
            citations=[],
            provider=provider,
            retrieval_scores=scores,
            latency_ms=latency_ms,
        )
    quote = (pred.evidence or pred.pixel_desc or "")[:280]
    thumb = None
    if top.chunk.image_path:
        thumb = f"/media/{Path(top.chunk.image_path).name}"
    cite = Citation(
        chunk_id=top.chunk.chunk_id,
        claim=q,
        quote=quote,
        char_start=0,
        char_end=len(quote),
        modality=top.chunk.modality,
        source_url=top.chunk.source_url,
        source_label=top.chunk.source_label,
        thumbnail_url=thumb,
        pdb_id=top.chunk.pdb_id,
        residues=top.chunk.residues,
    )
    return QueryResponse(
        trace_id=trace_id,
        status=AnswerStatus.ANSWERED,
        answer=pred.answer,
        citations=[cite],
        residues=list(top.chunk.residues[:24]),
        provider=provider,
        retrieval_scores=scores,
        latency_ms=latency_ms,
    )


def maxsim(query_toks: np.ndarray, doc_toks: np.ndarray) -> float:
    """ColBERT/ColPali late interaction: sum of per-query-token max similarities."""
    if query_toks.size == 0 or doc_toks.size == 0:
        return 0.0
    sims = query_toks @ doc_toks.T
    return float(sims.max(axis=1).sum())


def maxsim_matrix(query_toks: list[np.ndarray], doc_toks: list[np.ndarray]) -> np.ndarray:
    """Late-interaction scores, shape [n_queries, n_docs]. GPU if CUDA is free."""
    n_q = len(query_toks)
    n_d = len(doc_toks)
    out = np.zeros((n_q, n_d), dtype=np.float32)
    if n_q == 0 or n_d == 0:
        return out
    try:
        import torch

        if torch.cuda.is_available() and n_d >= 16:
            print(f"maxsim GPU n_q={n_q} n_d={n_d}", flush=True)
            return _maxsim_torch(query_toks, doc_toks)
    except Exception as exc:
        print(f"maxsim GPU failed ({type(exc).__name__}: {exc}); CPU fallback", flush=True)
    print(f"maxsim CPU n_q={n_q} n_d={n_d} (slow)", flush=True)
    for i, q in enumerate(query_toks):
        if q is None or np.asarray(q).size == 0:
            continue
        qv = np.asarray(q, dtype=np.float32)
        for j, d in enumerate(doc_toks):
            if d is None or np.asarray(d).size == 0:
                continue
            dv = np.asarray(d, dtype=np.float32)
            out[i, j] = float((qv @ dv.T).max(axis=1).sum())
    return out


def _maxsim_torch(query_toks: list[np.ndarray], doc_toks: list[np.ndarray]) -> np.ndarray:
    import torch

    device = torch.device("cuda")
    dim = 128
    for t in doc_toks:
        arr = np.asarray(t)
        if arr.ndim == 2 and arr.size:
            dim = int(arr.shape[-1])
            break
    max_d = 768
    max_q = 64
    n_q = len(query_toks)
    n_d = len(doc_toks)
    q_pad = torch.zeros((n_q, max_q, dim), device=device, dtype=torch.float16)
    q_mask = torch.zeros((n_q, max_q), device=device, dtype=torch.bool)
    for i, q in enumerate(query_toks):
        arr = np.asarray(q, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            continue
        n = min(int(arr.shape[0]), max_q)
        q_pad[i, :n].copy_(torch.from_numpy(arr[:n]).to(dtype=torch.float16))
        q_mask[i, :n] = True
    out = np.zeros((n_q, n_d), dtype=np.float32)
    chunk = 128
    for start in range(0, n_d, chunk):
        sl = doc_toks[start : start + chunk]
        nd = len(sl)
        d_pad = torch.zeros((nd, max_d, dim), device=device, dtype=torch.float16)
        d_mask = torch.zeros((nd, max_d), device=device, dtype=torch.bool)
        host = np.zeros((nd, max_d, dim), dtype=np.float16)
        host_m = np.zeros((nd, max_d), dtype=np.uint8)
        for j, d in enumerate(sl):
            arr = np.asarray(d, dtype=np.float32)
            if arr.ndim != 2 or arr.size == 0:
                continue
            n = min(int(arr.shape[0]), max_d)
            host[j, :n] = arr[:n].astype(np.float16, copy=False)
            host_m[j, :n] = 1
        d_pad.copy_(torch.from_numpy(host))
        d_mask.copy_(torch.from_numpy(host_m.astype(bool)))
        # [q,t,d] x [n,l,d] -> [q,n,t,l]
        sims = torch.einsum("qtd,nld->qntl", q_pad, d_pad)
        sims = sims.masked_fill(~d_mask.unsqueeze(0).unsqueeze(2), -1.0e4)
        per_qtok = sims.max(dim=3).values
        per_qtok = per_qtok.masked_fill(~q_mask.unsqueeze(1), 0.0)
        out[:, start : start + nd] = per_qtok.sum(dim=2).float().cpu().numpy()
        del d_pad, d_mask, sims, per_qtok
        print(f"  maxsim GPU docs {min(start + chunk, n_d)}/{n_d}", flush=True)
    return out
