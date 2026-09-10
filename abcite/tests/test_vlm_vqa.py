"""Cite-or-abstain VLM VQA: parse, PDB-free queries, heuristic default, eval split."""

from pathlib import Path

from abcite_core.figure_gallery import FigureGallery, FigureRec
from abcite_core.models import Chunk, Modality
from abcite_core.retrieval import HybridIndex
from abcite_core.vision import LocalImageEncoder
from abcite_core.visual_qa import (
    HEURISTIC_PATH,
    VQAResult,
    evaluate_visual_qa,
    heuristic_vqa,
    resolve_vqa_answerer,
    sanitize_visual_query,
)
from abcite_core.vlm_vqa import (
    build_vlm_prompt,
    generate_vlm_ok,
    is_colqwen_retrieval_id,
    parse_vlm_generation,
    reset_vlm_loader,
    try_load_vlm,
)
from agents.graph import EvidenceGraph

ROOT = Path(__file__).resolve().parents[1]


def test_sanitize_strips_1n8z_from_query():
    q = sanitize_visual_query(
        "Does 1N8Z show emerald nuclear speckles against a dark field?"
    )
    assert "1n8z" not in q.lower()
    assert "1N8Z" not in q
    assert "emerald" in q.lower()


def test_prompt_question_line_has_no_1n8z():
    prompt, q = build_vlm_prompt(
        "1N8Z emerald nuclear speckles against a dark field",
        "pmc-00313",
    )
    assert "1n8z" not in q.lower()
    qline = next(ln for ln in prompt.splitlines() if ln.startswith("Question:"))
    assert "1n8z" not in qline.lower()
    assert "1N8Z" not in qline
    assert "pmc-00313" in prompt


def test_parse_vlm_abstain():
    raw = (
        "STATUS: abstain\n"
        "CITE: none\n"
        "ANSWER:\n"
        "EVIDENCE: grayscale gel; no magenta star or polar plot"
    )
    pred = parse_vlm_generation(raw, "pmc-00001", query_used="rainbow polar plot")
    assert pred.status == "abstain"
    assert pred.figure_id is None
    assert "Abstaining" in pred.answer


def test_parse_vlm_answered_cites_shown_figure_not_invented_id():
    raw = (
        "STATUS: answered\n"
        "CITE: pmc-99999\n"
        "ANSWER: Four grayscale micrograph tiles of a wrinkled surface.\n"
        "EVIDENCE: 2x2 SEM-like tiles"
    )
    pred = parse_vlm_generation(raw, "pmc-00009")
    assert pred.status == "answered"
    assert pred.figure_id == "pmc-00009"
    assert "pmc-00009" in pred.answer


def test_parse_empty_and_refusal_abstain():
    empty = parse_vlm_generation("", "pmc-00001")
    assert empty.status == "abstain"
    refuse = parse_vlm_generation(
        "I cannot answer; the requested polar plot is not visible.",
        "pmc-00001",
    )
    assert refuse.status == "abstain"
    assert refuse.figure_id is None


def test_freeform_generation_cites_shown_figure_when_cues_match():
    raw = (
        "The image is a gel electrophoresis plate with seven vertical sample "
        "lanes of dark bands beside a rainbow-stained marker column."
    )
    pred = parse_vlm_generation(
        raw,
        "pmc-00323",
        query_used=(
            "seven vertical sample lanes of dark bands beside a rainbow-stained marker column"
        ),
    )
    assert pred.status == "answered"
    assert pred.figure_id == "pmc-00323"


def test_freeform_unrelated_caption_abstains():
    raw = "The figure shows a scatter plot with a linear trend line."
    pred = parse_vlm_generation(
        raw,
        "pmc-00313",
        query_used="emerald nuclear speckles against a dark field",
    )
    assert pred.status == "abstain"
    assert pred.figure_id is None


def test_colqwen2_architecture_is_not_generate_vlm():
    ok, reason = generate_vlm_ok({"architectures": ["ColQwen2"]})
    assert ok is False
    assert "custom_text_proj" in reason
    assert is_colqwen_retrieval_id("vidore/colqwen2-base")
    ok2, _ = generate_vlm_ok({"architectures": ["Qwen2VLForConditionalGeneration"]})
    assert ok2 is True


def test_try_load_rejects_colqwen_model_id(monkeypatch):
    monkeypatch.setenv("ABCITE_VLM", "1")
    monkeypatch.setenv("ABCITE_VLM_MODEL", "vidore/colqwen2-base")
    reset_vlm_loader()
    ans, skip = try_load_vlm()
    assert ans is None
    assert skip is not None
    assert "custom_text_proj" in skip or "MaxSim" in skip
    reset_vlm_loader()


def test_heuristic_path_when_vlm_off(monkeypatch):
    monkeypatch.setenv("ABCITE_VLM", "0")
    reset_vlm_loader()
    fn, path, skip = resolve_vqa_answerer()
    assert fn is heuristic_vqa
    assert path == HEURISTIC_PATH
    assert skip is None
    rec = FigureRec(
        figure_id="fig-x",
        path=ROOT / "corpus" / "figures" / "fig-1n8z.svg",
        caption="unused",
        pmcid=None,
        split="",
        visual_class="ribbon",
        source="test",
    )
    pred = heuristic_vqa(
        "rainbow circular dichroism polar plot with a magenta star on the origin",
        rec,
        rec.path,
    )
    assert pred.status == "abstain"
    assert pred.figure_id is None
    assert pred.path == HEURISTIC_PATH


def test_vlm_flag_falls_back_to_heuristic_when_loader_empty(monkeypatch):
    monkeypatch.setenv("ABCITE_VLM", "1")
    reset_vlm_loader()
    import abcite_core.vlm_vqa as mod

    monkeypatch.setattr(mod, "try_load_vlm", lambda: (None, "missing weights"))
    monkeypatch.setattr(mod, "_VLM_TRIED", False)
    fn, path, skip = resolve_vqa_answerer()
    assert fn is heuristic_vqa
    assert "heuristic" in path
    assert skip
    reset_vlm_loader()


def _svg_gallery() -> FigureGallery:
    figs = ROOT / "corpus" / "figures"
    recs = [
        FigureRec(
            "fig-alpha",
            figs / "fig-1n8z.svg",
            "alpha-unique-caption membrane-proximal",
            None,
            "",
            "ribbon",
            "test",
        ),
        FigureRec(
            "fig-beta",
            figs / "fig-6w41.svg",
            "beta-unique-caption cryptic face",
            None,
            "",
            "ribbon",
            "test",
        ),
    ]
    recs = [r for r in recs if r.path.exists()]
    assert len(recs) >= 2
    return FigureGallery(recs, LocalImageEncoder())


def test_injected_vlm_abstains_on_unanswerable_and_splits_metrics():
    gal = _svg_gallery()

    def mock_vlm(question, rec, path, **_kw):
        if "magenta" in question or "dichroism" in question:
            return VQAResult(
                status="abstain",
                answer="Pixels do not show a polar plot. Abstaining.",
                figure_id=None,
                path="mock-vlm",
                query_used=question,
            )
        if rec is None:
            return VQAResult(
                status="abstain",
                answer="No figure. Abstaining.",
                figure_id=None,
                path="mock-vlm",
                query_used=question,
            )
        return VQAResult(
            status="answered",
            answer=f"Ribbon cartoons. Cite {rec.figure_id}.",
            figure_id=rec.figure_id,
            path="mock-vlm",
            query_used=question,
        )

    mock_vlm.path = "mock-vlm"

    items = [
        {
            "id": "ans",
            "question": "ribbon cartoon of two coiled chains",
            "gold_figure_ids": ["fig-alpha"],
            "answer": "two coiled ribbons",
            "expected_status": "answered",
            "caption_bm25_should_fail": True,
        },
        {
            "id": "abs",
            "question": "rainbow circular dichroism polar plot with a magenta star on the origin",
            "gold_figure_ids": [],
            "answer": "",
            "expected_status": "abstain",
            "caption_bm25_should_fail": True,
        },
    ]
    report = evaluate_visual_qa(gal, items, rerank_k=5, answerer=mock_vlm)
    assert report["protocol"] == "visual_qa"
    assert "recall@1" in report["clip_image_only"]
    vqa = report["vqa"]
    assert "recall@1" not in vqa
    assert vqa["n"] == 2
    assert vqa["abstain_accuracy"] == 1.0
    assert vqa["n_abstain"] == 1
    assert vqa["path"] == "mock-vlm"
    assert vqa["not_vidore"] is True
    assert vqa["not_maxsim"] is True
    assert 0.0 <= vqa["end_to_end_accuracy"] <= 1.0
    assert 0.0 <= vqa["oracle_figure_accuracy"] <= 1.0


def test_eval_default_stays_heuristic_without_vlm_flag(monkeypatch):
    monkeypatch.setenv("ABCITE_VLM", "0")
    reset_vlm_loader()
    gal = _svg_gallery()
    items = [
        {
            "id": "abs",
            "question": "rainbow circular dichroism polar plot with a magenta star on the origin",
            "gold_figure_ids": [],
            "answer": "",
            "expected_status": "abstain",
            "caption_bm25_should_fail": True,
        },
    ]
    report = evaluate_visual_qa(gal, items, rerank_k=5)
    assert "heuristic" in report["vqa"]["path"]
    assert report["vqa"]["abstain_accuracy"] == 1.0
    assert "recall@1" not in report["vqa"]


def test_visual_qa_graph_strips_1n8z_from_query():
    chunks = [
        Chunk(
            chunk_id="fig-alpha",
            parent_id="txt-a",
            modality=Modality.FIGURE,
            title="Alpha epitope cartoon",
            text="Figure caption: alpha-unique-caption membrane-proximal.",
            caption="alpha-unique-caption membrane-proximal domain IV",
            source_url="https://example.test/a",
            source_label="a",
            image_path="corpus/figures/fig-1n8z.svg",
        ),
    ]
    idx = HybridIndex(chunks, vision=LocalImageEncoder())
    g = EvidenceGraph(idx, 0.12, False, "local")
    resp = g.invoke(
        "1N8Z ribbon cartoon of coiled chains on a dark field",
        task="visual_qa",
    )
    assert resp.status.value in {"answered", "abstain"}
    blob = (resp.answer or "") + " " + " ".join(c.claim for c in resp.citations)
    # Query path must not keep 1N8Z; citations may still name fig-1n8z as an id.
    assert "Does 1N8Z" not in blob
    for c in resp.citations:
        assert "1N8Z" not in (c.claim or "")
