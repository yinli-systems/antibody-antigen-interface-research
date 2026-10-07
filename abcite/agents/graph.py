"""LangGraph-compatible antibody evidence graph: retrieve → tools → cite → HITL."""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from abcite_core.models import AnswerStatus, Chunk, QueryResponse
from abcite_core.overlap import lookup_predicted_overlap
from abcite_core.rcsb import ALLOWLIST, RcsbError, fetch_entry
from abcite_core.retrieval import HybridIndex
from abcite_core.tracing import emit_run
from providers.base import LLMProvider, citations_from_chunks, format_evidence
from providers.local import LocalLLM

log = logging.getLogger("abcite.graph")

HIGH_STAKES = re.compile(
    r"clinical|patient|prescribe|therapeutic decision|change annotation|wet[- ]lab",
    re.I,
)

UNANSWERABLE_HINTS = re.compile(
    r"unpublished|secret accession|invented pdb|pdb 9zzz|made-up pmid|"
    r"pmid 00000000|ignore retrieved|nhs number|alicesecret|1abcde|"
    r"pdb 1foo|not in the index|malaria antigen|peeking at frozen|"
    r"official-test sota|patient nhs|no public structure|synthetic_exam|"
    r"toy pdb 0xxx|fabricated complex|not stored locally|closed index",
    re.I,
)

PDB_TOKEN = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")


@dataclass
class GraphState:
    question: str
    high_stakes: bool = False
    top_k: int = 6
    hits: list[Any] = field(default_factory=list)
    tool_notes: list[str] = field(default_factory=list)
    status: AnswerStatus = AnswerStatus.ABSTAIN
    answer: str = ""
    chunks: list[Chunk] = field(default_factory=list)
    interrupt: bool = False
    hitl_reason: str | None = None
    trace_id: str = ""


def unknown_pdb_mentioned(question: str, known_pdb_ids: set[str]) -> bool:
    """True if the question names a 4-char PDB that is not in the local corpus.

    PDB codes start with a digit and contain at least one letter (skips years/PMIDs).
    """
    known = {p.upper() for p in known_pdb_ids if p}
    for token in PDB_TOKEN.findall(question):
        if not re.search(r"[A-Za-z]", token):
            continue
        pid = token.upper()
        if pid not in known and pid not in ALLOWLIST:
            return True
    return False


def should_abstain(
    question: str,
    hits: list[Any],
    threshold: float,
    known_pdb_ids: set[str] | None = None,
) -> bool:
    if UNANSWERABLE_HINTS.search(question):
        return True
    if known_pdb_ids is not None and unknown_pdb_mentioned(question, known_pdb_ids):
        return True
    if not hits:
        return True
    best = hits[0]
    if best.bm25 < threshold and best.dense < 0.08:
        return True
    return False


def needs_hitl(question: str, high_stakes_flag: bool, enabled: bool) -> bool:
    if not enabled:
        return False
    return high_stakes_flag or bool(HIGH_STAKES.search(question))


def compose_extractive(question: str, chunks: list[Chunk], tool_notes: list[str]) -> str:
    parts = [c.text.strip() for c in chunks[:4]]
    body = " ".join(parts)
    extra = (" " + " ".join(tool_notes)) if tool_notes else ""
    return (
        f"Grounded summary for: {question}\n\n{body[:1400]}{extra}\n\n"
        "Every claim above is supported by the cited chunks. "
        "This is not a therapeutic recommendation."
    )


def compose_with_provider(
    llm: LLMProvider,
    question: str,
    chunks: list[Chunk],
    tool_notes: list[str],
) -> str:
    extractive = compose_extractive(question, chunks, tool_notes)
    if llm.name == "local":
        return extractive
    prompt = (
        "Answer the antibody epitope question using ONLY the evidence chunks. "
        "Do not invent PMIDs or PDB codes. If evidence is insufficient, reply ABSTAIN.\n\n"
        f"Question: {question}\n\nEvidence:\n{format_evidence(chunks)}\n"
    )
    if tool_notes:
        prompt += "\nTools:\n" + "\n".join(tool_notes)
    drafted = llm.complete(prompt).strip()
    blob = " ".join(c.text for c in chunks).lower()
    words = [w for w in re.findall(r"[a-z0-9]{4,}", drafted.lower()) if w not in {"abstain", "chunk"}]
    overlap = sum(1 for w in words if w in blob)
    if drafted.upper().startswith("ABSTAIN") or (words and overlap / max(1, len(words)) < 0.08):
        return extractive
    return drafted + "\n\n[Citations attached from retrieved chunks; uncited generation is discarded.]"


def run_tools(question: str) -> list[str]:
    notes: list[str] = []
    if re.search(r"framecontact|predicted overlap|hotspot excess", question, re.I):
        ov = lookup_predicted_overlap()
        notes.append(
            f"FrameContact public diagnostic (not official test): "
            f"true_epitope_jaccard={ov['true_epitope_jaccard_mean']:.4f}; "
            f"a0 predicted_topk_jaccard_mean={ov['a0_predicted_topk_jaccard_mean']:.3f}. "
            f"Boundary: {ov['claim_boundary']}"
        )
    pdb = re.search(r"\b([0-9][A-Za-z0-9]{3})\b", question)
    if pdb:
        pid = pdb.group(1).upper()
        try:
            meta = fetch_entry(pid)
            notes.append(
                f"RCSB {meta['pdb_id']}: {meta.get('title') or 'cached entry'} "
                f"({meta.get('method') or 'method n/a'}; cache_hit={meta.get('cache_hit', True)}). "
                f"{meta.get('rcsb')}"
            )
        except RcsbError:
            notes.append(f"PDB {pid} is not in the AbCite RCSB allowlist/cache.")
    return notes


class EvidenceGraph:
    def __init__(
        self,
        index: HybridIndex,
        abstain_threshold: float,
        hitl_enabled: bool,
        provider_name: str,
        llm: LLMProvider | None = None,
        audio_gallery=None,
        video_gallery=None,
        figure_gallery=None,
    ) -> None:
        self.index = index
        self.abstain_threshold = abstain_threshold
        self.hitl_enabled = hitl_enabled
        self.llm = llm or LocalLLM()
        self.provider_name = provider_name or self.llm.name
        self.audio_gallery = audio_gallery
        self.video_gallery = video_gallery
        self.figure_gallery = figure_gallery
        self._pmc_gallery_loaded = figure_gallery is not None

    def _ensure_pmc_gallery(self):
        if self.figure_gallery is not None:
            return self.figure_gallery
        if self._pmc_gallery_loaded:
            return None
        self._pmc_gallery_loaded = True
        from abcite_core.figure_gallery import load_product_pmc_gallery

        try:
            self.figure_gallery = load_product_pmc_gallery(self.index.vision)
        except Exception as exc:
            log.warning("PMC gallery skipped: %s", exc)
            self.figure_gallery = None
        return self.figure_gallery

    def search_by_image(self, path, top_k: int = 5, fuse_captions: bool = True):
        """Pixels→pixels on the same gallery as visual_qa (fixture or PMC npz)."""
        gal = self._ensure_pmc_gallery()
        if gal is not None:
            return gal.search_by_image(path, top_k=top_k, fuse_captions=fuse_captions)
        return self.index.search_by_image(path, top_k=top_k, fuse_captions=fuse_captions)

    def search_visual(self, query: str, top_k: int = 6):
        """visual_qa retrieval: PMC npz MaxSim when enabled, else fixture HybridIndex."""
        gal = self._ensure_pmc_gallery()
        if gal is not None:
            return gal.search_visual(query, top_k=top_k)
        return self.index.search_visual(query, top_k=top_k)

    def visual_gallery_kind(self) -> str:
        gal = self.figure_gallery
        if gal is not None:
            npz = getattr(gal, "npz_path", None)
            return f"pmc-npz n={gal.n}" + (f" {npz.name}" if npz else "")
        from abcite_core.figure_gallery import resolve_pmc_gallery_npz

        npz = resolve_pmc_gallery_npz(self.index.vision)
        if npz is not None:
            return f"pmc-npz pending {npz.name}"
        return "fixture"

    def invoke(
        self,
        question: str,
        high_stakes: bool = False,
        top_k: int = 6,
        task: str = "evidence",
        image_path: str | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> QueryResponse:
        t0 = time.perf_counter()
        trace_id = str(uuid.uuid4())
        log.info("trace=%s retrieve task=%s q=%r", trace_id, task, question[:180])
        task_l = (task or "evidence").lower()
        if task_l in {"visual_qa", "visual", "vqa"}:
            from abcite_core.visual_qa import sanitize_visual_query, visual_qa_response

            q_vis = sanitize_visual_query(question)
            hits = self.search_visual(q_vis, top_k=top_k) if q_vis else []
            resp = visual_qa_response(
                q_vis,
                hits,
                provider=self.provider_name,
                trace_id=trace_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            emit_run(
                "abcite.visual_qa",
                trace_id,
                {"question": question, "task": "visual_qa"},
                {"status": resp.status.value},
            )
            return resp
        if task_l in {"audio_qa", "audio"}:
            from abcite_core.audio_rag import audio_qa_response, load_audio_records, AudioGallery

            gal = self.audio_gallery or AudioGallery(load_audio_records(prefer_cache=True))
            self.audio_gallery = gal
            resp = audio_qa_response(
                question,
                gal,
                provider=self.provider_name,
                trace_id=trace_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
                top_k=top_k,
            )
            emit_run("abcite.audio_qa", trace_id, {"question": question, "task": "audio_qa"}, {"status": resp.status.value})
            return resp
        if task_l in {"video_qa", "video"}:
            from abcite_core.video_rag import video_qa_response, load_video_records, VideoGallery

            gal = self.video_gallery or VideoGallery(load_video_records(prefer_cache=True))
            self.video_gallery = gal
            resp = video_qa_response(
                question,
                gal,
                provider=self.provider_name,
                trace_id=trace_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
                top_k=top_k,
            )
            emit_run("abcite.video_qa", trace_id, {"question": question, "task": "video_qa"}, {"status": resp.status.value})
            return resp
        if task_l in {"unified", "rag", "multimodal"}:
            return self._invoke_unified(question, high_stakes, top_k, trace_id, t0)
        hits = self.index.search(question, top_k=top_k)
        tools = run_tools(question)
        chunks = [h.chunk for h in hits]
        scores = {h.chunk.chunk_id: h.score for h in hits}
        known_pdbs = {(c.pdb_id or "").upper() for c in self.index.chunks if c.pdb_id}

        if should_abstain(question, hits, self.abstain_threshold, known_pdbs):
            resp = QueryResponse(
                trace_id=trace_id,
                status=AnswerStatus.ABSTAIN,
                answer=(
                    "Insufficient evidence in the indexed literature, figures, and structures "
                    "to support an epitope-overlap claim. Abstaining."
                ),
                citations=[],
                provider=self.provider_name,
                retrieval_scores=scores,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            emit_run("abcite.query", trace_id, {"question": question}, {"status": resp.status.value})
            log.info("trace=%s abstain", trace_id)
            return resp

        citations = citations_from_chunks(chunks[:5], claim=question)
        residues: list[str] = []
        for c in chunks:
            residues.extend(c.residues)
        seen: set[str] = set()
        uniq = []
        for r in residues:
            if r not in seen:
                seen.add(r)
                uniq.append(r)

        answer = compose_with_provider(self.llm, question, chunks, tools)
        if needs_hitl(question, high_stakes, self.hitl_enabled):
            status = AnswerStatus.HITL_PENDING
            reason = "High-stakes claim requires human approval before release."
            log.info("trace=%s interrupt HITL", trace_id)
        else:
            status = AnswerStatus.ANSWERED
            reason = None
            log.info("trace=%s answered n_cite=%s provider=%s", trace_id, len(citations), self.llm.name)

        resp = QueryResponse(
            trace_id=trace_id,
            status=status,
            answer=answer,
            citations=citations,
            residues=uniq[:24],
            hitl_reason=reason,
            provider=self.provider_name,
            retrieval_scores=scores,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        emit_run(
            "abcite.query",
            trace_id,
            {"question": question},
            {"status": resp.status.value, "n_cite": len(citations)},
        )
        return resp

    def _invoke_unified(
        self,
        question: str,
        high_stakes: bool,
        top_k: int,
        trace_id: str,
        t0: float,
    ) -> QueryResponse:
        """Per-modality retrieve + cite. No cross-modal RRF into caption BM25."""
        from abcite_core.models import Citation
        from providers.base import citations_from_chunks

        text_hits = self.index.search(question, top_k=top_k)
        from abcite_core.visual_qa import sanitize_visual_query

        q_vis = sanitize_visual_query(question)
        visual_hits = self.search_visual(q_vis, top_k=min(3, top_k)) if q_vis else []
        citations: list[Citation] = []
        scores: dict[str, float] = {}
        notes: list[str] = []
        if text_hits and not should_abstain(
            question,
            text_hits,
            self.abstain_threshold,
            {(c.pdb_id or "").upper() for c in self.index.chunks if c.pdb_id},
        ):
            citations.extend(citations_from_chunks([h.chunk for h in text_hits[:3]], claim=question))
            scores.update({h.chunk.chunk_id: h.score for h in text_hits[:3]})
            notes.append("text")
        if visual_hits and visual_hits[0].image > 0.05:
            from abcite_core.visual_qa import visual_qa_response

            vresp = visual_qa_response(
                q_vis,
                visual_hits,
                provider=self.provider_name,
                trace_id=trace_id,
                latency_ms=0,
            )
            if vresp.status == AnswerStatus.ANSWERED:
                citations.extend(vresp.citations)
                scores.update(vresp.retrieval_scores)
                notes.append("figure")
        try:
            from abcite_core.audio_rag import AudioGallery, audio_qa_response, load_audio_records

            gal = self.audio_gallery or AudioGallery(load_audio_records(prefer_cache=True))
            self.audio_gallery = gal
            aresp = audio_qa_response(
                question, gal, provider=self.provider_name, trace_id=trace_id, latency_ms=0, top_k=3
            )
            if aresp.status == AnswerStatus.ANSWERED:
                citations.extend(aresp.citations)
                scores.update(aresp.retrieval_scores)
                notes.append("audio")
        except Exception:
            pass
        try:
            from abcite_core.video_rag import VideoGallery, load_video_records, video_qa_response

            vgal = self.video_gallery or VideoGallery(load_video_records(prefer_cache=True))
            self.video_gallery = vgal
            vresp = video_qa_response(
                question, vgal, provider=self.provider_name, trace_id=trace_id, latency_ms=0, top_k=3
            )
            if vresp.status == AnswerStatus.ANSWERED:
                citations.extend(vresp.citations)
                scores.update(vresp.retrieval_scores)
                notes.append("video")
        except Exception:
            pass
        seen: set[str] = set()
        uniq: list[Citation] = []
        for c in citations:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            uniq.append(c)
        latency = (time.perf_counter() - t0) * 1000
        if not uniq:
            resp = QueryResponse(
                trace_id=trace_id,
                status=AnswerStatus.ABSTAIN,
                answer=(
                    "Insufficient evidence across text, figures, audio, and video. Abstaining."
                ),
                citations=[],
                provider=self.provider_name,
                retrieval_scores=scores,
                latency_ms=latency,
            )
            emit_run("abcite.unified", trace_id, {"question": question}, {"status": "abstain"})
            return resp
        answer = (
            f"Unified cite-or-abstain over {', '.join(notes) or 'no'} modalit"
            f"{'y' if len(notes) == 1 else 'ies'}. "
            "Each citation is from its own retriever; vision was not RRF'd into caption BM25."
        )
        if needs_hitl(question, high_stakes, self.hitl_enabled):
            status = AnswerStatus.HITL_PENDING
            reason = "High-stakes claim requires human approval before release."
        else:
            status = AnswerStatus.ANSWERED
            reason = None
        resp = QueryResponse(
            trace_id=trace_id,
            status=status,
            answer=answer,
            citations=uniq[:8],
            hitl_reason=reason,
            provider=self.provider_name,
            retrieval_scores=scores,
            latency_ms=latency,
        )
        emit_run(
            "abcite.unified",
            trace_id,
            {"question": question, "modalities": notes},
            {"status": resp.status.value, "n_cite": len(uniq)},
        )
        return resp


def build_langgraph(graph: EvidenceGraph) -> Callable[..., QueryResponse] | None:
    """Optional LangGraph wrapper. Returns None if langgraph is not installed."""
    try:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
    except Exception:
        return None

    class LGState(TypedDict, total=False):
        question: str
        high_stakes: bool
        response: QueryResponse

    def node_run(state: LGState) -> LGState:
        resp = graph.invoke(state["question"], bool(state.get("high_stakes")), 6)
        if resp.status == AnswerStatus.HITL_PENDING:
            decision = interrupt({"trace_id": resp.trace_id, "draft": resp.answer})
            if decision == "reject":
                resp.status = AnswerStatus.ABSTAIN
                resp.answer = "Human reviewer rejected the high-stakes draft."
            else:
                resp.status = AnswerStatus.ANSWERED
        state["response"] = resp
        return state

    sg = StateGraph(LGState)
    sg.add_node("run", node_run)
    sg.add_edge(START, "run")
    sg.add_edge("run", END)
    compiled = sg.compile()

    def _invoke(question: str, high_stakes: bool = False, top_k: int = 6) -> QueryResponse:
        out = compiled.invoke({"question": question, "high_stakes": high_stakes})
        return out["response"]

    return _invoke
