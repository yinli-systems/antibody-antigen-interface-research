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
    ) -> None:
        self.index = index
        self.abstain_threshold = abstain_threshold
        self.hitl_enabled = hitl_enabled
        self.llm = llm or LocalLLM()
        self.provider_name = provider_name or self.llm.name

    def invoke(self, question: str, high_stakes: bool = False, top_k: int = 6) -> QueryResponse:
        t0 = time.perf_counter()
        trace_id = str(uuid.uuid4())
        log.info("trace=%s retrieve q=%r", trace_id, question[:180])
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
