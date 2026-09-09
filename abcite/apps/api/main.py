"""FastAPI serving: /query /feedback /health /docs."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from abcite_core.models import FeedbackRecord, FeedbackRequest, QueryRequest, QueryResponse
from abcite_core.retrieval import HybridIndex
from abcite_core.settings import Settings
from agents.graph import EvidenceGraph
from ingest.chunking import load_corpus
from providers import get_llm

FEEDBACK: list[FeedbackRecord] = []
log = logging.getLogger("abcite.api")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    key = settings.langsmith_api_key or settings.langchain_api_key
    if settings.langsmith_tracing and key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", key)
        os.environ.setdefault("LANGSMITH_API_KEY", key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        log.info("LangSmith tracing enabled for project %s", settings.langsmith_project)
    else:
        log.info("LangSmith tracing off; structured logs only")
    corpus = load_corpus(_root() / settings.corpus_dir)
    index = HybridIndex(corpus)
    llm = get_llm(settings)
    app.state.settings = settings
    app.state.graph = EvidenceGraph(
        index,
        abstain_threshold=settings.abstain_threshold,
        hitl_enabled=settings.hitl_high_stakes,
        provider_name=llm.name,
        llm=llm,
    )
    log.info("loaded %s chunks provider=%s vision=%s", len(corpus), llm.name, index.vision.name)
    yield


app = FastAPI(title="AbCite", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

figures = _root() / "corpus" / "figures"
if figures.exists():
    app.mount("/media", StaticFiles(directory=str(figures)), name="media")


@app.get("/health")
def health() -> dict:
    vision = "unloaded"
    graph = getattr(app.state, "graph", None)
    if graph is not None:
        vision = graph.index.vision.name
    return {"ok": True, "service": "abcite", "vision": vision}


@app.post("/search-image")
def search_image(
    path: str | None = None,
    query: str | None = None,
    task: str = "evidence",
    top_k: int = 5,
) -> list[dict]:
    graph: EvidenceGraph = app.state.graph
    visual = (task or "").lower() in {"visual_qa", "visual", "vqa"} or bool(query)
    if visual and query:
        hits = graph.index.search_visual(query, top_k=top_k)
        return [
            {
                "chunk_id": h.chunk.chunk_id,
                "score": h.image,
                "title": h.chunk.title,
                "task": "visual_qa",
            }
            for h in hits
        ]
    if not path:
        return []
    hits = graph.index.search_by_image(Path(path), top_k=top_k)
    return [{"chunk_id": h.chunk.chunk_id, "score": h.image, "title": h.chunk.title} for h in hits]


@app.post("/query", response_model=QueryResponse)
@app.post("/rag", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    graph: EvidenceGraph = app.state.graph
    return graph.invoke(
        req.question,
        high_stakes=req.high_stakes,
        top_k=req.top_k,
        task=req.task,
        image_path=req.image_path,
        audio_path=req.audio_path,
        video_path=req.video_path,
    )


@app.post("/feedback", response_model=FeedbackRecord)
def feedback(req: FeedbackRequest) -> FeedbackRecord:
    rec = FeedbackRecord(
        trace_id=req.trace_id,
        decision=req.decision,
        note=req.note,
        edited_answer=req.edited_answer,
    )
    FEEDBACK.append(rec)
    log.info("feedback trace=%s decision=%s", rec.trace_id, rec.decision)
    return rec


@app.get("/feedback")
def list_feedback() -> list[FeedbackRecord]:
    return FEEDBACK
