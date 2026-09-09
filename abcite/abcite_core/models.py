"""Shared types for AbCite evidence answers."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    FIGURE = "figure"
    STRUCTURE = "structure"


class Chunk(BaseModel):
    chunk_id: str
    parent_id: str
    modality: Modality
    title: str
    text: str
    source_url: str
    source_label: str
    pdb_id: str | None = None
    residues: list[str] = Field(default_factory=list)
    image_path: str | None = None
    caption: str | None = None
    char_start: int = 0
    char_end: int = 0
    antigen: str | None = None
    antibody: str | None = None
    tags: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    chunk_id: str
    claim: str
    quote: str
    char_start: int
    char_end: int
    modality: Modality
    source_url: str
    source_label: str
    thumbnail_url: str | None = None
    pdb_id: str | None = None
    residues: list[str] = Field(default_factory=list)


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    ABSTAIN = "abstain"
    HITL_PENDING = "hitl_pending"


class QueryRequest(BaseModel):
    question: str
    high_stakes: bool = False
    top_k: int = 6
    task: str = "evidence"


class QueryResponse(BaseModel):
    trace_id: str
    status: AnswerStatus
    answer: str
    citations: list[Citation]
    residues: list[str] = Field(default_factory=list)
    hitl_reason: str | None = None
    provider: str
    retrieval_scores: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class FeedbackRequest(BaseModel):
    trace_id: str
    decision: Literal["approve", "reject", "edit"]
    note: str | None = None
    edited_answer: str | None = None


class FeedbackRecord(BaseModel):
    trace_id: str
    decision: str
    note: str | None = None
    edited_answer: str | None = None
