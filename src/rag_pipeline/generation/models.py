"""Models for answer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_pipeline.retrieval.models import RetrievalResult


@dataclass
class BuiltContext:
    """A context string assembled from retrieval results plus citation mapping."""

    context: str
    citations: dict[str, RetrievalResult] = field(default_factory=dict)


@dataclass
class GeneratedAnswer:
    """A generated answer from an LLM."""

    answer: str
    model_name: str


@dataclass
class AnswerResult:
    """Final result from the full RAG pipeline."""

    query: str
    answer: str
    context: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    intent: str | None = None
    session_id: str | None = None
    turn_no: int | None = None
    memory_used: bool = False


@dataclass
class GenerationEvent:
    """A streaming event emitted during RAG answer generation."""

    type: str  # "progress" | "token" | "done" | "error"
    step: str | None = None  # "rewrite" | "retrieval" | "context" | "generation"
    message: str | None = None
    data: Any | None = None
