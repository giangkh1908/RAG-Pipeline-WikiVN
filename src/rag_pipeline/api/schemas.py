"""Pydantic schemas for the RAG API."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ─── Request ──────────────────────────────────────────────────────────────────

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class ChatRequest(BaseModel):
    """Request body for chat endpoints."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Câu hỏi của người dùng",
    )
    session_id: str | None = Field(
        default=None,
        description="Anonymous session ID (UUID-like). Server generates one if omitted.",
    )

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not _SESSION_ID_RE.match(value):
            raise ValueError("session_id must match [A-Za-z0-9_-]{8,64}")
        return value


# ─── Response ─────────────────────────────────────────────────────────────────


class SourceResponse(BaseModel):
    """A single source citation returned to the frontend."""

    citation: str = Field(..., description="Nhãn trích dẫn, ví dụ [1]")
    title: str = Field(default="", description="Tiêu đề nguồn")
    content: str = Field(default="", description="Nội dung chunk gốc")
    chunk_id: str = Field(default="", description="ID của chunk")


class ChatResponse(BaseModel):
    """Response body for the non-streaming chat endpoint."""

    answer: str = Field(..., description="Câu trả lời được sinh ra")
    sources: list[SourceResponse] = Field(default_factory=list)
    intent: str = Field(default="", description="Intent được phân loại")
    latency_ms: float = Field(default=0.0, description="Tổng latency (ms)")
    session_id: str | None = Field(default=None, description="Session id (echoed back)")
    turn_no: int | None = Field(default=None, description="Turn number inside the session")
    memory_used: bool = Field(default=False, description="Whether chat memory was applied")


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: Literal["ok", "degraded", "error"] = "ok"
    qdrant: Literal["connected", "disconnected"] = "connected"
    version: str = "0.3.0"


# ─── SSE Stream ───────────────────────────────────────────────────────────────


class StreamProgress(BaseModel):
    """Progress event emitted during pipeline execution."""

    type: Literal["progress"] = "progress"
    step: str
    message: str


class StreamToken(BaseModel):
    """Token event emitted by the LLM."""

    type: Literal["token"] = "token"
    content: str


class StreamDone(BaseModel):
    """Done event with the final answer and sources."""

    type: Literal["done"] = "done"
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    intent: str = ""
    session_id: str | None = None
    turn_no: int | None = None
    memory_used: bool = False


class StreamError(BaseModel):
    """Error event when the pipeline fails."""

    type: Literal["error"] = "error"
    message: str


StreamEvent = StreamProgress | StreamToken | StreamDone | StreamError


# ─── Suggestions ─────────────────────────────────────────────────────────────


class SuggestionRequest(BaseModel):
    """Request body for the suggestions endpoint."""

    session_id: str | None = Field(default=None)
    last_question: str = Field(..., min_length=1, max_length=500)
    last_answer: str = Field(..., min_length=1, max_length=2000)


class SuggestionResponse(BaseModel):
    """Response body for the suggestions endpoint."""

    suggestions: list[str] = Field(default_factory=list)
    fallback: bool = Field(default=False, description="True if using default suggestions")
