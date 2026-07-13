"""Pydantic schemas for the RAG API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ─── Request ──────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request body for chat endpoints."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Câu hỏi của ngườ i dùng",
    )


# ─── Response ─────────────────────────────────────────────────────────────────


class SourceResponse(BaseModel):
    """A single source citation returned to the frontend."""

    citation: str = Field(..., description="Nhãn trích dẫn, ví dụ [1]")
    title: str = Field(default="", description="Tiêu đề nguồn")
    content: str = Field(default="", description="Nội dung chunk gốc")
    chunk_id: str = Field(default="", description="ID của chunk")


class ChatResponse(BaseModel):
    """Response body for the non-streaming chat endpoint."""

    answer: str = Field(..., description="Câu trả lờ i được sinh ra")
    sources: list[SourceResponse] = Field(default_factory=list)
    intent: str = Field(default="", description="Intent được phân loại")
    latency_ms: float = Field(default=0.0, description="Tổng latency (ms)")


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: Literal["ok", "degraded", "error"] = "ok"
    qdrant: Literal["connected", "disconnected"] = "connected"
    version: str = "0.2.0"


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


class StreamError(BaseModel):
    """Error event when the pipeline fails."""

    type: Literal["error"] = "error"
    message: str


StreamEvent = StreamProgress | StreamToken | StreamDone | StreamError
