"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ─── Request ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    question: str = Field(..., min_length=1, max_length=1000, description="User question")
    use_reranker: bool = Field(default=False, description="Use Cohere re-ranker")
    use_llm: bool = Field(default=True, description="Use LLM for query rewrite")


class EvalRequest(BaseModel):
    """Request body for eval endpoint."""

    dataset: str = Field(default="documents/eval.csv", description="Eval dataset path")
    limit: int = Field(default=50, ge=1, le=500, description="Max samples to evaluate")
    use_qdrant: bool = Field(default=True, description="Use Qdrant vector store")


# ─── Response ───────────────────────────────────────────────────────────────────

class CitationResponse(BaseModel):
    """Citation in chat response."""

    claim: str = Field(..., description="Claim from the answer")
    title: str = Field(..., description="Source article title")
    source_url: str = Field(default="", description="Wikipedia URL")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    """Response body for non-streaming chat endpoint."""

    answer: str = Field(..., description="Generated answer")
    citations: list[CitationResponse] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, description="Total latency in milliseconds")


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: Literal["ok", "degraded", "error"] = "ok"
    qdrant: Literal["connected", "disconnected"] = "connected"
    langsmith: Literal["enabled", "disabled"] = "disabled"
    version: str = "0.1.0"


class EvalResponse(BaseModel):
    """Response body for eval endpoint."""

    scores: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)
    sample_count: int = 0
    passed: bool = False


# ─── SSE Stream ─────────────────────────────────────────────────────────────────

class StreamToken(BaseModel):
    """SSE token chunk."""

    type: Literal["token"] = "token"
    content: str


class StreamDone(BaseModel):
    """SSE done signal with citations."""

    type: Literal["done"] = "done"
    answer: str = ""
    citations: list[CitationResponse] = Field(default_factory=list)
    confidence: float = 0.0


class StreamError(BaseModel):
    """SSE error signal."""

    type: Literal["error"] = "error"
    message: str
