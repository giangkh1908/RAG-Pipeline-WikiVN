"""Chat endpoints — non-streaming and SSE streaming."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from rag_pipeline.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
)

router = APIRouter(tags=["chat"])

# Lazy-loaded pipeline (initialized on first request)
_pipeline = None


def _get_pipeline() -> Any:
    """Get or create the RAG pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from rag_pipeline.main import build_ask_pipeline

        _pipeline = build_ask_pipeline()
    return _pipeline


def _format_citations(result: Any) -> list[CitationResponse]:
    """Convert AnswerResult citations to API response format."""
    return [
        CitationResponse(
            claim=c.claim,
            title=c.title,
            source_url=c.source_url,
            confidence=c.confidence,
        )
        for c in result.citations
    ]


# ─── Non-streaming endpoint ─────────────────────────────────────────────────────

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Ask a question and get a full answer with citations.

    Args:
        request: ChatRequest with question and options

    Returns:
        ChatResponse with answer, citations, confidence, and latency
    """
    pipeline = _get_pipeline()

    start = time.perf_counter()
    result = pipeline.ask(request.question)
    latency_ms = (time.perf_counter() - start) * 1000

    return ChatResponse(
        answer=result.answer,
        citations=_format_citations(result),
        confidence=result.confidence,
        latency_ms=round(latency_ms, 2),
    )


# ─── SSE streaming endpoint ─────────────────────────────────────────────────────

@router.get("/api/chat/stream")
async def chat_stream(
    question: str = Query(..., min_length=1, max_length=1000, description="User question"),
    use_reranker: bool = Query(default=False, description="Use Cohere re-ranker"),
    use_llm: bool = Query(default=True, description="Use LLM for query rewrite"),
) -> StreamingResponse:
    """Stream answer tokens via SSE (Server-Sent Events).

    Flow:
    1. Query processing → ProcessedQuery
    2. Retrieval → RetrievalResult
    3. Streaming generation → yield tokens
    4. Done signal with citations

    SSE format:
        data: {"type":"token","content":"Xin"}
        data: {"type":"token","content":"chào"}
        data: {"type":"done","answer":"Xin chào","citations":[...],"confidence":0.9}

    Args:
        question: User question (query param)
        use_reranker: Use Cohere re-ranker
        use_llm: Use LLM for query rewrite

    Returns:
        StreamingResponse with text/event-stream content type
    """
    pipeline = _get_pipeline()

    async def generate():
        """Generate SSE stream of tokens."""
        try:
            # Step 1: Query processing
            processed_query = pipeline._run_query_processing(question)

            # Step 2: Retrieval
            retrieval_result = pipeline._run_retrieval(processed_query)

            # Step 3: Streaming generation
            chunk_gen, build_result = pipeline.answer_generator.generate_stream(retrieval_result)

            full_text = ""
            for chunk in chunk_gen:
                full_text += chunk
                token_data = json.dumps({"type": "token", "content": chunk}, ensure_ascii=False)
                yield f"data: {token_data}\n\n"

            # Step 4: Build final result with guardrails
            answer_result = build_result(full_text)
            checked_result = pipeline._run_output_guardrails(answer_result, retrieval_result)

            # Step 5: Send done signal with citations
            done_data = json.dumps(
                {
                    "type": "done",
                    "answer": checked_result.answer,
                    "citations": [
                        {
                            "claim": c.claim,
                            "title": c.title,
                            "source_url": c.source_url,
                            "confidence": round(c.confidence, 4),
                        }
                        for c in checked_result.citations
                    ],
                    "confidence": round(checked_result.confidence, 4),
                },
                ensure_ascii=False,
            )
            yield f"data: {done_data}\n\n"

        except Exception as e:
            error_data = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )
