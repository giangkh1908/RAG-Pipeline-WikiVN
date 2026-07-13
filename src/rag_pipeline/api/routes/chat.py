"""Chat endpoints — non-streaming and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from rag_pipeline.api.dependencies import get_rag_pipeline
from rag_pipeline.api.schemas import (
    ChatRequest,
    ChatResponse,
    SourceResponse,
)
from rag_pipeline.generation import RAGPipeline
from rag_pipeline.generation.models import AnswerResult, GenerationEvent

router = APIRouter(tags=["chat"])
_executor = ThreadPoolExecutor(max_workers=4)


def _to_source_response(source: dict[str, Any]) -> SourceResponse:
    return SourceResponse(
        citation=source.get("citation", ""),
        title=source.get("title", ""),
        content=source.get("content", ""),
        chunk_id=source.get("chunk_id", ""),
    )


# ─── Non-streaming ────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ChatResponse:
    """Answer a single question and return the full response."""
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    result: AnswerResult = await loop.run_in_executor(_executor, pipeline.answer, request.question)
    latency_ms = (time.perf_counter() - start) * 1000
    return ChatResponse(
        answer=result.answer,
        sources=[_to_source_response(s) for s in result.sources],
        intent=result.intent or "",
        latency_ms=round(latency_ms, 2),
    )


# ─── SSE Streaming ───────────────────────────────────────────────────────────


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> StreamingResponse:
    """Stream answer tokens and progress events via SSE."""
    queue: asyncio.Queue[GenerationEvent | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _run() -> None:
        try:
            for event in pipeline.answer_stream(request.question):
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    _executor.submit(_run)

    async def _generate() -> Any:
        while True:
            event = await queue.get()
            if event is None:
                break
            payload = _event_to_payload(event)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _event_to_payload(event: GenerationEvent) -> dict[str, Any]:
    if event.type == "progress":
        return {
            "type": "progress",
            "step": event.step,
            "message": event.message,
        }
    if event.type == "token":
        return {"type": "token", "content": event.data}
    if event.type == "done":
        result: AnswerResult = event.data
        return {
            "type": "done",
            "answer": result.answer,
            "sources": result.sources,
            "intent": result.intent or "",
        }
    return {"type": "error", "message": event.message or "Unknown error"}
