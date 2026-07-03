"""Chat endpoints — non-streaming and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from rag_pipeline.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
)

router = APIRouter(tags=["chat"])

_executor = ThreadPoolExecutor(max_workers=4)
_pipeline = None


def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        from rag_pipeline.main import build_ask_pipeline
        _pipeline = build_ask_pipeline()
    return _pipeline


def _format_citations(result: Any) -> list[CitationResponse]:
    return [
        CitationResponse(
            doc_id=c.doc_id or "",
            title=c.title or "Wikipedia",
            url=c.source_url or "",
            score=c.confidence,
        )
        for c in result.citations
    ]


# ─── Non-streaming ────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    pipeline = _get_pipeline()
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    result = await loop.run_in_executor(_executor, pipeline.ask, request.question)
    latency_ms = (time.perf_counter() - start) * 1000
    return ChatResponse(
        answer=result.answer,
        citations=_format_citations(result),
        confidence=result.confidence,
        latency_ms=round(latency_ms, 2),
    )


# ─── SSE Streaming ───────────────────────────────────────────────────────────

@router.get("/chat/stream")
async def chat_stream(
    question: str = Query(..., min_length=1, max_length=1000),
    skip_rewrite: bool = Query(default=True, description="Skip LLM query rewrite for faster response"),
) -> StreamingResponse:
    """Stream answer tokens via SSE.

    - skip_rewrite=true (default): ~6-8s (retrieval + LLM)
    - skip_rewrite=false: ~12-16s (full pipeline with query rewrite)
    """
    pipeline = _get_pipeline()
    queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _run():
        try:
            t0 = time.perf_counter()

            # Query processing
            if skip_rewrite:
                processed = pipeline._run_query_processing_fast(question)
            else:
                processed = pipeline._run_query_processing(question)
            t1 = time.perf_counter()
            print(f"[STREAM] Query: {(t1-t0)*1000:.0f}ms")

            # Retrieval
            retrieval = pipeline._run_retrieval(processed)
            t2 = time.perf_counter()
            print(f"[STREAM] Retrieval: {(t2-t1)*1000:.0f}ms")

            # Stream tokens
            chunk_gen, build_result = pipeline.answer_generator.generate_stream(retrieval)

            full = ""
            for chunk in chunk_gen:
                full += chunk
                asyncio.run_coroutine_threadsafe(queue.put(("token", chunk)), loop)

            t3 = time.perf_counter()
            print(f"[STREAM] LLM: {(t3-t2)*1000:.0f}ms")

            # Build result + guardrails
            result = build_result(full)
            checked = pipeline._run_output_guardrails(result, retrieval)

            print(f"[STREAM] Total: {(time.perf_counter()-t0)*1000:.0f}ms")

            done = {
                "type": "done",
                "answer": checked.answer,
                "citations": [
                    {"doc_id": c.doc_id or "", "title": c.title or "Wikipedia",
                     "url": c.source_url or "", "score": round(c.confidence, 4)}
                    for c in checked.citations
                ],
                "confidence": round(checked.confidence, 4),
            }
            asyncio.run_coroutine_threadsafe(queue.put(("done", json.dumps(done, ensure_ascii=False))), loop)

        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))),
                loop,
            )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    _executor.submit(_run)

    async def generate():
        while True:
            item = await queue.get()
            if item is None:
                break
            etype, data = item
            if etype == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': data}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
