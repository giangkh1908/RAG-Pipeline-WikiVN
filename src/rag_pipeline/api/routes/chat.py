"""Chat endpoints — non-streaming and SSE streaming."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from rag_pipeline.api.dependencies import get_conversation_store, get_rag_pipeline
from rag_pipeline.api.schemas import (
    ChatRequest,
    ChatResponse,
    SourceResponse,
    SuggestionRequest,
    SuggestionResponse,
)
from rag_pipeline.generation import RAGPipeline
from rag_pipeline.generation.models import AnswerResult, GenerationEvent
from rag_pipeline.storage.conversation import ConversationStore

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
    store: ConversationStore | None = Depends(get_conversation_store),
) -> ChatResponse:
    """Answer a single question and return the full response."""
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    sid = request.session_id
    if sid is None:
        sid = uuid.uuid4().hex
        if store is not None:
            await loop.run_in_executor(_executor, store.upsert_session, sid)
    elif store is not None:
        await loop.run_in_executor(_executor, store.upsert_session, sid)
    result: AnswerResult = await loop.run_in_executor(
        _executor, pipeline.answer, request.question, sid
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return ChatResponse(
        answer=result.answer,
        sources=[_to_source_response(s) for s in result.sources],
        intent=result.intent or "",
        latency_ms=round(latency_ms, 2),
        session_id=sid,
        turn_no=result.turn_no,
        memory_used=result.memory_used,
    )


# ─── SSE Streaming ───────────────────────────────────────────────────────────


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    store: ConversationStore | None = Depends(get_conversation_store),
) -> StreamingResponse:
    """Stream answer tokens and progress events via SSE."""
    loop = asyncio.get_event_loop()

    async def _resolve_and_run() -> tuple[str, Iterator[GenerationEvent]]:
        sid = request.session_id
        if sid is None:
            sid = uuid.uuid4().hex
        if store is not None:
            try:
                await loop.run_in_executor(_executor, store.upsert_session, sid)
            except Exception as exc:
                print(f"ERROR in upsert_session: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        return sid, pipeline.answer_stream(request.question, session_id=sid)

    session_id, event_iter = await _resolve_and_run()
    queue: asyncio.Queue[GenerationEvent | None] = asyncio.Queue()

    def _run() -> None:
        try:
            for event in event_iter:
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        except Exception as exc:
            print(f"ERROR in answer_stream: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            asyncio.run_coroutine_threadsafe(
                queue.put(GenerationEvent(type="error", message=str(exc))), loop
            )
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
            "session_id": result.session_id,
            "turn_no": result.turn_no,
            "memory_used": result.memory_used,
        }
    return {"type": "error", "message": event.message or "Unknown error"}


# ─── Session management ──────────────────────────────────────────────────────


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    store: ConversationStore | None = Depends(get_conversation_store),
) -> dict[str, Any]:
    """Hard-delete a session and all its turns."""
    if store is None:
        return {"deleted": False, "reason": "memory_disabled"}
    # Cascade through FK relation manually (we don't enforce FK to keep
    # tests simple — see Phase 1 note about WAL + FK).
    if not store.session_exists(session_id):
        return {"deleted": False, "reason": "not_found"}
    deleted_turns = store.delete_session(session_id)
    return {"deleted": True, "deleted_turns": deleted_turns}


@router.post("/session/gc")
async def gc_sessions(
    store: ConversationStore | None = Depends(get_conversation_store),
) -> dict[str, Any]:
    """Delete sessions inactive for longer than ``MemoryConfig.session_ttl_hours``."""
    if store is None:
        return {"deleted": 0, "reason": "memory_disabled"}
    deleted = store.gc_sessions_ttl()
    return {"deleted": deleted}


# ─── Suggestions ─────────────────────────────────────────────────────────────

_DEFAULT_SUGGESTIONS = [
    "Vịnh Hạ Long nằm ở đâu?",
    "Du lịch Hội An nên đi mùa nào?",
    "Có món ăn đặc sản nào ở Đà Nẵng?",
    "Nha Trang có bãi biển nổi tiếng nào?",
]


@router.post("/suggestions", response_model=SuggestionResponse)
async def suggestions(
    request: SuggestionRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> SuggestionResponse:
    """Generate follow-up suggestions based on the last Q&A pair."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        pipeline.answer_generator.generate_suggestions,
        request.last_question,
        request.last_answer,
    )
    if result:
        return SuggestionResponse(suggestions=result, fallback=False)
    return SuggestionResponse(
        suggestions=_DEFAULT_SUGGESTIONS, fallback=True
    )
