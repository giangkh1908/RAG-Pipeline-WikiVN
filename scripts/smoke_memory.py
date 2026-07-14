"""End-to-end smoke test for the chat memory feature.

Runs against the in-process FastAPI app via ``TestClient``. The RAG
pipeline (LLM, embedder, Qdrant) is replaced with a fake so we can
exercise session lifecycle, validation, and the chat API contracts
without any external infrastructure.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Switch cwd into a temp dir so the default SQLite path resolves to
# ``<tmp>/data/rag_storage.db`` and the smoke run leaves no trace.
TMP_ROOT = Path(tempfile.mkdtemp(prefix="rag_smoke_"))
os.chdir(TMP_ROOT)

# Make sure env has an API key so generation init doesn't fail.
os.environ.setdefault("OPENROUTER_API_KEY", "smoke-test-key")

import rag_pipeline.api.dependencies as deps  # noqa: E402
from rag_pipeline.generation.models import AnswerResult, GenerationEvent  # noqa: E402

# Open a real ConversationStore against the temp cwd so turn bookkeeping
# is exercised the same way it would be in production.
from rag_pipeline.api.dependencies import _build_app_state  # noqa: E402

_build_app_state()
_conv_store = deps._state.conversation_store
assert _conv_store is not None, "conversation_store must be created"

# Build a fake RAGPipeline that mirrors the real pipeline's
# session/turn bookkeeping so the smoke test exercises the same
# ConversationStore state machine without LLM/embedder/Qdrant.
from rag_pipeline.storage.conversation import ChatTurn  # noqa: E402

_fake_pipeline = MagicMock()


def _persist_and_answer(query: str, session_id: str | None) -> AnswerResult:
    turn_no: int | None = None
    if session_id is not None:
        _conv_store.upsert_session(session_id)
        turn_no = _conv_store.next_turn_no(session_id)
        _conv_store.insert_turn(
            ChatTurn(session_id=session_id, turn_no=turn_no, question=query)
        )
        _conv_store.update_turn_answer(
            session_id, turn_no, f"[fake] {query}", "factual", 5
        )
    return AnswerResult(
        query=query,
        answer=f"[fake] {query}",
        context="",
        sources=[],
        intent="factual",
        session_id=session_id,
        turn_no=turn_no,
        memory_used=session_id is not None,
    )


def _fake_answer(query: str, session_id: str | None = None) -> AnswerResult:
    return _persist_and_answer(query, session_id)


def _fake_stream(query: str, session_id: str | None = None):
    yield GenerationEvent(type="progress", step="rewrite", message="...")
    yield GenerationEvent(type="token", data=f"[fake] {query}")
    yield GenerationEvent(type="done", data=_persist_and_answer(query, session_id))


_fake_pipeline.answer.side_effect = _fake_answer
_fake_pipeline.answer_stream.side_effect = _fake_stream
# Suggestions: fake pipeline has no real LLM → return [] → fallback defaults.
_fake_pipeline.answer_generator.generate_suggestions.return_value = []

# Swap the real pipeline (which needs Qdrant/LLM) for our fake.
deps._state.pipeline = _fake_pipeline  # type: ignore[attr-defined]

from fastapi.testclient import TestClient  # noqa: E402

from rag_pipeline.api.app import app  # noqa: E402


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> int:
    client = TestClient(app)

    banner("1. Health check")
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] in {"ok", "degraded", "error"}
    assert data["version"] == "0.3.0"
    print(f"  health = {data}")
    print("  ✓ health ok")

    banner("2. Chat without session_id (server auto-creates one)")
    r = client.post("/api/chat", json={"question": "Vinh Ha Long o dau?"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["session_id"], "session_id should be auto-generated"
    assert data["turn_no"] == 1
    # The server minted a fresh session, so memory bookkeeping is on.
    assert data["memory_used"] is True
    assert data["answer"] == "[fake] Vinh Ha Long o dau?"
    sid = data["session_id"]
    print(f"  session_id = {sid}")
    print(f"  turn_no    = {data['turn_no']}")
    print(f"  answer     = {data['answer']!r}")
    print(f"  memory_used= {data['memory_used']}")
    print("  ✓ chat works, server mints session_id on the fly")

    banner("3. Chat with same session_id (memory continuity)")
    r = client.post(
        "/api/chat",
        json={"question": "Di mua nao dep?", "session_id": sid},
    )
    assert r.status_code == 200, r.text
    data2 = r.json()
    assert data2["session_id"] == sid
    assert data2["turn_no"] == 2, f"expected turn 2 got {data2['turn_no']}"
    assert data2["memory_used"] is True
    print(f"  turn_no    = {data2['turn_no']} (incremented)")
    print(f"  memory_used= {data2['memory_used']}")
    print("  ✓ session persists turn counter + memory flag")

    banner("4. Validation: malformed session_id rejected")
    bad_sid = "short"  # too short (< 8 chars)
    r = client.post(
        "/api/chat",
        json={"question": "test", "session_id": bad_sid},
    )
    assert r.status_code == 422, f"expected 422 got {r.status_code}: {r.text}"
    print(f"  status = 422 (rejected {bad_sid!r})")
    print("  ✓ validator works")

    banner("5. Validation: question too long rejected")
    long_q = "x" * 501
    r = client.post("/api/chat", json={"question": long_q})
    assert r.status_code == 422, r.text
    print(f"  status = 422 (rejected 501-char question)")
    print("  ✓ max_length=500 enforced")

    banner("6. Streaming endpoint basic shape")
    r = client.post(
        "/api/chat/stream",
        json={"question": "test stream", "session_id": sid},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "data:" in body
    assert "[fake] test stream" in body
    # The done event carries session_id + turn_no.
    assert sid in body
    print("  ✓ SSE stream emits session_id, turn_no, answer")

    banner("7. Session delete")
    r = client.delete(f"/api/session/{sid}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["deleted"] is True
    assert data["deleted_turns"] >= 2
    print(f"  deleted_turns = {data['deleted_turns']}")
    print("  ✓ session hard-deleted")

    # After delete, a chat with the same session_id should start fresh
    # (turn 1 again).
    r = client.post(
        "/api/chat",
        json={"question": "After delete", "session_id": sid},
    )
    assert r.status_code == 200, r.text
    assert r.json()["turn_no"] == 1, "session should restart at turn 1"
    print("  ✓ re-using id after delete starts at turn 1")

    banner("8. Session GC")
    r = client.post("/api/session/gc")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "deleted" in data
    print(f"  gc result = {data}")
    print("  ✓ GC endpoint responds")

    banner("9. Suggestions endpoint (fallback to defaults)")
    r = client.post(
        "/api/suggestions",
        json={
            "session_id": sid,
            "last_question": "Vinh Ha Long o dau?",
            "last_answer": "Vinh Ha Long nam o Quang Ninh.",
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)
    # With the fake pipeline, the LLM call will fail → fallback defaults.
    assert len(data["suggestions"]) > 0
    print(f"  suggestions = {data['suggestions'][:2]}...")
    print(f"  fallback = {data['fallback']}")
    print("  ✓ suggestions endpoint responds with fallback defaults")

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        try:
            deps.PipelineStore.close()
        except Exception:  # noqa: BLE001
            pass
        # Best-effort cleanup of the temp dir.
        try:
            shutil.rmtree(TMP_ROOT, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
    sys.exit(code)
