"""Tests for the FastAPI RAG endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from rag_pipeline.api.app import app
from rag_pipeline.api.dependencies import PipelineStore
from rag_pipeline.generation.models import AnswerResult, GenerationEvent


@pytest.fixture(autouse=True)
def _generation_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chat dependency may build the real generator via get_conversation_store;
    supply a dummy key so construction does not raise (the pipeline itself is mocked)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    result = AnswerResult(
        query="Ha Long Bay ở đâu?",
        answer="Vịnh Hạ Long nằm ở Quảng Ninh.",
        context="Context",
        sources=[
            {
                "citation": "[1]",
                "title": "Vịnh Hạ Long",
                "content": "Nội dung",
                "chunk_id": str(uuid4()),
            }
        ],
        intent="factual",
    )
    pipeline.answer.return_value = result
    pipeline.answer_stream.return_value = [
        GenerationEvent(type="progress", step="rewrite", message="Rewriting..."),
        GenerationEvent(type="token", data="Vịnh "),
        GenerationEvent(type="token", data="Hạ Long"),
        GenerationEvent(type="done", data=result),
    ]
    return pipeline


def test_health_endpoint(client: TestClient) -> None:
    with patch("rag_pipeline.api.routes.health.httpx.AsyncClient") as mock_client_class:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=False)

        response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["qdrant"] == "connected"


def test_chat_non_streaming(client: TestClient, mock_pipeline: MagicMock) -> None:
    with patch.object(PipelineStore, "get_pipeline", return_value=mock_pipeline):
        response = client.post("/api/chat", json={"question": "Ha Long Bay ở đâu?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Vịnh Hạ Long nằm ở Quảng Ninh."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["citation"] == "[1]"
    assert data["intent"] == "factual"


def test_chat_streaming(client: TestClient, mock_pipeline: MagicMock) -> None:
    with patch.object(PipelineStore, "get_pipeline", return_value=mock_pipeline):
        response = client.post("/api/chat/stream", json={"question": "Ha Long Bay ở đâu?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert 'type"' in body and "progress" in body
    assert "Vịnh " in body
    assert "Hạ Long" in body
    assert "done" in body
