"""Tests for FastAPI API endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rag_pipeline.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert "qdrant" in data
        assert "langsmith" in data
        assert "version" in data

    def test_health_status_values(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert data["status"] in ["ok", "degraded", "error"]
        assert data["qdrant"] in ["connected", "disconnected"]
        assert data["langsmith"] in ["enabled", "disabled"]


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_format(self, client):
        response = client.get("/")
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            # Frontend is built — root serves index.html
            assert response.status_code == 200
        else:
            data = response.json()
            assert "name" in data


class TestChatEndpoint:
    """Tests for POST /api/chat."""

    def test_chat_returns_200(self, client):
        response = client.post("/api/chat", json={"question": "Test?"})
        assert response.status_code == 200

    def test_chat_response_format(self, client):
        response = client.post("/api/chat", json={"question": "Test?"})
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert "latency_ms" in data

    def test_chat_empty_question_rejected(self, client):
        response = client.post("/api/chat", json={"question": ""})
        assert response.status_code == 422  # Validation error

    def test_chat_missing_question_rejected(self, client):
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_with_history(self, client):
        response = client.post("/api/chat", json={
            "question": "Dân số bao nhiêu?",
            "history": [
                {"role": "user", "content": "Thủ đô Việt Nam ở đâu?"},
                {"role": "assistant", "content": "Thủ đô Việt Nam là Hà Nội."},
            ],
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data


class TestChatStreamEndpoint:
    """Tests for POST /api/chat/stream."""

    def test_stream_returns_200(self, client):
        response = client.post("/api/chat/stream", json={"question": "Test"})
        assert response.status_code == 200

    def test_stream_content_type(self, client):
        response = client.post("/api/chat/stream", json={"question": "Test"})
        assert "text/event-stream" in response.headers["content-type"]

    def test_stream_has_data(self, client):
        response = client.post("/api/chat/stream", json={"question": "Test"})
        content = response.text
        assert "data:" in content

    def test_stream_empty_question_rejected(self, client):
        response = client.post("/api/chat/stream", json={"question": ""})
        assert response.status_code == 422

    def test_stream_with_history(self, client):
        response = client.post("/api/chat/stream", json={
            "question": "Dân số bao nhiêu?",
            "history": [
                {"role": "user", "content": "Thủ đô Việt Nam ở đâu?"},
                {"role": "assistant", "content": "Thủ đô Việt Nam là Hà Nội."},
            ],
        })
        assert response.status_code == 200
        assert "data:" in response.text


class TestEvalEndpoint:
    """Tests for POST /api/eval."""

    def test_eval_returns_200(self, client):
        pytest.importorskip("ragas")
        response = client.post("/api/eval", json={})
        assert response.status_code == 200

    def test_eval_response_format(self, client):
        pytest.importorskip("ragas")
        response = client.post("/api/eval", json={})
        data = response.json()
        assert "scores" in data
        assert "latency" in data
        assert "sample_count" in data
        assert "passed" in data
