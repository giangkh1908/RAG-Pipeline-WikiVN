"""Tests for the LLM-judge output classifier (judge.py)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from rag_pipeline.generation.judge import judge_answer


def _mock_post_response(content: str | None) -> MagicMock:
    """Build a mock non-streaming httpx response returning ``content``."""
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status.return_value = None
    if content is None:
        resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
    else:
        resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _client_with_post(post_return: MagicMock | Exception) -> MagicMock:
    client = MagicMock(spec=httpx.Client)
    if isinstance(post_return, Exception):
        client.post.side_effect = post_return
    else:
        client.post.return_value = post_return
    return client


class TestJudgeAnswer:
    def test_valid_true_returns_dict(self) -> None:
        client = _client_with_post(_mock_post_response('{"valid": true, "reason": "ok"}'))
        out = judge_answer(client, "model", "Q?", "A")
        assert out == {"valid": True, "reason": "ok"}

    def test_valid_false_returns_dict(self) -> None:
        client = _client_with_post(
            _mock_post_response('{"valid": false, "reason": "in số 1..50"}')
        )
        out = judge_answer(client, "model", "Q?", "A")
        assert out == {"valid": False, "reason": "in số 1..50"}

    def test_strips_code_fences(self) -> None:
        fenced = '```json\n{"valid": true, "reason": "x"}\n```'
        client = _client_with_post(_mock_post_response(fenced))
        out = judge_answer(client, "model", "Q?", "A")
        assert out == {"valid": True, "reason": "x"}

    def test_strips_bare_fences(self) -> None:
        fenced = '```\n{"valid": false, "reason": "y"}\n```'
        client = _client_with_post(_mock_post_response(fenced))
        out = judge_answer(client, "model", "Q?", "A")
        assert out == {"valid": False, "reason": "y"}

    def test_http_error_returns_none(self) -> None:
        client = _client_with_post(httpx.HTTPStatusError("boom", request=httpx.Request("POST", "http://x"), response=httpx.Response(500)))
        assert judge_answer(client, "model", "Q?", "A") is None

    def test_network_error_returns_none(self) -> None:
        client = _client_with_post(httpx.ConnectError("down"))
        assert judge_answer(client, "model", "Q?", "A") is None

    def test_timeout_returns_none(self) -> None:
        client = _client_with_post(httpx.TimeoutException("slow"))
        assert judge_answer(client, "model", "Q?", "A") is None

    def test_malformed_json_returns_none(self) -> None:
        client = _client_with_post(_mock_post_response("not json at all"))
        assert judge_answer(client, "model", "Q?", "A") is None

    def test_empty_content_returns_none(self) -> None:
        client = _client_with_post(_mock_post_response(None))
        assert judge_answer(client, "model", "Q?", "A") is None

    def test_dict_missing_valid_key_returns_none(self) -> None:
        client = _client_with_post(_mock_post_response('{"reason": "no valid field"}'))
        assert judge_answer(client, "model", "Q?", "A") is None

    def test_non_dict_json_returns_none(self) -> None:
        client = _client_with_post(_mock_post_response('["valid", true]'))
        assert judge_answer(client, "model", "Q?", "A") is None

    def test_sends_deterministic_temperature(self) -> None:
        client = _client_with_post(_mock_post_response('{"valid": true, "reason": "ok"}'))
        judge_answer(client, "model", "Q?", "A", max_tokens=123, temperature=0.0)
        payload = client.post.call_args.kwargs["json"]
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 123
        assert payload["model"] == "model"
        assert len(payload["messages"]) == 2
        assert "Câu hỏi của người dùng:\nQ?" in payload["messages"][1]["content"]
        assert "Câu trả lời của trợ lý:\nA" in payload["messages"][1]["content"]