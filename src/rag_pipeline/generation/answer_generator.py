"""LLM-based answer generator with streaming support."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

import httpx

from rag_pipeline.config import GenerationConfig
from rag_pipeline.generation.models import GeneratedAnswer


class LLMAnswerGenerator:
    """Generate answers using an LLM via OpenRouter."""

    _SYSTEM_PROMPT = (
        "Bạn là trợ lý du lịch Việt Nam. Hãy trả lờ i câu hỏi dựa vào ngữ cảnh được cung cấp.\n\n"
        "Yêu cầu:\n"
        "- Trả lờ i bằng tiếng Việt, ngắn gọn, rõ ràng\n"
        "- Chỉ dựa vào thông tin trong ngữ cảnh\n"
        '- Nếu không đủ thông tin, hãy nói "Tôi không có đủ thông tin để trả lờ i"\n'
        "- Trích dẫn nguồn bằng các số trong ngoặc vuông [1], [2], ..."
    )

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self._client = httpx.Client(
            base_url=self.config.api_base,
            timeout=self.config.timeout_seconds,
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://rag-pipeline.local",
                "X-Title": "RAG Pipeline",
            },
        )

    def _api_key(self) -> str:
        key = os.getenv(self.config.api_key_env)
        if not key:
            raise RuntimeError(
                f"Missing API key: set the {self.config.api_key_env} environment variable"
            )
        return key

    @staticmethod
    def _user_prompt(query: str, context: str) -> str:
        return f"""Ngữ cảnh:
{context}

Câu hỏi: {query}"""

    def generate(self, query: str, context: str) -> GeneratedAnswer:
        """Generate a complete answer synchronously."""
        answer = ""
        for token in self.generate_stream(query, context):
            answer += token
        return GeneratedAnswer(answer=answer, model_name=self.config.model_name)

    def generate_stream(self, query: str, context: str) -> Iterator[str]:
        """Stream answer tokens from the LLM.

        Yields individual content tokens as they arrive from the model.
        """
        last_exception: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with self._client.stream(
                    "POST",
                    "/chat/completions",
                    json={
                        "model": self.config.model_name,
                        "messages": [
                            {"role": "system", "content": self._SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": self._user_prompt(query, context),
                            },
                        ],
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    yield from self._parse_stream(response)
                return
            except Exception as exc:
                last_exception = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"Answer generation failed after {self.config.max_retries} retries"
        ) from last_exception

    def _parse_stream(self, response: httpx.Response) -> Iterator[str]:
        """Parse Server-Sent Events from an OpenAI-compatible streaming response."""
        for line in response.iter_lines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                parsed: dict[str, Any] = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = parsed.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content

    def close(self) -> None:
        self._client.close()
