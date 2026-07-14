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
        "Bạn là trợ lý du lịch Việt Nam. Hãy trả lời câu hỏi dựa vào ngữ cảnh được cung cấp.\n\n"
        "Yêu cầu:\n"
        "- Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng\n"
        "- Chỉ dựa vào thông tin trong ngữ cảnh\n"
        '- Nếu không đủ thông tin, hãy nói "Tôi không có đủ thông tin để trả lời"'
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
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(query, context)},
        ]
        yield from self.generate_stream_messages(messages)

    def generate_stream_messages(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream answer tokens using a pre-built message list.

        ``messages`` follows the OpenAI Chat Completions format:
        ``[{"role": ..., "content": ...}, ...]``. The list is sent as-is
        (apart from a server-side system prompt prefix), which is how
        the chat-memory layer plugs in earlier turns and summaries.
        """
        # Always ensure a system message at the top.
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self._SYSTEM_PROMPT}, *messages]

        last_exception: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with self._client.stream(
                    "POST",
                    "/chat/completions",
                    json={
                        "model": self.config.model_name,
                        "messages": messages,
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

    def generate_suggestions(
        self, question: str, answer: str, max_suggestions: int = 4
    ) -> list[str]:
        """Generate follow-up question suggestions based on the last Q&A pair.

        Returns a list of 1-``max_suggestions`` suggestion strings. On
        failure, returns an empty list so the caller can fall back to
        defaults.
        """
        prompt = (
            f"Câu hỏi vừa rồi: {question}\n"
            f"Câu trả lời: {answer}\n\n"
            f"Dựa vào đó, gợi ý {max_suggestions} câu hỏi tiếp theo mà người dùng "
            f"có thể quan tâm. Mỗi câu trên 1 dòng, không đánh số, không giải thích."
        )
        try:
            response = self._client.post(
                "/chat/completions",
                json={
                    "model": self.config.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Bạn là trợ lý gợi ý câu hỏi cho chat du lịch Việt Nam. "
                                "Chỉ trả về các câu hỏi, mỗi câu trên 1 dòng."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 128,
                    "temperature": 0.5,
                },
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            content = data["choices"][0]["message"]["content"]
            lines = [
                ln.strip().lstrip("0123456789.-) ").strip()
                for ln in content.strip().splitlines()
                if ln.strip()
            ]
            return lines[:max_suggestions]
        except Exception:
            return []
