"""LLM clients for query processing."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Protocol

from rag_pipeline.config import LLMConfig


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send chat messages and return assistant response text."""


@dataclass(slots=True)
class OpenRouterLLMClient:
    """LLM client via OpenRouter /chat/completions endpoint.

    Features:
    - Bearer token auth from env var
    - Exponential backoff on 429 rate limit
    - JSON or text response parsing
    - Streaming support via stream()
    """

    config: LLMConfig
    retry_base_delay: float = 2.0

    def _get_api_key(self) -> str:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing environment variable `{self.config.api_key_env}` for OpenRouter access."
            )
        return api_key

    def _build_payload(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        return {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send chat messages and return assistant response text."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "The `httpx` package is required for LLM calls. "
                "Install with `pip install .[indexing]`."
            ) from exc

        api_key = self._get_api_key()
        payload = self._build_payload(messages, **kwargs)
        headers = {"Authorization": f"Bearer {api_key}"}

        with httpx.Client(base_url=self.config.api_base, timeout=self.config.timeout_seconds) as client:
            for attempt in range(self.config.max_retries + 1):
                response = client.post("/chat/completions", json=payload, headers=headers)

                if response.status_code == 429:
                    if attempt == self.config.max_retries:
                        response.raise_for_status()
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content if content is not None else ""

        return ""

    def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> Generator[str, None, None]:
        """Stream chat response token by token.

        Yields:
            Text chunks as they arrive from the LLM.
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "The `httpx` package is required for LLM calls. "
                "Install with `pip install .[indexing]`."
            ) from exc

        api_key = self._get_api_key()
        payload = self._build_payload(messages, **kwargs)
        payload["stream"] = True
        headers = {"Authorization": f"Bearer {api_key}"}

        with httpx.Client(base_url=self.config.api_base, timeout=self.config.timeout_seconds) as client:
            with client.stream("POST", "/chat/completions", json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Send chat messages and parse JSON response."""
        raw = self.chat(messages, **kwargs)
        if not raw:
            raise RuntimeError("LLM returned empty or null response")
        # Try to extract JSON from response (may be wrapped in markdown code block)
        text = raw.strip()
        if text.startswith("```"):
            # Remove markdown code block
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return json.loads(text)


@dataclass(slots=True)
class DeterministicTestLLM:
    """Fast deterministic LLM for dev/test — no API calls.

    Attributes:
        response_mode: Controls chat_json output format.
            - "query" (default): returns query rewrite fields
            - "generation": returns answer/citations/confidence fields
    """

    response_mode: str = "query"

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Return a canned response based on the last user message."""
        last_msg = messages[-1]["content"] if messages else ""
        return f"Test response for: {last_msg[:50]}"

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Return a canned JSON response based on response_mode."""
        last_msg = messages[-1]["content"] if messages else ""

        if self.response_mode == "generation":
            return {
                "answer": f"Đây là câu trả lời test cho: {last_msg[:80]}",
                "citations": [
                    {"claim": "claim test từ passage 1", "source_index": 1}
                ],
                "confidence": 0.8,
            }

        # Default: query rewrite mode
        return {
            "normalized_query": last_msg.lower().strip(),
            "rewrite_query": f"rewrite: {last_msg}",
            "bm25_query": last_msg.lower().strip(),
            "intent": "general",
        }
