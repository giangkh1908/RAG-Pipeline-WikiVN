"""LLM-based query preprocessor: rewrite + intent classification."""

from __future__ import annotations

import json
import os
import time

import httpx

from rag_pipeline.config import LLMQueryConfig
from rag_pipeline.retrieval.query_cache import CachedQuery, QueryCache


class ProcessedQuery:
    """Result of query preprocessing."""

    def __init__(
        self,
        raw_query: str,
        normalized_query: str,
        rewritten_query: str,
        intent: str,
        from_cache: bool = False,
    ) -> None:
        self.raw_query = raw_query
        self.normalized_query = normalized_query
        self.rewritten_query = rewritten_query
        self.intent = intent
        self.from_cache = from_cache


class LLMQueryProcessor:
    """Rewrite queries and classify intent using an LLM via OpenRouter."""

    _SYSTEM_PROMPT = """Bạn là hệ thống xử lý truy vấn cho RAG về Du lịch Việt Nam.

Nhiệm vụ:
1. Viết lại truy vấn của người dùng để tìm kiếm thông tin du lịch Việt Nam tốt hơn.
2. Phân loại ý định (intent) của truy vấn thành một trong các loại:
   - factual: hỏi thông tin thực tế (là gì, ở đâu, khi nào, ...)
   - recommendation: đề xuất, gợi ý (nên đi đâu, tốt nhất, ...)
   - comparison: so sánh (với, so với, khác nhau, ...)
   - list: danh sách (các, những, danh sách, ...)
   - procedural: hướng dẫn (làm thế nào, cách, thủ tục, ...)

Quy tắc viết lại:
- Giữ nguyên ngôn ngữ tiếng Việt
- Mở rộng từ viết tắt và thuật ngữ mơ hồ
- Thêm từ khóa du lịch nếu cần
- Không thêm thông tin không có trong truy vấn gốc
- Nếu có [Ngữ cảnh hội thoại], dùng nó để giải thích các từ tham chiếu
  (ví dụ: "nó", "chỗ đó", "5 cái trên", "cái kia", ...) bằng nội dung cụ thể
  từ ngữ cảnh. Ví dụ: nếu ngữ cảnh nhắc "Hạ Long, Hội An, Đà Nẵng" và truy vấn
  là "5 cái trên có gì chơi?", viết lại thành "Hạ Long Hội An Đà Nẵng có gì chơi".

Chỉ trả về JSON hợp lệ theo định dạng:
{{"rewritten_query": "...", "intent": "...", "reasoning": "..."}}"""

    def __init__(
        self,
        config: LLMQueryConfig | None = None,
        cache: QueryCache | None = None,
    ) -> None:
        self.config = config or LLMQueryConfig()
        self.cache = cache or QueryCache()
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
    def normalize_query(query: str) -> str:
        """Normalize a raw query string."""
        return " ".join(query.lower().split())

    def process(
        self, query: str, conversation_context: str | None = None
    ) -> ProcessedQuery:
        """Process a query, using cache when available.

        When ``conversation_context`` is provided, the cache is bypassed
        because the rewrite depends on prior turns, not just the raw
        query.
        """
        normalized = self.normalize_query(query)

        if conversation_context is None:
            cached = self.cache.get(
                self.config.model_name,
                self.config.prompt_version,
                query,
                ttl_days=self.config.cache_ttl_days,
            )
            if cached is not None:
                return ProcessedQuery(
                    raw_query=query,
                    normalized_query=normalized,
                    rewritten_query=cached.rewritten_query,
                    intent=cached.intent,
                    from_cache=True,
                )

        try:
            rewritten, intent = self._call_llm(normalized, conversation_context)
        except Exception:
            if self.config.fallback_to_normalized:
                rewritten, intent = normalized, "factual"
            else:
                raise

        # Only cache results that don't depend on conversation context.
        if conversation_context is None:
            self.cache.set(
                CachedQuery(
                    raw_query=query,
                    rewritten_query=rewritten,
                    intent=intent,
                    model_name=self.config.model_name,
                    prompt_version=self.config.prompt_version,
                )
            )

        return ProcessedQuery(
            raw_query=query,
            normalized_query=normalized,
            rewritten_query=rewritten,
            intent=intent,
            from_cache=False,
        )

    def _call_llm(
        self, normalized_query: str, conversation_context: str | None = None
    ) -> tuple[str, str]:
        last_exception: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                if conversation_context:
                    user_content = (
                        f"[Ngữ cảnh hội thoại]\n{conversation_context}\n\n"
                        f"[Truy vấn cần viết lại]\n{normalized_query}"
                    )
                else:
                    user_content = normalized_query

                response = self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.config.model_name,
                        "messages": [
                            {"role": "system", "content": self._SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.1,
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return str(parsed["rewritten_query"]), str(parsed["intent"])
            except Exception as exc:
                last_exception = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"LLM query preprocessing failed after {self.config.max_retries} retries"
        ) from last_exception

    def close(self) -> None:
        self._client.close()
