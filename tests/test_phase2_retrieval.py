"""Tests for Phase 2 query preprocessing and retrieval pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from rag_pipeline.config import LLMQueryConfig
from rag_pipeline.retrieval import (
    FilterBuilder,
    HybridRetriever,
    LLMQueryProcessor,
    ProcessedQuery,
    QueryCache,
    RetrievalPipeline,
)
from rag_pipeline.storage import SQLiteStorage


@pytest.fixture
def temp_cache() -> QueryCache:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = QueryCache(db_path=os.path.join(tmpdir, "rag_storage.db"))
        yield cache
        cache.close()


@pytest.fixture
def llm_config() -> LLMQueryConfig:
    return LLMQueryConfig(
        model_name="test-model",
        api_key_env="OPENROUTER_API_KEY",
        prompt_version="v1",
        fallback_to_normalized=True,
    )


class TestQueryCache:
    def test_cache_miss_returns_none(self, temp_cache: QueryCache) -> None:
        assert temp_cache.get("m", "v1", "hello") is None

    def test_cache_hit_returns_cached_query(self, temp_cache: QueryCache) -> None:
        from rag_pipeline.retrieval.query_cache import CachedQuery

        cached = CachedQuery(
            raw_query="hello",
            rewritten_query="hello world",
            intent="factual",
            model_name="m",
            prompt_version="v1",
        )
        temp_cache.set(cached)

        result = temp_cache.get("m", "v1", "hello")
        assert result is not None
        assert result.rewritten_query == "hello world"
        assert result.intent == "factual"

    def test_cache_respects_ttl(self, temp_cache: QueryCache) -> None:
        from rag_pipeline.retrieval.query_cache import CachedQuery

        cached = CachedQuery(
            raw_query="hello",
            rewritten_query="hello world",
            intent="factual",
            model_name="m",
            prompt_version="v1",
        )
        temp_cache.set(cached)
        assert temp_cache.get("m", "v1", "hello", ttl_days=-1) is None

    def test_cache_key_depends_on_model_and_prompt(self, temp_cache: QueryCache) -> None:
        from rag_pipeline.retrieval.query_cache import CachedQuery

        cached = CachedQuery(
            raw_query="hello",
            rewritten_query="hello world",
            intent="factual",
            model_name="m",
            prompt_version="v1",
        )
        temp_cache.set(cached)

        assert temp_cache.get("m", "v2", "hello") is None
        assert temp_cache.get("other", "v1", "hello") is None


class TestLLMQueryProcessor:
    @patch("rag_pipeline.retrieval.llm_query_processor.httpx.Client")
    def test_process_calls_llm_and_caches_result(
        self,
        mock_client_class: MagicMock,
        temp_cache: QueryCache,
        llm_config: LLMQueryConfig,
    ) -> None:
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "rewritten_query": "câu hỏi về du lịch việt nam",
                                "intent": "factual",
                                "reasoning": "test",
                            }
                        )
                    }
                }
            ]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        processor = LLMQueryProcessor(config=llm_config, cache=temp_cache)
        result = processor.process("du lịch việt nam")

        assert result.rewritten_query == "câu hỏi về du lịch việt nam"
        assert result.intent == "factual"
        assert result.from_cache is False

        # Second call should hit cache
        result2 = processor.process("du lịch việt nam")
        assert result2.from_cache is True
        assert mock_client.post.call_count == 1

    @patch("rag_pipeline.retrieval.llm_query_processor.httpx.Client")
    def test_process_fallback_on_llm_failure(
        self,
        mock_client_class: MagicMock,
        temp_cache: QueryCache,
        llm_config: LLMQueryConfig,
    ) -> None:
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        mock_client = MagicMock()
        mock_client.post.side_effect = RuntimeError("network error")
        mock_client_class.return_value = mock_client

        processor = LLMQueryProcessor(config=llm_config, cache=temp_cache)
        result = processor.process("  Nha Trang  ")

        assert result.normalized_query == "nha trang"
        assert result.rewritten_query == "nha trang"
        assert result.intent == "factual"


class TestFilterBuilder:
    def test_excludes_reference_sections_for_common_intents(self) -> None:
        builder = FilterBuilder()
        for intent in ["factual", "recommendation", "comparison", "list"]:
            processed = ProcessedQuery(
                raw_query="x",
                normalized_query="x",
                rewritten_query="x",
                intent=intent,
            )
            filters = builder.build(processed)
            assert filters is not None
            assert filters.get("is_reference_section") is False

    def test_no_filter_for_procedural_intent(self) -> None:
        builder = FilterBuilder()
        processed = ProcessedQuery(
            raw_query="x",
            normalized_query="x",
            rewritten_query="x",
            intent="procedural",
        )
        filters = builder.build(processed)
        assert filters is None


class TestRetrievalPipeline:
    def test_pipeline_orchestrates_preprocessor_and_retriever(self) -> None:
        storage = SQLiteStorage(":memory:")

        llm_processor = MagicMock(spec=LLMQueryProcessor)
        llm_processor.process.return_value = ProcessedQuery(
            raw_query="Nha Trang",
            normalized_query="nha trang",
            rewritten_query="du lịch nha trang",
            intent="recommendation",
        )

        retriever = MagicMock(spec=HybridRetriever)
        retriever.retrieve.return_value = []

        pipeline = RetrievalPipeline(
            llm_processor=llm_processor,
            filter_builder=FilterBuilder(),
            retriever=retriever,
        )
        pipeline.search("Nha Trang", top_k=2)

        llm_processor.process.assert_called_once_with("Nha Trang", None)
        retriever.retrieve.assert_called_once()
        assert retriever.retrieve.call_args.args[0] == "du lịch nha trang"
        assert retriever.retrieve.call_args.kwargs["top_k"] == 2
        assert retriever.retrieve.call_args.kwargs["filters"] == {"is_reference_section": False}

        storage.close()
