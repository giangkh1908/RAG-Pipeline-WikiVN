"""Tests for generation components."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from rag_pipeline.config import ContextBuilderConfig, GenerationConfig
from rag_pipeline.generation import (
    AnswerResult,
    CitationContextBuilder,
    LLMAnswerGenerator,
    RAGPipeline,
)
from rag_pipeline.generation.context_builder import NoRelevantContextError
from rag_pipeline.retrieval.models import RetrievalResult


class TestCitationContextBuilder:
    def test_build_formats_chunks_with_citations(self) -> None:
        results = [
            RetrievalResult(
                chunk_id=uuid4(),
                content="Content one",
                rrf_score=0.9,
                rank=1,
                metadata={"title": "Title One"},
            ),
            RetrievalResult(
                chunk_id=uuid4(),
                content="Content two",
                rrf_score=0.8,
                rank=2,
                metadata={"title": "Title Two"},
            ),
        ]
        builder = CitationContextBuilder(ContextBuilderConfig(max_chunks=5))
        built = builder.build(results)

        assert "[1] Tiêu đề: Title One" in built.context
        assert "Content one" in built.context
        assert "[2] Tiêu đề: Title Two" in built.context
        assert len(built.citations) == 2
        assert "[1]" in built.citations

    def test_build_respects_max_chunks(self) -> None:
        results = [
            RetrievalResult(
                chunk_id=uuid4(),
                content=f"Content {i}",
                rrf_score=1.0 - i * 0.1,
                rank=i,
                metadata={},
            )
            for i in range(10)
        ]
        builder = CitationContextBuilder(ContextBuilderConfig(max_chunks=3))
        built = builder.build(results)

        assert len(built.citations) == 3
        assert "[3]" in built.citations
        assert "[4]" not in built.citations

    def test_build_raises_on_empty_results(self) -> None:
        builder = CitationContextBuilder()
        with pytest.raises(NoRelevantContextError):
            builder.build([])


class TestLLMAnswerGenerator:
    @patch("rag_pipeline.generation.answer_generator.httpx.Client")
    def test_generate_returns_full_answer(self, mock_client_class: MagicMock) -> None:
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        mock_response = self._mock_stream_response(["Hello ", "world"])
        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        generator = LLMAnswerGenerator(GenerationConfig())
        answer = generator.generate("Q", "C")

        assert answer.answer == "Hello world"
        assert answer.model_name == GenerationConfig().model_name

    @patch("rag_pipeline.generation.answer_generator.httpx.Client")
    def test_generate_stream_yields_tokens(self, mock_client_class: MagicMock) -> None:
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        mock_response = self._mock_stream_response(["Nha ", "Trang"])
        mock_client = MagicMock()
        mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_class.return_value = mock_client

        generator = LLMAnswerGenerator(GenerationConfig())
        tokens = list(generator.generate_stream("Q", "C"))

        assert tokens == ["Nha ", "Trang"]

    @staticmethod
    def _mock_stream_response(tokens: list[str]) -> MagicMock:
        """Build a mock httpx Response that yields SSE chunks."""
        lines = []
        for token in tokens:
            payload = json.dumps({"choices": [{"delta": {"content": token}}]})
            lines.append(f"data: {payload}")
        lines.append("data: [DONE]")

        response = MagicMock(spec=httpx.Response)
        response.iter_lines.return_value = lines
        response.raise_for_status.return_value = None
        return response


class TestRAGPipeline:
    def test_answer_stream_emits_progress_and_done(self) -> None:
        result = RetrievalResult(
            chunk_id=uuid4(),
            content="Chunk content",
            rrf_score=0.9,
            rank=1,
            metadata={"title": "Topic"},
        )

        retrieval_pipeline = MagicMock()
        processed = MagicMock()
        processed.rewritten_query = "rewritten"
        processed.normalized_query = "query"
        processed.intent = "factual"
        retrieval_pipeline.preprocess.return_value = processed
        retrieval_pipeline.search_processed.return_value = [result]

        context_builder = CitationContextBuilder()
        answer_generator = MagicMock()
        answer_generator.generate_stream.return_value = iter(["Answer ", "text"])

        pipeline = RAGPipeline(retrieval_pipeline, context_builder, answer_generator)
        events = list(pipeline.answer_stream("query"))

        progress_events = [e for e in events if e.type == "progress"]
        token_events = [e for e in events if e.type == "token"]
        done_events = [e for e in events if e.type == "done"]

        assert len(progress_events) == 4  # rewrite, retrieval, context, generation
        assert len(token_events) == 2
        assert len(done_events) == 1

        done = done_events[0].data
        assert isinstance(done, AnswerResult)
        assert done.answer == "Answer text"
        assert done.intent == "factual"
        assert len(done.sources) == 0  # answer has no citations

    def test_answer_stream_returns_error_when_no_results(self) -> None:
        retrieval_pipeline = MagicMock()
        processed = MagicMock()
        processed.rewritten_query = "rewritten"
        processed.normalized_query = "query"
        processed.intent = "factual"
        retrieval_pipeline.preprocess.return_value = processed
        retrieval_pipeline.search_processed.return_value = []

        pipeline = RAGPipeline(
            retrieval_pipeline,
            CitationContextBuilder(),
            MagicMock(),
        )
        events = list(pipeline.answer_stream("query"))

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "Không đủ thông tin" in error_events[0].message

    def test_answer_sync_returns_result(self) -> None:
        result = RetrievalResult(
            chunk_id=uuid4(),
            content="Chunk content",
            rrf_score=0.9,
            rank=1,
            metadata={"title": "Topic"},
        )

        retrieval_pipeline = MagicMock()
        processed = MagicMock()
        processed.rewritten_query = "rewritten"
        processed.normalized_query = "query"
        processed.intent = "factual"
        retrieval_pipeline.preprocess.return_value = processed
        retrieval_pipeline.search_processed.return_value = [result]

        answer_generator = MagicMock()
        answer_generator.generate_stream.return_value = iter(["Final ", "answer"])

        pipeline = RAGPipeline(
            retrieval_pipeline,
            CitationContextBuilder(),
            answer_generator,
        )
        answer_result = pipeline.answer("query")

        assert answer_result.answer == "Final answer"
