"""End-to-end RAG pipeline with streaming support."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from rag_pipeline.config import RAGConfig
from rag_pipeline.generation.answer_generator import LLMAnswerGenerator
from rag_pipeline.generation.context_builder import (
    CitationContextBuilder,
    NoRelevantContextError,
)
from rag_pipeline.generation.models import (
    AnswerResult,
    BuiltContext,
    GenerationEvent,
)
from rag_pipeline.retrieval import FilterBuilder, HybridRetriever, LLMQueryProcessor
from rag_pipeline.retrieval.query_cache import QueryCache
from rag_pipeline.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_pipeline.storage.base import Storage


class RAGPipeline:
    """Full RAG pipeline: retrieval → context → answer generation."""

    _NO_CONTEXT_MESSAGE = "Không đủ thông tin để trả lờ i câu hỏi này."

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: CitationContextBuilder,
        answer_generator: LLMAnswerGenerator,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.context_builder = context_builder
        self.answer_generator = answer_generator

    @classmethod
    def from_config(
        cls,
        config: RAGConfig,
        storage: Storage,
        vector_store: Any,
        dense_embedder: Any,
        sparse_embedder: Any,
    ) -> "RAGPipeline":
        """Build a RAGPipeline from configuration and dependencies."""
        cache = QueryCache(config.storage.db_path)
        llm_processor = LLMQueryProcessor(config.llm_query, cache=cache)
        filter_builder = FilterBuilder()
        retriever = HybridRetriever(
            config=config.retrieval,
            storage=storage,
            vector_store=vector_store,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
        retrieval_pipeline = RetrievalPipeline(llm_processor, filter_builder, retriever)
        context_builder = CitationContextBuilder(config.context_builder)
        answer_generator = LLMAnswerGenerator(config.generation)
        return cls(retrieval_pipeline, context_builder, answer_generator)

    def answer(self, query: str) -> AnswerResult:
        """Generate a complete answer synchronously."""
        for event in self.answer_stream(query):
            if event.type == "done":
                return event.data
            if event.type == "error":
                return AnswerResult(
                    query=query,
                    answer=event.message or self._NO_CONTEXT_MESSAGE,
                    context="",
                    sources=[],
                )
        return AnswerResult(
            query=query,
            answer=self._NO_CONTEXT_MESSAGE,
            context="",
            sources=[],
        )

    def answer_stream(self, query: str) -> Iterator[GenerationEvent]:
        """Stream RAG progress events and answer tokens.

        Events are yielded as ``GenerationEvent`` objects. Consumers can
        serialize them to JSON for frontend streaming.
        """
        yield GenerationEvent(
            type="progress",
            step="rewrite",
            message="Đang viết lại câu hỏi...",
        )
        processed = self.retrieval_pipeline.preprocess(query)

        yield GenerationEvent(
            type="progress",
            step="retrieval",
            message="Đang tìm kiếm thông tin...",
        )
        results = self.retrieval_pipeline.search_processed(processed)

        if not results:
            yield GenerationEvent(
                type="error",
                message=self._NO_CONTEXT_MESSAGE,
            )
            return

        yield GenerationEvent(
            type="progress",
            step="context",
            message=f"Tìm thấy {len(results)} đoạn văn bản",
        )
        try:
            built = self.context_builder.build(results, query=query)
        except NoRelevantContextError:
            yield GenerationEvent(
                type="error",
                message=self._NO_CONTEXT_MESSAGE,
            )
            return

        yield GenerationEvent(
            type="progress",
            step="generation",
            message="Đang tạo câu trả lời...",
        )
        answer_parts: list[str] = []
        for token in self.answer_generator.generate_stream(query, built.context):
            answer_parts.append(token)
            yield GenerationEvent(type="token", data=token)

        answer = "".join(answer_parts)
        sources = self._extract_sources(answer, built)
        result = AnswerResult(
            query=query,
            answer=answer,
            context=built.context,
            sources=sources,
            intent=processed.intent,
        )
        yield GenerationEvent(type="done", data=result)

    @staticmethod
    def _extract_sources(answer: str, built: BuiltContext) -> list[dict[str, Any]]:
        """Extract citations used in the answer and map them to source metadata."""
        used_citations = sorted(set(re.findall(r"\[(\d+)\]", answer)))
        sources: list[dict[str, Any]] = []
        for num in used_citations:
            citation = f"[{num}]"
            if citation not in built.citations:
                continue
            result = built.citations[citation]
            sources.append(
                {
                    "citation": citation,
                    "title": result.metadata.get("title", ""),
                    "content": result.content,
                    "chunk_id": str(result.chunk_id),
                }
            )
        return sources

    def close(self) -> None:
        self.retrieval_pipeline.close()
        self.answer_generator.close()
