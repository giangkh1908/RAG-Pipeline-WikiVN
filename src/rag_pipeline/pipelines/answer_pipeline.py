"""Answer pipeline — orchestrates query → retrieval → generation → guardrails."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from rag_pipeline.generation.answer_generator import AnswerGenerator
from rag_pipeline.generation.output_guardrails import OutputGuardrails
from rag_pipeline.models import AnswerResult, ProcessedQuery
from rag_pipeline.pipelines.query_pipeline import QueryPipeline
from rag_pipeline.pipelines.retrieval_pipeline import RetrievalPipeline


def _traceable(name: str):
    """Decorator that wraps a function with LangSmith tracing if available, otherwise no-op."""
    try:
        from langsmith import traceable
        return traceable(name=name)
    except ImportError:
        # No langsmith — return identity decorator
        def identity_decorator(fn):
            return fn
        return identity_decorator


@dataclass(slots=True)
class AnswerPipeline:
    """Orchestrates the full RAG pipeline: query → retrieval → generation.

    Flow:
    1. QueryPipeline: guardrails → normalize → rewrite
    2. RetrievalPipeline: dense + BM25 → RRF → rerank
    3. AnswerGenerator: prompt → LLM → parse AnswerResult
    4. OutputGuardrails: hallucination + safety + quality check
    """

    query_pipeline: QueryPipeline
    retrieval_pipeline: RetrievalPipeline
    answer_generator: AnswerGenerator
    output_guardrails: OutputGuardrails

    @_traceable("answer_pipeline.ask")
    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> AnswerResult:
        """Run full RAG pipeline: question → answer with citations.

        Args:
            question: User question in natural language
            history: Optional conversation history [{role, content}, ...]

        Returns:
            AnswerResult with answer, citations, and confidence
        """
        # Step 1: Query processing (Phase 2)
        processed_query = self._run_query_processing(question)

        # Step 2: Retrieval (Phase 3)
        retrieval_result = self._run_retrieval(processed_query)

        # Step 3: Generation (Phase 4)
        answer_result = self._run_generation(retrieval_result, history=history)

        # Step 4: Output guardrails
        checked_result = self._run_output_guardrails(answer_result, retrieval_result)

        return checked_result

    @_traceable("query_processing")
    def _run_query_processing(self, question: str):
        return self.query_pipeline.run(question, qid="ask")

    def _run_query_processing_fast(self, question: str) -> ProcessedQuery:
        """Fast query processing — skip LLM rewrite, use normalization only.

        Saves ~4 seconds by not calling LLM for query reformulation.
        """
        normalized = self.query_pipeline.normalizer.normalize(question)
        return ProcessedQuery(
            qid=str(uuid.uuid4()),
            original_query=question,
            normalized_query=normalized.normalized_text,
            rewrite_query=question,
            bm25_query=normalized.normalized_text,
            intent=normalized.intent,
            filters=normalized.filters,
            risk_flags=[],
        )

    @_traceable("retrieval")
    def _run_retrieval(self, processed_query):
        return self.retrieval_pipeline.run(processed_query)

    @_traceable("generation")
    def _run_generation(self, retrieval_result, history=None):
        return self.answer_generator.generate(retrieval_result, history=history)

    @_traceable("output_guardrails")
    def _run_output_guardrails(self, answer_result, retrieval_result):
        return self.output_guardrails.check(answer_result, retrieval_result)
