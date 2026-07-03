"""Tests for Phase 4: Generation — prompt builder, answer generator, output guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_pipeline.config import GenerationConfig, OutputGuardrailsConfig
from rag_pipeline.generation.answer_generator import AnswerGenerator
from rag_pipeline.generation.output_guardrails import OutputGuardrails
from rag_pipeline.generation.prompt_builder import PromptBuilder
from rag_pipeline.indexing.llm_client import DeterministicTestLLM
from rag_pipeline.models import AnswerResult, Citation, Passage, ProcessedQuery, RetrievalResult
from rag_pipeline.pipelines.answer_pipeline import AnswerPipeline
from rag_pipeline.pipelines.query_pipeline import QueryPipeline
from rag_pipeline.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_pipeline.query.guardrails import QueryGuardrails
from rag_pipeline.query.normalizer import QueryNormalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_processed_query(question: str = "Wikipedia là gì?") -> ProcessedQuery:
    return ProcessedQuery(
        qid="test-1",
        original_query=question,
        normalized_query=question.lower(),
        rewrite_query=question,
        bm25_query=question.lower(),
        intent="definition",
    )


def _make_passages() -> list[Passage]:
    return [
        Passage(
            chunk_id="chunk-001",
            doc_id="doc-001",
            title="Wikipedia",
            text="Wikipedia là bách khoa toàn thư mở, được viết bởi các tình nguyện viên trên toàn thế giới.",
            source_url="https://vi.wikipedia.org/wiki/Wikipedia",
            dense_score=0.85,
            bm25_score=3.2,
            rrf_score=0.031,
            rerank_score=0.92,
            rank=1,
        ),
        Passage(
            chunk_id="chunk-002",
            doc_id="doc-002",
            title="Bách khoa toàn thư",
            text="Bách khoa toàn thư là loại sách tham khảo chứa đựng kiến thức về nhiều lĩnh vực.",
            source_url="https://vi.wikipedia.org/wiki/Bách_khoa_toàn_thư",
            dense_score=0.72,
            bm25_score=2.1,
            rrf_score=0.025,
            rerank_score=0.78,
            rank=2,
        ),
    ]


def _make_retrieval_result(
    question: str = "Wikipedia là gì?",
    passages: list[Passage] | None = None,
) -> RetrievalResult:
    if passages is None:
        passages = _make_passages()
    query = _make_processed_query(question)
    context = "\n\n".join(f"[{p.rank}] ({p.title}) {p.text}" for p in passages)
    return RetrievalResult(query=query, passages=passages, context=context)


def _build_test_pipeline() -> AnswerPipeline:
    """Build a full test pipeline with deterministic components."""
    gen_config = GenerationConfig()
    output_config = OutputGuardrailsConfig()
    llm = DeterministicTestLLM(response_mode="generation")

    prompt_builder = PromptBuilder(gen_config)
    answer_generator = AnswerGenerator(
        llm_client=llm,
        prompt_builder=prompt_builder,
        config=gen_config,
    )
    output_guardrails = OutputGuardrails(output_config)

    # Query pipeline with test components
    query_pipeline = QueryPipeline(
        config=__import__("rag_pipeline.config", fromlist=["QueryConfig"]).QueryConfig(),
        normalizer=QueryNormalizer(),
        guardrails=QueryGuardrails(),
        rewriter=None,
    )

    # Retrieval pipeline is not needed for unit tests — we mock retrieval_result
    return AnswerPipeline(
        query_pipeline=query_pipeline,
        retrieval_pipeline=None,  # type: ignore[arg-type]
        answer_generator=answer_generator,
        output_guardrails=output_guardrails,
    )


# ---------------------------------------------------------------------------
# PromptBuilder tests
# ---------------------------------------------------------------------------


class TestPromptBuilder:
    def test_build_returns_two_messages(self) -> None:
        config = GenerationConfig()
        builder = PromptBuilder(config)
        result = _make_retrieval_result()
        messages = builder.build(result)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_contains_instructions(self) -> None:
        config = GenerationConfig()
        builder = PromptBuilder(config)
        result = _make_retrieval_result()
        messages = builder.build(result)
        system_content = messages[0]["content"]
        assert "JSON" in system_content
        assert "citations" in system_content
        assert "tiếng Việt" in system_content

    def test_user_message_contains_passages(self) -> None:
        config = GenerationConfig()
        builder = PromptBuilder(config)
        result = _make_retrieval_result()
        messages = builder.build(result)
        user_content = messages[1]["content"]
        assert "[1]" in user_content
        assert "[2]" in user_content
        assert "Wikipedia" in user_content

    def test_user_message_contains_question(self) -> None:
        config = GenerationConfig()
        builder = PromptBuilder(config)
        result = _make_retrieval_result("Ai sáng lập Wikipedia?")
        messages = builder.build(result)
        user_content = messages[1]["content"]
        assert "Ai sáng lập Wikipedia?" in user_content

    def test_empty_passages(self) -> None:
        config = GenerationConfig()
        builder = PromptBuilder(config)
        result = _make_retrieval_result(passages=[])
        messages = builder.build(result)
        user_content = messages[1]["content"]
        assert "Không tìm thấy" in user_content


# ---------------------------------------------------------------------------
# AnswerGenerator tests
# ---------------------------------------------------------------------------


class TestAnswerGenerator:
    def test_generate_basic(self) -> None:
        gen_config = GenerationConfig()
        llm = DeterministicTestLLM(response_mode="generation")
        builder = PromptBuilder(gen_config)
        generator = AnswerGenerator(llm, builder, gen_config)

        result = _make_retrieval_result()
        answer = generator.generate(result)

        assert isinstance(answer, AnswerResult)
        assert answer.question == "Wikipedia là gì?"
        assert len(answer.answer) > 0

    def test_generate_has_citations(self) -> None:
        gen_config = GenerationConfig()
        llm = DeterministicTestLLM(response_mode="generation")
        builder = PromptBuilder(gen_config)
        generator = AnswerGenerator(llm, builder, gen_config)

        result = _make_retrieval_result()
        answer = generator.generate(result)

        assert len(answer.citations) >= 1
        assert answer.citations[0].chunk_id == "chunk-001"
        assert answer.citations[0].title == "Wikipedia"

    def test_generate_confidence(self) -> None:
        gen_config = GenerationConfig()
        llm = DeterministicTestLLM(response_mode="generation")
        builder = PromptBuilder(gen_config)
        generator = AnswerGenerator(llm, builder, gen_config)

        result = _make_retrieval_result()
        answer = generator.generate(result)

        assert 0.0 <= answer.confidence <= 1.0
        assert answer.confidence == 0.8  # DeterministicTestLLM returns 0.8

    def test_generate_passages_used(self) -> None:
        gen_config = GenerationConfig()
        llm = DeterministicTestLLM(response_mode="generation")
        builder = PromptBuilder(gen_config)
        generator = AnswerGenerator(llm, builder, gen_config)

        result = _make_retrieval_result()
        answer = generator.generate(result)

        assert answer.passages_used == 2

    def test_generate_fallback_on_bad_json(self) -> None:
        """When chat_json fails, fallback to chat() text mode."""
        gen_config = GenerationConfig()

        class BadJsonLLM:
            def chat(self, messages, **kwargs):
                return "Fallback text answer"

            def chat_json(self, messages, **kwargs):
                raise ValueError("Invalid JSON")

        builder = PromptBuilder(gen_config)
        generator = AnswerGenerator(BadJsonLLM(), builder, gen_config)  # type: ignore[arg-type]

        result = _make_retrieval_result()
        answer = generator.generate(result)

        assert answer.answer == "Fallback text answer"
        assert answer.citations == []
        assert answer.metadata["parse_mode"] == "fallback_text"


# ---------------------------------------------------------------------------
# OutputGuardrails tests
# ---------------------------------------------------------------------------


class TestOutputGuardrails:
    def test_safe_answer_passes(self) -> None:
        config = OutputGuardrailsConfig()
        guardrails = OutputGuardrails(config)

        answer = AnswerResult(
            question="Wikipedia là gì?",
            answer="Wikipedia là bách khoa toàn thư mở.",
            citations=[Citation(claim="test", chunk_id="c1", doc_id="d1", title="t")],
            confidence=0.8,
            passages_used=1,
        )
        result = _make_retrieval_result()

        checked = guardrails.check(answer, result)
        assert checked.metadata["guardrail_checked"] is True
        assert "unsafe_content_detected" not in checked.metadata.get("guardrail_flags", [])

    def test_hallucination_no_context(self) -> None:
        config = OutputGuardrailsConfig()
        guardrails = OutputGuardrails(config)

        answer = AnswerResult(
            question="Test?",
            answer="Some answer without context",
            citations=[],
            confidence=0.5,
            passages_used=0,
        )
        result = _make_retrieval_result(passages=[])

        checked = guardrails.check(answer, result)
        assert "hallucination_no_context" in checked.metadata["guardrail_flags"]

    def test_safety_unsafe_content(self) -> None:
        config = OutputGuardrailsConfig()
        guardrails = OutputGuardrails(config)

        answer = AnswerResult(
            question="Test?",
            answer="Nội dung có bom và vu_khí ở đây",
            citations=[Citation(claim="test", chunk_id="c1", doc_id="d1", title="t")],
            confidence=0.5,
            passages_used=1,
        )
        result = _make_retrieval_result()

        checked = guardrails.check(answer, result)
        assert "unsafe_content_detected" in checked.metadata["guardrail_flags"]

    def test_quality_too_short(self) -> None:
        config = OutputGuardrailsConfig()
        guardrails = OutputGuardrails(config)

        answer = AnswerResult(
            question="Test?",
            answer="Short",
            citations=[Citation(claim="test", chunk_id="c1", doc_id="d1", title="t")],
            confidence=0.5,
            passages_used=1,
        )
        result = _make_retrieval_result()

        checked = guardrails.check(answer, result)
        assert "answer_too_short" in checked.metadata["guardrail_flags"]

    def test_quality_insufficient_citations(self) -> None:
        config = OutputGuardrailsConfig(min_citations=2)
        guardrails = OutputGuardrails(config)

        answer = AnswerResult(
            question="Test?",
            answer="This is a longer answer that should not trigger the too-short flag.",
            citations=[Citation(claim="test", chunk_id="c1", doc_id="d1", title="t")],
            confidence=0.5,
            passages_used=1,
        )
        result = _make_retrieval_result()

        checked = guardrails.check(answer, result)
        assert "insufficient_citations" in checked.metadata["guardrail_flags"]

    def test_confidence_reduced_on_flags(self) -> None:
        config = OutputGuardrailsConfig()
        guardrails = OutputGuardrails(config)

        answer = AnswerResult(
            question="Test?",
            answer="Short",
            citations=[],
            confidence=0.8,
            passages_used=0,
        )
        result = _make_retrieval_result(passages=[])

        checked = guardrails.check(answer, result)
        assert checked.confidence < 0.8  # Confidence should be reduced


# ---------------------------------------------------------------------------
# AnswerPipeline end-to-end tests
# ---------------------------------------------------------------------------


class TestAnswerPipeline:
    def test_e2e_with_test_llm(self) -> None:
        """Full pipeline with deterministic test LLM — no API calls."""
        from rag_pipeline.indexing.embedder import DeterministicTestEmbedder
        from rag_pipeline.indexing.vector_store import InMemoryVectorStore

        gen_config = GenerationConfig()
        output_config = OutputGuardrailsConfig()
        llm = DeterministicTestLLM(response_mode="generation")

        query_pipeline = QueryPipeline(
            config=__import__("rag_pipeline.config", fromlist=["QueryConfig"]).QueryConfig(),
            normalizer=QueryNormalizer(),
            guardrails=QueryGuardrails(),
            rewriter=None,
        )

        # Build a minimal retrieval pipeline with in-memory store
        vector_store = InMemoryVectorStore()
        embedder = DeterministicTestEmbedder()

        from rag_pipeline.config import RetrievalConfig
        from rag_pipeline.indexing.bm25_index import BM25Index

        retrieval_pipeline = RetrievalPipeline(
            config=RetrievalConfig(enable_rerank=False),
            embedder=embedder,
            vector_store=vector_store,
            bm25_index=BM25Index(index_path=Path("index/bm25_test.pkl")),
        )

        prompt_builder = PromptBuilder(gen_config)
        answer_generator = AnswerGenerator(llm, prompt_builder, gen_config)
        output_guardrails = OutputGuardrails(output_config)

        pipeline = AnswerPipeline(
            query_pipeline=query_pipeline,
            retrieval_pipeline=retrieval_pipeline,
            answer_generator=answer_generator,
            output_guardrails=output_guardrails,
        )

        result = pipeline.ask("Wikipedia là gì?")

        assert isinstance(result, AnswerResult)
        assert result.question == "Wikipedia là gì?"
        assert len(result.answer) > 0
        assert result.metadata.get("guardrail_checked") is True

    def test_citation_mapping(self) -> None:
        """Citations map correctly to source passages."""
        gen_config = GenerationConfig()
        llm = DeterministicTestLLM(response_mode="generation")
        builder = PromptBuilder(gen_config)
        generator = AnswerGenerator(llm, builder, gen_config)

        retrieval_result = _make_retrieval_result()
        answer = generator.generate(retrieval_result)

        # DeterministicTestLLM returns source_index=1
        assert answer.citations[0].chunk_id == "chunk-001"
        assert answer.citations[0].doc_id == "doc-001"
        assert answer.citations[0].title == "Wikipedia"
        assert answer.citations[0].source_url == "https://vi.wikipedia.org/wiki/Wikipedia"
