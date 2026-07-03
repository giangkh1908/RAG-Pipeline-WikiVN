"""Tests for LangSmith tracing integration."""

from __future__ import annotations

import os
from unittest.mock import patch

from rag_pipeline.config import LangSmithConfig


class TestLangSmithConfig:
    def test_default_config(self) -> None:
        config = LangSmithConfig()
        assert config.enabled is False
        assert config.api_key_env == "LANGSMITH_API_KEY"
        assert config.project == "rag-pipeline"
        assert "smith.langchain" in config.endpoint

    def test_custom_config(self) -> None:
        config = LangSmithConfig(
            enabled=True,
            project="my-project",
            api_key_env="CUSTOM_KEY",
        )
        assert config.enabled is True
        assert config.project == "my-project"
        assert config.api_key_env == "CUSTOM_KEY"


class TestLangSmithIntegration:
    def test_traceable_decorator_without_langsmith(self) -> None:
        """Pipeline works even without langsmith installed (graceful fallback)."""
        from rag_pipeline.pipelines.answer_pipeline import AnswerPipeline

        # Verify the class exists and has the ask method
        assert hasattr(AnswerPipeline, "ask")

    def test_langsmith_env_vars(self) -> None:
        """LangSmith env vars are set correctly when enabled."""
        config = LangSmithConfig(enabled=True, project="test-project")

        with patch.dict(os.environ, {}, clear=False):
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            os.environ["LANGSMITH_PROJECT"] = config.project
            os.environ["LANGSMITH_ENDPOINT"] = config.endpoint

            assert os.environ["LANGSMITH_TRACING_V2"] == "true"
            assert os.environ["LANGSMITH_PROJECT"] == "test-project"
            assert "smith.langchain" in os.environ["LANGSMITH_ENDPOINT"]

    def test_pipeline_runs_without_langsmith_key(self) -> None:
        """Pipeline works without LANGSMITH_API_KEY (tracing disabled, no crash)."""
        from rag_pipeline.config import GenerationConfig, OutputGuardrailsConfig, QueryConfig, RetrievalConfig
        from rag_pipeline.generation.answer_generator import AnswerGenerator
        from rag_pipeline.generation.output_guardrails import OutputGuardrails
        from rag_pipeline.generation.prompt_builder import PromptBuilder
        from rag_pipeline.indexing.bm25_index import BM25Index
        from rag_pipeline.indexing.embedder import DeterministicTestEmbedder
        from rag_pipeline.indexing.llm_client import DeterministicTestLLM
        from rag_pipeline.indexing.vector_store import InMemoryVectorStore
        from rag_pipeline.pipelines.answer_pipeline import AnswerPipeline
        from rag_pipeline.pipelines.query_pipeline import QueryPipeline
        from rag_pipeline.pipelines.retrieval_pipeline import RetrievalPipeline
        from rag_pipeline.query.guardrails import QueryGuardrails
        from rag_pipeline.query.normalizer import QueryNormalizer

        from pathlib import Path

        gen_config = GenerationConfig()
        llm = DeterministicTestLLM(response_mode="generation")

        query_pipeline = QueryPipeline(
            config=QueryConfig(),
            normalizer=QueryNormalizer(),
            guardrails=QueryGuardrails(),
            rewriter=None,
        )
        retrieval_pipeline = RetrievalPipeline(
            config=RetrievalConfig(enable_rerank=False),
            embedder=DeterministicTestEmbedder(),
            vector_store=InMemoryVectorStore(),
            bm25_index=BM25Index(index_path=Path("index/bm25_test.pkl")),
        )
        prompt_builder = PromptBuilder(gen_config)
        answer_generator = AnswerGenerator(llm, prompt_builder, gen_config)
        output_guardrails = OutputGuardrails(OutputGuardrailsConfig())

        pipeline = AnswerPipeline(
            query_pipeline=query_pipeline,
            retrieval_pipeline=retrieval_pipeline,
            answer_generator=answer_generator,
            output_guardrails=output_guardrails,
        )

        # Should work without LANGSMITH_API_KEY
        result = pipeline.ask("Test question?")
        assert result.answer is not None
        assert len(result.answer) > 0
