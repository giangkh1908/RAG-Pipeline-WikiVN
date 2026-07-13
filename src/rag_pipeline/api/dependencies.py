"""Shared dependencies for the RAG API."""

from __future__ import annotations

from typing import Any

from rag_pipeline.config import RAGConfig
from rag_pipeline.generation import RAGPipeline
from rag_pipeline.indexing import DenseEmbedder, QdrantVectorStore, SparseEmbedder
from rag_pipeline.retrieval import FilterBuilder, HybridRetriever, LLMQueryProcessor
from rag_pipeline.retrieval.query_cache import QueryCache
from rag_pipeline.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_pipeline.storage import SQLiteStorage


class PipelineStore:
    """Singleton store for the RAG pipeline."""

    _instance: RAGPipeline | None = None

    @classmethod
    def get_pipeline(cls) -> RAGPipeline:
        if cls._instance is None:
            cls._instance = _build_pipeline()
        return cls._instance

    @classmethod
    def close(cls) -> None:
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None


def _build_pipeline() -> RAGPipeline:
    """Build the production RAG pipeline from configuration."""
    config = RAGConfig()
    storage = SQLiteStorage(config.retrieval.storage.db_path)
    vector_store = QdrantVectorStore(config.retrieval.qdrant)
    dense_embedder = DenseEmbedder(config.retrieval.dense)
    sparse_embedder = SparseEmbedder(config.retrieval.sparse)

    cache = QueryCache(config.retrieval.storage.db_path)
    llm_processor = LLMQueryProcessor(config.retrieval.llm_query, cache=cache)
    filter_builder = FilterBuilder()
    retriever = HybridRetriever(
        config=config.retrieval,
        storage=storage,
        vector_store=vector_store,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )
    retrieval_pipeline = RetrievalPipeline(llm_processor, filter_builder, retriever)

    from rag_pipeline.generation import CitationContextBuilder, LLMAnswerGenerator

    context_builder = CitationContextBuilder(config.context_builder)
    answer_generator = LLMAnswerGenerator(config.generation)

    return RAGPipeline(retrieval_pipeline, context_builder, answer_generator)


def get_rag_pipeline() -> Any:
    """FastAPI dependency that returns the shared RAG pipeline."""
    return PipelineStore.get_pipeline()
