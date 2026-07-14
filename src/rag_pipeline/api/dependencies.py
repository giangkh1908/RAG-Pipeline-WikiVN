"""Shared dependencies for the RAG API."""

from __future__ import annotations

from typing import Any

from rag_pipeline.config import RAGConfig
from rag_pipeline.generation import RAGPipeline
from rag_pipeline.indexing import DenseEmbedder, QdrantVectorStore, SparseEmbedder
from rag_pipeline.retrieval import FilterBuilder, HybridRetriever, LLMQueryProcessor
from rag_pipeline.retrieval.query_cache import QueryCache
from rag_pipeline.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_pipeline.storage import ConversationStore, SQLiteStorage


class _AppState:
    """Bundle of long-lived singletons used by FastAPI dependencies."""

    storage: SQLiteStorage | None = None
    conversation_store: ConversationStore | None = None
    pipeline: RAGPipeline | None = None
    compactor: Any = None


_state = _AppState()


class PipelineStore:
    """Singleton store for the RAG pipeline."""

    @classmethod
    def get_pipeline(cls) -> RAGPipeline:
        if _state.pipeline is None:
            _build_app_state()
        assert _state.pipeline is not None
        return _state.pipeline

    @classmethod
    def close(cls) -> None:
        if _state.pipeline is not None:
            _state.pipeline.close()
            _state.pipeline = None
        if _state.compactor is not None:
            _state.compactor.close()
            _state.compactor = None
        if _state.storage is not None:
            _state.storage.close()
            _state.storage = None
        _state.conversation_store = None


def _build_app_state() -> None:
    """Build the production RAG pipeline + memory store from configuration."""
    config = RAGConfig()
    storage = SQLiteStorage(config.retrieval.storage.db_path)
    vector_store = QdrantVectorStore(config.retrieval.qdrant)
    dense_embedder = DenseEmbedder(config.retrieval.dense)
    sparse_embedder = SparseEmbedder(config.retrieval.sparse)

    cache = QueryCache(storage)
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

    from rag_pipeline.generation import (
        CitationContextBuilder,
        LLMAnswerGenerator,
        MemoryCompactor,
    )

    context_builder = CitationContextBuilder(config.context_builder)
    answer_generator = LLMAnswerGenerator(config.generation)

    conversation_store: ConversationStore | None = None
    compactor: MemoryCompactor | None = None
    if config.memory.enabled:
        conversation_store = ConversationStore(storage)
        compactor = MemoryCompactor(config.memory, conversation_store)

    pipeline = RAGPipeline.from_config(
        config=config,
        storage=storage,
        vector_store=vector_store,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        conversation_store=conversation_store,
        compactor=compactor,
    )

    _state.storage = storage
    _state.conversation_store = conversation_store
    _state.compactor = compactor
    _state.pipeline = pipeline


def get_rag_pipeline() -> Any:
    """FastAPI dependency that returns the shared RAG pipeline."""
    return PipelineStore.get_pipeline()


def get_conversation_store() -> ConversationStore | None:
    """FastAPI dependency that returns the shared conversation store (or None)."""
    if _state.conversation_store is None and _state.pipeline is None:
        _build_app_state()
    return _state.conversation_store
