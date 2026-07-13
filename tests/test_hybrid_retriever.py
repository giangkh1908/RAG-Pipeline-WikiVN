"""Tests for the hybrid retriever."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from rag_pipeline.config import RetrievalConfig
from rag_pipeline.indexing import QdrantVectorStore
from rag_pipeline.indexing.models import SearchResult
from rag_pipeline.retrieval import HybridRetriever
from rag_pipeline.storage import Chunk, SQLiteStorage


@pytest.fixture
def storage() -> SQLiteStorage:
    db = SQLiteStorage(":memory:")
    yield db
    db.close()


@pytest.fixture
def retriever(storage: SQLiteStorage) -> HybridRetriever:
    config = RetrievalConfig(rrf_k=60, rrf_top_k=3)
    vector_store = MagicMock(spec=QdrantVectorStore)
    dense_embedder = MagicMock()
    sparse_embedder = MagicMock()

    dense_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    sparse_embedder.embed.return_value = [{1: 0.5, 2: 0.3}]

    return HybridRetriever(
        config=config,
        storage=storage,
        vector_store=vector_store,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )


class TestHybridRetriever:
    def test_normalize_query(self) -> None:
        assert HybridRetriever.normalize_query("  Hello   WORLD  ") == "hello world"

    def test_retrieve_fuses_and_ranks_results(
        self, retriever: HybridRetriever, storage: SQLiteStorage
    ) -> None:
        chunk_a = storage.save_chunk(
            Chunk(document_id=uuid4(), chunk_order=0, content="chunk a", token_count=2)
        )
        chunk_b = storage.save_chunk(
            Chunk(document_id=uuid4(), chunk_order=1, content="chunk b", token_count=2)
        )
        chunk_c = storage.save_chunk(
            Chunk(document_id=uuid4(), chunk_order=2, content="chunk c", token_count=2)
        )

        # Dense ranking: a > b > c
        # Sparse ranking: b > c
        retriever.vector_store.search_dense.return_value = [
            SearchResult(chunk_id=chunk_a.id, score=0.9, metadata={"source": "dense"}),
            SearchResult(chunk_id=chunk_b.id, score=0.8, metadata={"source": "dense"}),
            SearchResult(chunk_id=chunk_c.id, score=0.7, metadata={"source": "dense"}),
        ]
        retriever.vector_store.search_sparse.return_value = [
            SearchResult(chunk_id=chunk_b.id, score=0.95, metadata={"source": "sparse"}),
            SearchResult(chunk_id=chunk_c.id, score=0.85, metadata={"source": "sparse"}),
        ]

        results = retriever.retrieve("test query")

        assert len(results) == 3
        assert results[0].chunk_id == chunk_b.id
        assert results[0].rank == 1
        assert results[0].dense_score == 0.8
        assert results[0].sparse_score == 0.95
        assert results[0].content == "chunk b"

        # RRF scores should be positive and ordered descending.
        assert results[0].rrf_score > results[1].rrf_score > results[2].rrf_score

    def test_retrieve_respects_top_k(
        self, retriever: HybridRetriever, storage: SQLiteStorage
    ) -> None:
        chunks = [
            storage.save_chunk(
                Chunk(document_id=uuid4(), chunk_order=i, content=f"chunk {i}", token_count=2)
            )
            for i in range(5)
        ]

        retriever.vector_store.search_dense.return_value = [
            SearchResult(chunk_id=chunk.id, score=1.0 - i * 0.1) for i, chunk in enumerate(chunks)
        ]
        retriever.vector_store.search_sparse.return_value = []

        results = retriever.retrieve("query", top_k=2)
        assert len(results) == 2

    def test_retrieve_passes_filters_to_vector_store(
        self, retriever: HybridRetriever, storage: SQLiteStorage
    ) -> None:
        chunk = storage.save_chunk(
            Chunk(document_id=uuid4(), chunk_order=0, content="filtered chunk", token_count=2)
        )
        retriever.vector_store.search_dense.return_value = [
            SearchResult(chunk_id=chunk.id, score=0.9)
        ]
        retriever.vector_store.search_sparse.return_value = []

        retriever.retrieve("query", filters={"document_id": str(chunk.document_id)})

        retriever.vector_store.search_dense.assert_called_once()
        call_kwargs = retriever.vector_store.search_dense.call_args.kwargs
        assert call_kwargs["filters"] == {"document_id": str(chunk.document_id)}
