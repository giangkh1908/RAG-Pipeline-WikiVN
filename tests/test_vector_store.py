"""Tests for QdrantVectorStore.

These tests mock the Qdrant client to avoid requiring a running server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from rag_pipeline.config import DenseEmbeddingConfig, QdrantConfig
from rag_pipeline.indexing.vector_store import QdrantVectorStore
from rag_pipeline.storage.models import IndexEntry


class TestQdrantVectorStore:
    @patch("rag_pipeline.indexing.vector_store.QdrantClient")
    def test_create_collection(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_collections.return_value.collections = []

        store = QdrantVectorStore(QdrantConfig())
        store.create_collection(dense_dim=DenseEmbeddingConfig().dense_dim)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "rag_chunks_v2"

    @patch("rag_pipeline.indexing.vector_store.QdrantClient")
    def test_upsert_index_entries(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        store = QdrantVectorStore(QdrantConfig())
        entry = IndexEntry(
            chunk_id=uuid4(),
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={1: 1.5},
            metadata={"title": "Test"},
        )
        store.upsert([entry])

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "rag_chunks_v2"
        assert len(call_kwargs["points"]) == 1
