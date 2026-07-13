"""Tests for the indexing service."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag_pipeline.config import QdrantConfig
from rag_pipeline.indexing.indexing_service import IndexingService
from rag_pipeline.indexing.vector_store import QdrantVectorStore
from rag_pipeline.storage import Chunk, Document, IndexEntry, Source, SQLiteStorage


@pytest.fixture
def storage() -> SQLiteStorage:
    db = SQLiteStorage(":memory:")
    yield db
    db.close()


@pytest.fixture
def mock_embedders() -> tuple[MagicMock, MagicMock]:
    dense = MagicMock()
    sparse = MagicMock()
    return dense, sparse


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock(spec=QdrantVectorStore)
    store.config = QdrantConfig()
    return store


class TestIndexingService:
    def test_index_source_embeds_and_upserts(
        self,
        storage: SQLiteStorage,
        mock_embedders: tuple[MagicMock, MagicMock],
        mock_vector_store: MagicMock,
    ) -> None:
        dense, sparse = mock_embedders
        dense.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        sparse.embed.return_value = [{1: 0.5}, {2: 0.6}]

        source = storage.save_source(Source(tenant_id="t1", type="test", version="v1"))
        document = storage.save_document(
            Document(source_id=source.id, checksum="abc", status="indexed")
        )
        chunk_1 = storage.save_chunk(
            Chunk(document_id=document.id, chunk_order=0, content="hello", token_count=1)
        )
        chunk_2 = storage.save_chunk(
            Chunk(document_id=document.id, chunk_order=1, content="world", token_count=1)
        )

        service = IndexingService(
            storage=storage,
            vector_store=mock_vector_store,
            dense_embedder=dense,
            sparse_embedder=sparse,
        )
        indexed_count = service.index_source(str(source.id))

        assert indexed_count == 2
        dense.embed.assert_called_once_with(["hello", "world"])
        sparse.embed.assert_called_once_with(["hello", "world"])
        mock_vector_store.upsert.assert_called_once()

        saved_entries = storage.list_index_entries()
        assert len(saved_entries) == 2
        entry_by_chunk = {entry.chunk_id: entry for entry in saved_entries}
        assert entry_by_chunk[chunk_1.id].dense_vector == [0.1, 0.2]
        assert entry_by_chunk[chunk_1.id].sparse_vector == {1: 0.5}
        assert entry_by_chunk[chunk_2.id].dense_vector == [0.3, 0.4]
        assert entry_by_chunk[chunk_2.id].sparse_vector == {2: 0.6}

    def test_index_source_skips_already_indexed_chunks(
        self,
        storage: SQLiteStorage,
        mock_embedders: tuple[MagicMock, MagicMock],
        mock_vector_store: MagicMock,
    ) -> None:
        dense, sparse = mock_embedders
        dense.embed.return_value = [[0.1, 0.2]]
        sparse.embed.return_value = [{1: 0.5}]

        source = storage.save_source(Source(tenant_id="t1", type="test", version="v1"))
        document = storage.save_document(
            Document(source_id=source.id, checksum="abc", status="indexed")
        )
        chunk_1 = storage.save_chunk(
            Chunk(document_id=document.id, chunk_order=0, content="hello", token_count=1)
        )
        storage.save_chunk(
            Chunk(document_id=document.id, chunk_order=1, content="world", token_count=1)
        )
        storage.save_index_entry(
            IndexEntry(chunk_id=chunk_1.id, dense_vector=[0.9], sparse_vector={9: 0.9})
        )

        service = IndexingService(
            storage=storage,
            vector_store=mock_vector_store,
            dense_embedder=dense,
            sparse_embedder=sparse,
        )
        indexed_count = service.index_source(str(source.id))

        assert indexed_count == 1
        dense.embed.assert_called_once_with(["world"])
