"""Tests for the ingestion pipeline."""

from unittest.mock import MagicMock
from uuid import UUID

import pytest

from rag_pipeline.chunking import ChunkingPipeline, StructureChunker
from rag_pipeline.indexing import IndexingService
from rag_pipeline.indexing.vector_store import QdrantVectorStore
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.storage import SQLiteStorage


@pytest.fixture
def storage() -> SQLiteStorage:
    db = SQLiteStorage(":memory:")
    yield db
    db.close()


@pytest.fixture
def pipeline(storage: SQLiteStorage) -> IngestionPipeline:
    chunking = ChunkingPipeline(chunker=StructureChunker(max_tokens=200))
    return IngestionPipeline(storage=storage, chunking_pipeline=chunking)


class TestIngestionPipeline:
    def test_ingest_file_creates_source_document_and_chunks(
        self, pipeline: IngestionPipeline, storage: SQLiteStorage
    ) -> None:
        source = pipeline.ingest_file(
            path="documents/vietnam_tourism_v2.json",
            tenant_id="tenant-1",
            source_type="vietnam_tourism",
            source_version="v2",
        )

        assert isinstance(source.id, UUID)
        assert source.tenant_id == "tenant-1"
        assert source.type == "vietnam_tourism"

        documents = storage.list_documents(source_id=source.id)
        assert len(documents) > 0

        chunks = storage.list_chunks(documents[0].id)
        assert len(chunks) > 0

    def test_document_status_is_indexed_after_ingestion(
        self, pipeline: IngestionPipeline, storage: SQLiteStorage
    ) -> None:
        source = pipeline.ingest_file(
            path="documents/vietnam_tourism_v2.json",
            tenant_id="tenant-1",
            source_type="vietnam_tourism",
            source_version="v2",
        )

        documents = storage.list_documents(source_id=source.id, status="indexed")
        assert len(documents) > 0

    def test_ingest_file_with_indexing_service_creates_index_entries(
        self, storage: SQLiteStorage
    ) -> None:
        dense = MagicMock()
        sparse = MagicMock()
        vector_store = MagicMock(spec=QdrantVectorStore)
        vector_store.config = MagicMock()
        vector_store.config.collection_name = "test"

        indexing_service = IndexingService(
            storage=storage,
            vector_store=vector_store,
            dense_embedder=dense,
            sparse_embedder=sparse,
        )
        pipeline = IngestionPipeline(
            storage=storage,
            chunking_pipeline=ChunkingPipeline(chunker=StructureChunker(max_tokens=200)),
            indexing_service=indexing_service,
        )

        dense.embed.return_value = []
        sparse.embed.return_value = []

        source = pipeline.ingest_file(
            path="documents/vietnam_tourism_v2.json",
            tenant_id="tenant-1",
            source_type="vietnam_tourism",
            source_version="v2",
        )

        documents = storage.list_documents(source_id=source.id)
        chunks = storage.list_chunks(documents[0].id)

        dense.embed.assert_called_once()
        sparse.embed.assert_called_once()
        assert len(chunks) > 0
