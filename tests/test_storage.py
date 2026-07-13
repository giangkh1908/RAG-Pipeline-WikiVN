"""Tests for the storage layer."""

from uuid import UUID

import pytest

from rag_pipeline.storage import Chunk, Document, IndexEntry, Source, SQLiteStorage


@pytest.fixture
def storage() -> SQLiteStorage:
    db = SQLiteStorage(":memory:")
    yield db
    db.close()


class TestSourceStorage:
    def test_save_and_get_source(self, storage: SQLiteStorage) -> None:
        source = Source(tenant_id="tenant-1", type="wikipedia", version="1.0")
        saved = storage.save_source(source)

        assert isinstance(saved.id, UUID)
        assert storage.get_source(saved.id) == saved

    def test_list_sources_by_tenant(self, storage: SQLiteStorage) -> None:
        source_a = Source(tenant_id="tenant-1", type="wikipedia", version="1.0")
        source_b = Source(tenant_id="tenant-2", type="legal", version="2.0")
        storage.save_source(source_a)
        storage.save_source(source_b)

        results = storage.list_sources(tenant_id="tenant-1")
        assert len(results) == 1
        assert results[0].tenant_id == "tenant-1"


class TestDocumentStorage:
    def test_save_and_get_document(self, storage: SQLiteStorage) -> None:
        source = storage.save_source(Source(tenant_id="tenant-1", type="wikipedia", version="1.0"))
        document = Document(source_id=source.id, checksum="abc123", status="pending")
        saved = storage.save_document(document)

        assert saved.source_id == source.id
        assert storage.get_document(saved.id) == saved

    def test_list_documents_by_source_and_status(self, storage: SQLiteStorage) -> None:
        source = storage.save_source(Source(tenant_id="tenant-1", type="wikipedia", version="1.0"))
        doc_pending = Document(source_id=source.id, checksum="a", status="pending")
        doc_indexed = Document(source_id=source.id, checksum="b", status="indexed")
        storage.save_document(doc_pending)
        storage.save_document(doc_indexed)

        pending = storage.list_documents(source_id=source.id, status="pending")
        assert len(pending) == 1
        assert pending[0].status == "pending"


class TestChunkStorage:
    def test_save_and_list_chunks(self, storage: SQLiteStorage) -> None:
        source = storage.save_source(Source(tenant_id="tenant-1", type="wikipedia", version="1.0"))
        document = storage.save_document(Document(source_id=source.id, checksum="abc123"))

        chunk_1 = Chunk(
            document_id=document.id,
            chunk_order=0,
            content="First chunk",
            token_count=5,
        )
        chunk_2 = Chunk(
            document_id=document.id,
            chunk_order=1,
            content="Second chunk",
            token_count=5,
        )
        storage.save_chunk(chunk_1)
        storage.save_chunk(chunk_2)

        chunks = storage.list_chunks(document.id)
        assert len(chunks) == 2
        assert chunks[0].chunk_order == 0
        assert chunks[1].chunk_order == 1


class TestIndexStorage:
    def test_save_and_get_index_entry(self, storage: SQLiteStorage) -> None:
        source = storage.save_source(Source(tenant_id="tenant-1", type="wikipedia", version="1.0"))
        document = storage.save_document(Document(source_id=source.id, checksum="abc123"))
        chunk = storage.save_chunk(
            Chunk(
                document_id=document.id,
                chunk_order=0,
                content="Hello world",
                token_count=2,
            )
        )

        entry = IndexEntry(
            chunk_id=chunk.id,
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={1: 1.5, 2: 0.8},
        )
        saved = storage.save_index_entry(entry)

        assert storage.get_index_entry(chunk.id) == saved

    def test_list_index_entries_by_chunk_ids(self, storage: SQLiteStorage) -> None:
        source = storage.save_source(Source(tenant_id="tenant-1", type="wikipedia", version="1.0"))
        document = storage.save_document(Document(source_id=source.id, checksum="abc123"))
        chunk = storage.save_chunk(
            Chunk(
                document_id=document.id,
                chunk_order=0,
                content="Hello",
                token_count=1,
            )
        )
        entry = IndexEntry(chunk_id=chunk.id, dense_vector=[0.1])
        storage.save_index_entry(entry)

        results = storage.list_index_entries(chunk_ids=[chunk.id])
        assert len(results) == 1
        assert results[0].chunk_id == chunk.id
