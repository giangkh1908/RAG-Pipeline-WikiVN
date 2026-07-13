"""Abstract storage interface."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag_pipeline.storage.models import Chunk, Document, IndexEntry, Source


class Storage(Protocol):
    """Storage layer for Source → Document → Chunk → Index."""

    # Source operations
    def save_source(self, source: Source) -> Source: ...

    def get_source(self, source_id: UUID) -> Source | None: ...

    def list_sources(self, tenant_id: str | None = None) -> list[Source]: ...

    # Document operations
    def save_document(self, document: Document) -> Document: ...

    def get_document(self, document_id: UUID) -> Document | None: ...

    def list_documents(
        self,
        source_id: UUID | None = None,
        status: str | None = None,
    ) -> list[Document]: ...

    # Chunk operations
    def save_chunk(self, chunk: Chunk) -> Chunk: ...

    def get_chunk(self, chunk_id: UUID) -> Chunk | None: ...

    def list_chunks(self, document_id: UUID) -> list[Chunk]: ...

    # Index operations
    def save_index_entry(self, entry: IndexEntry) -> IndexEntry: ...

    def get_index_entry(self, chunk_id: UUID) -> IndexEntry | None: ...

    def list_index_entries(self, chunk_ids: list[UUID] | None = None) -> list[IndexEntry]: ...
