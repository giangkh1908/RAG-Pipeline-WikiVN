"""Ingestion pipeline orchestrator."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable
from uuid import UUID

from rag_pipeline.chunking import ChunkingPipeline
from rag_pipeline.indexing.indexing_service import IndexingService
from rag_pipeline.ingestion.loader import VietnamTourismLoader
from rag_pipeline.storage.base import Storage
from rag_pipeline.storage.models import Document, Source


class IngestionPipeline:
    """Ingest raw JSON data into the storage layer.

    Optionally, if an ``IndexingService`` is provided, the pipeline will also
    embed and index chunks into the vector store after chunking.
    """

    def __init__(
        self,
        storage: Storage,
        chunking_pipeline: ChunkingPipeline,
        indexing_service: IndexingService | None = None,
    ) -> None:
        self.storage = storage
        self.chunking_pipeline = chunking_pipeline
        self.indexing_service = indexing_service

    def ingest_file(
        self,
        path: str | Path,
        tenant_id: str,
        source_type: str,
        source_version: str,
    ) -> Source:
        """Ingest a JSON file and return the created source."""
        loader = VietnamTourismLoader(path)
        source = Source(
            tenant_id=tenant_id,
            type=source_type,
            version=source_version,
        )
        self.storage.save_source(source)

        for topic in loader.load():
            self._ingest_topic(source.id, topic.title, topic.paragraphs)

        if self.indexing_service is not None:
            self.indexing_service.index_source(str(source.id))

        return source

    def _ingest_topic(
        self,
        source_id: UUID,
        topic_title: str,
        paragraphs: Iterable,
    ) -> None:
        for paragraph in paragraphs:
            checksum = _compute_checksum(paragraph.context)

            document = Document(
                source_id=source_id,
                checksum=checksum,
                status="processing",
                metadata={"title": topic_title},
            )
            self.storage.save_document(document)

            chunks = self.chunking_pipeline.process(
                paragraph.to_raw_document(document.id, source_id, topic_title)
            )

            for chunk in chunks:
                self.storage.save_chunk(chunk)

            document.status = "indexed"
            self.storage.save_document(document)


def _compute_checksum(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
